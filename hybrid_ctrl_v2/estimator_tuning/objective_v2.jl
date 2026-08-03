# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/objective_v2.jl — EstimatorObjectiveV2Mod
# =============================================================================
# Ports `tuning/objectives.jl`'s `estimator_objective_abs` (absolute-tolerance
# scoring: score ~= 1.0 means "at target", not normalised by signal RMS) to
# `HarnessV2Mod.EstimatorLogV2`, then wraps it as a `theta -> NamedTuple`
# closure over `run_and_log_replay_v2` — the same shape
# `StageOptimizerMod.optimize_staged` (hybrid_ctrl_v2/controller_tuning/) already
# expects, so that generic optimizer is reused unmodified.
#
# `tuning/objectives.jl` is NEVER edited — this is a new, additive module.
# Must be `include`d after harness_v2.jl, param_space_v2.jl, sensors_v2.jl.
# =============================================================================
module EstimatorObjectiveV2Mod

using LinearAlgebra
using Statistics: mean, median

export estimator_objective_abs_v2, make_replay_objective_v2

_rms(x) = sqrt(mean(x .^ 2))
_wrap_diff(a, b) = atan.(sin.(a .- b), cos.(a .- b))

"""
    estimator_objective_abs_v2(logs; v_tol, rate_tol, pos_tol, heading_tol,
                              λ_slip, λ_smooth) -> NamedTuple

Identical formula to `TuningObjectivesMod.estimator_objective_abs`, operating
on `EstimatorLogV2`. Absolute physical-unit RMS errors normalised by target
tolerances (position ~1cm, velocity ~1mm/s, heading ~0.01 rad -- so score
terms near 1.0 mean "at target"), NOT by signal RMS.
"""
function estimator_objective_abs_v2(logs::Vector;
                                    v_tol::Real=1e-3, rate_tol::Real=1e-2,
                                    pos_tol::Real=1e-2, heading_tol::Real=1e-2,
                                    λ_slip::Real=1.0, λ_smooth::Real=0.05)
    n = length(logs)
    n == 0 && return (vel_rmse=Inf, inslip_vel_rmse=Inf, rate_rmse=Inf,
                      pos_rmse=Inf, heading_rmse=Inf, smoothness=Inf, score=Inf)

    vel_sum = 0.0; inslip_sum = 0.0; rate_sum = 0.0
    smooth_sum = 0.0; pos_sum = 0.0; head_sum = 0.0

    for log in logs
        e_vx = log.v_hat[1,:] .- log.v_true[1,:]
        e_vy = log.v_hat[2,:] .- log.v_true[2,:]
        e_ps = log.v_hat[3,:] .- log.v_true[3,:]

        vel_sum  += 0.5 * (_rms(e_vx) + _rms(e_vy))
        rate_sum += _rms(e_ps)

        in_mask = log.slip .> median(log.slip)
        if any(in_mask)
            inslip_sum += 0.5 * (_rms(e_vx[in_mask]) + _rms(e_vy[in_mask]))
        else
            inslip_sum += 0.5 * (_rms(e_vx) + _rms(e_vy))
        end

        if length(log.time) > 1
            v_mag = [norm(log.v_hat[:,i]) for i in axes(log.v_hat, 2)]
            rates = diff(v_mag) ./ max.(diff(log.time), 1e-6)
            smooth_sum += mean(rates.^2)
        end

        e_x = log.pose_hat[1,:] .- log.pose_true[1,:]
        e_y = log.pose_hat[2,:] .- log.pose_true[2,:]
        e_h = _wrap_diff(log.pose_hat[3,:], log.pose_true[3,:])
        pos_sum  += _rms(sqrt.(e_x.^2 .+ e_y.^2))
        head_sum += _rms(e_h)
    end

    vel_rmse     = vel_sum / n
    inslip_rmse  = inslip_sum / n
    rate_rmse    = rate_sum / n
    smoothness   = smooth_sum / n
    pos_rmse     = pos_sum / n
    heading_rmse = head_sum / n

    score = vel_rmse / v_tol + rate_rmse / rate_tol + λ_slip * inslip_rmse / v_tol +
            pos_rmse / pos_tol + heading_rmse / heading_tol + λ_smooth * smoothness

    return (vel_rmse=vel_rmse, inslip_vel_rmse=inslip_rmse, rate_rmse=rate_rmse,
            pos_rmse=pos_rmse, heading_rmse=heading_rmse, smoothness=smoothness, score=score)
end

const _FAILED_METRICS_V2 = (vel_rmse=Inf, inslip_vel_rmse=Inf, rate_rmse=Inf,
                            pos_rmse=Inf, heading_rmse=Inf, smoothness=Inf, score=Inf)

"""
    make_replay_objective_v2(space, trajs; kwargs...) -> Function

Build a `theta::Vector{Float64} -> NamedTuple` closure, in the exact shape
`StageOptimizerMod.optimize_staged` expects (a `.score` field it minimises).
Each call: decode `theta` via `apply_params_v2!`, build a FRESH `SensorSuite`
per trajectory (independent per-modality RNG streams, same `seed` every call
so noise realizations are held fixed across candidates -- only the estimator
parameters vary), replay via `run_and_log_replay_v2` (no ODE solve), then
score with `estimator_objective_abs_v2`.

On ANY replay failure for ANY trajectory, returns the `Inf`-score sentinel
(mirroring `StageObjectiveMod`'s failed-trajectory contract) -- a candidate
that can't even replay cleanly is worse than any candidate that can.
"""
function make_replay_objective_v2(space, trajs;
                                  seed::Int=42, sensor_kind::Symbol=:default,
                                  flow::Bool=true, fix_tier::Symbol=:docking,
                                  v_tol::Real=1e-3, rate_tol::Real=1e-2,
                                  pos_tol::Real=1e-2, heading_tol::Real=1e-2,
                                  λ_slip::Real=1.0, λ_smooth::Real=0.05)
    return function (theta::Vector{Float64})
        est_cfg = Main.ParamSpaceV2Mod.apply_params_v2!(theta, space)

        logs = Main.HarnessV2Mod.EstimatorLogV2[]
        per_traj = Dict{String,NamedTuple}()
        n_fail = 0

        for tr in trajs
            key = "$(tr.name)_c$(tr.combo_idx)_mu$(tr.mu)"
            log = try
                suite = Main.SensorModV2.build_suite(sensor_kind; seed=seed, flow=flow, fix_tier=fix_tier)
                Main.HarnessV2Mod.run_and_log_replay_v2(est_cfg, tr, suite; seed=seed)
            catch e
                @warn "EstimatorObjectiveV2Mod: replay failed for $key" exception = e
                n_fail += 1
                nothing
            end
            if log !== nothing
                push!(logs, log)
                m1 = estimator_objective_abs_v2([log]; v_tol=v_tol, rate_tol=rate_tol,
                    pos_tol=pos_tol, heading_tol=heading_tol, λ_slip=λ_slip, λ_smooth=λ_smooth)
                per_traj[key] = m1
            else
                per_traj[key] = _FAILED_METRICS_V2
            end
        end

        if n_fail > 0
            return merge(_FAILED_METRICS_V2, (per_traj=per_traj, n_fail=n_fail))
        end

        m = estimator_objective_abs_v2(logs; v_tol=v_tol, rate_tol=rate_tol,
            pos_tol=pos_tol, heading_tol=heading_tol, λ_slip=λ_slip, λ_smooth=λ_smooth)
        return merge(m, (per_traj=per_traj, n_fail=n_fail))
    end
end

end # module
