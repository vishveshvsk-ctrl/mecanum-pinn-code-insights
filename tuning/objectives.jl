# =============================================================================
# tuning/objectives.jl — estimator-tuning objective (block-specific slot #3)
# =============================================================================
module TuningObjectivesMod

using LinearAlgebra
using Statistics: mean, median

using Main.TuningHarnessMod: EstimatorLog

export estimator_objective, estimator_objective_by_mode, estimator_objective_abs

function _weighted_rms(y, w)
    sw = sum(w)
    sw < 1e-12 && return 0.0
    return sqrt(sum(w .* y.^2) / sw)
end

function _per_channel_nrmse(y_hat::AbstractMatrix, y_true::AbstractMatrix, w::AbstractVector)
    n_ch = size(y_hat, 1)
    n_ch == size(y_true, 1) || error("channel count mismatch")
    n_ch == 3 || error("expected 3 channels")
    vals = Float64[_weighted_rms(y_hat[i,:] .- y_true[i,:], w) /
                   max(_weighted_rms(y_true[i,:], w), 1e-12) for i in 1:n_ch]
    return mean(vals)
end

"""
    estimator_objective(logs; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5) -> NamedTuple

Reduce per-trajectory `EstimatorLog`s to a scalar score and sub-metrics.

  overall_nrmse  :: velocity NRMSE, weighted by `1 + λ_slip * normalized_slip`,
                    averaged over channels and trajectories.
  inslip_nrmse   :: same as `overall_nrmse` but only samples where
                    `slip > median(slip)`.
  pose_drift     :: terminal pose error normalised by path length, averaged over
                    PosRef entries; 0.0 for VelRef-only log sets.
  smoothness     :: mean squared rate of change of `|v_hat|`.
  score          :: overall_nrmse + inslip_nrmse + λ_pose*pose_drift
                    + λ_smooth*smoothness (scalar to minimise).
"""
function estimator_objective(logs::Vector{EstimatorLog};
                             λ_slip::Real=2.0,
                             λ_smooth::Real=0.1,
                             λ_pose::Real=0.5)
    n = length(logs)
    n == 0 && return (
        overall_nrmse = Inf,
        inslip_nrmse  = Inf,
        pose_drift    = Inf,
        smoothness    = Inf,
        score         = Inf,
    )

    overall_sum = 0.0
    inslip_sum  = 0.0
    smooth_sum  = 0.0
    pose_sum    = 0.0
    n_poseref   = 0

    for log in logs
        slip_mean = max(mean(log.slip), 1e-6)
        slip_norm = log.slip ./ slip_mean
        w = 1.0 .+ λ_slip .* slip_norm

        ov = _per_channel_nrmse(log.v_hat, log.v_true, w)
        overall_sum += ov

        in_mask = log.slip .> median(log.slip)
        if any(in_mask)
            w_in = w[in_mask]
            inslip_sum += _per_channel_nrmse(log.v_hat[:,in_mask], log.v_true[:,in_mask], w_in)
        else
            inslip_sum += ov
        end

        # Smoothness: mean squared finite-difference rate of |v_hat|.
        if length(log.time) > 1
            v_mag = [norm(log.v_hat[:,i]) for i in axes(log.v_hat, 2)]
            dv = diff(v_mag)
            dt = diff(log.time)
            rates = dv ./ max.(dt, 1e-6)
            smooth_sum += mean(rates.^2)
        end

        # Pose drift for PosRef entries only.
        if log.ref_type == :posref
            terminal_err = norm(log.pose_true[:,end] .- log.pose_hat[:,end])
            v_mag_true = [norm(log.v_true[:,i]) for i in axes(log.v_true, 2)]
            path_len = sum(v_mag_true[2:end] .* diff(log.time))
            pose_sum += terminal_err / max(path_len, 1e-3)
            n_poseref += 1
        end
    end

    overall_nrmse = overall_sum / n
    inslip_nrmse  = inslip_sum / n
    smoothness    = smooth_sum / n
    pose_drift    = n_poseref > 0 ? pose_sum / n_poseref : 0.0

    score = overall_nrmse + inslip_nrmse + λ_pose * pose_drift + λ_smooth * smoothness

    return (
        overall_nrmse = overall_nrmse,
        inslip_nrmse  = inslip_nrmse,
        pose_drift    = pose_drift,
        smoothness    = smoothness,
        score         = score,
    )
end

"""
    estimator_objective_by_mode(logs; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5) -> NamedTuple

Mode-stratified version of `estimator_objective`.  Returns the overall metrics
plus separate VelRef and PosRef aggregates, so the two estimators can be
compared on the regimes they will actually serve.

  overall_*        :: metrics across all entries
  velref_overall   :: velocity NRMSE averaged over VelRef entries
  velref_inslip    :: in-slip velocity NRMSE averaged over VelRef entries
  posref_overall   :: velocity NRMSE averaged over PosRef entries
  posref_pose_rmse :: RMS pose error √(x²+y²+ψ²) averaged over PosRef entries
  posref_pose_drift:: terminal pose drift / path length averaged over PosRef entries
  score            :: scalar to minimise (same blend as `estimator_objective`)
"""
function estimator_objective_by_mode(logs::Vector{EstimatorLog};
                                     λ_slip::Real=2.0,
                                     λ_smooth::Real=0.1,
                                     λ_pose::Real=0.5)
    n = length(logs)
    overall = estimator_objective(logs; λ_slip=λ_slip, λ_smooth=λ_smooth, λ_pose=λ_pose)
    n == 0 && return merge(overall,
        (velref_overall=NaN, velref_inslip=NaN,
         posref_overall=NaN, posref_pose_rmse=NaN, posref_pose_drift=NaN))

    velref_logs = filter(l -> l.ref_type == :velref, logs)
    posref_logs = filter(l -> l.ref_type == :posref, logs)

    function _vel_metrics(logs_subset)
        isempty(logs_subset) && return (overall=NaN, inslip=NaN)
        ov_sum = 0.0
        in_sum = 0.0
        for log in logs_subset
            slip_mean = max(mean(log.slip), 1e-6)
            w = 1.0 .+ λ_slip .* (log.slip ./ slip_mean)
            ov_sum += _per_channel_nrmse(log.v_hat, log.v_true, w)
            in_mask = log.slip .> median(log.slip)
            if any(in_mask)
                w_in = w[in_mask]
                in_sum += _per_channel_nrmse(log.v_hat[:,in_mask], log.v_true[:,in_mask], w_in)
            else
                in_sum += ov_sum
            end
        end
        return (overall=ov_sum/length(logs_subset), inslip=in_sum/length(logs_subset))
    end

    function _pose_metrics(logs_subset)
        isempty(logs_subset) && return (overall=NaN, pose_rmse=NaN, pose_drift=NaN)
        ov_sum = 0.0
        rmse_sum = 0.0
        drift_sum = 0.0
        for log in logs_subset
            slip_mean = max(mean(log.slip), 1e-6)
            w = 1.0 .+ λ_slip .* (log.slip ./ slip_mean)
            ov_sum += _per_channel_nrmse(log.v_hat, log.v_true, w)
            pose_err = sqrt.((log.pose_true[1,:] .- log.pose_hat[1,:]).^2 .+
                             (log.pose_true[2,:] .- log.pose_hat[2,:]).^2 .+
                             (atan.(sin.(log.pose_true[3,:] .- log.pose_hat[3,:]),
                                    cos.(log.pose_true[3,:] .- log.pose_hat[3,:]))).^2)
            rmse_sum += sqrt(mean(pose_err.^2))
            terminal_err = norm(log.pose_true[:,end] .- log.pose_hat[:,end])
            v_mag_true = [norm(log.v_true[:,i]) for i in axes(log.v_true, 2)]
            path_len = sum(v_mag_true[2:end] .* diff(log.time))
            drift_sum += terminal_err / max(path_len, 1e-3)
        end
        nsub = length(logs_subset)
        return (overall=ov_sum/nsub, pose_rmse=rmse_sum/nsub, pose_drift=drift_sum/nsub)
    end

    vel = _vel_metrics(velref_logs)
    pos = _pose_metrics(posref_logs)

    return (
        overall_nrmse      = overall.overall_nrmse,
        inslip_nrmse       = overall.inslip_nrmse,
        pose_drift         = overall.pose_drift,
        smoothness         = overall.smoothness,
        score              = overall.score,
        velref_overall     = vel.overall,
        velref_inslip      = vel.inslip,
        posref_overall     = pos.overall,
        posref_pose_rmse   = pos.pose_rmse,
        posref_pose_drift  = pos.pose_drift,
    )
end

_rms(x) = sqrt(mean(x .^ 2))
_wrap_diff(a, b) = atan.(sin.(a .- b), cos.(a .- b))

"""
    estimator_objective_abs(logs; v_tol, rate_tol, pos_tol, heading_tol,
                            λ_slip, λ_smooth) -> NamedTuple

Absolute-tolerance objective: plain RMS errors in physical units, normalised by
target tolerances (score ≈ 1.0 means "at target"), NOT by signal RMS — small
commanded values are not inflated (per user spec: position ~1 cm, velocity
~1 mm/s, heading ~0.01 rad).

  vel_rmse      :: mean of per-channel velocity RMS error [m/s] over all logs
  inslip_vel_rmse:: same, samples with slip > median(slip) [m/s]
  rate_rmse     :: yaw-rate RMS error [rad/s]
  pos_rmse      :: horizontal position RMS error [m], PosRef logs only (0 else)
  heading_rmse  :: heading RMS error [rad], PosRef logs only (0 else)
  smoothness    :: mean squared rate of change of |v_hat| (jitter proxy)
  score         :: mean over logs of (vel_ratio + rate_ratio + inslip_ratio)
                   + mean over PosRef logs of (pos_ratio + heading_ratio)
                   + λ_smooth * smoothness   (scalar to minimise)
"""
function estimator_objective_abs(logs::Vector{EstimatorLog};
                                 v_tol::Real=1e-3,
                                 rate_tol::Real=1e-2,
                                 pos_tol::Real=1e-2,
                                 heading_tol::Real=1e-2,
                                 λ_slip::Real=1.0,
                                 λ_smooth::Real=0.05)
    n = length(logs)
    n == 0 && return (
        vel_rmse=Inf, inslip_vel_rmse=Inf, rate_rmse=Inf,
        pos_rmse=Inf, heading_rmse=Inf, smoothness=Inf, score=Inf,
    )

    vel_sum = 0.0
    inslip_sum = 0.0
    rate_sum = 0.0
    smooth_sum = 0.0
    pos_sum = 0.0
    head_sum = 0.0
    n_pos = 0

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

        # Pose estimation error: include for all logs so pose-aided VelRef tuning
        # penalises ESKF pose drift, not only PosRef entries.
        e_x = log.pose_hat[1,:] .- log.pose_true[1,:]
        e_y = log.pose_hat[2,:] .- log.pose_true[2,:]
        e_h = _wrap_diff(log.pose_hat[3,:], log.pose_true[3,:])
        pos_sum  += _rms(sqrt.(e_x.^2 .+ e_y.^2))
        head_sum += _rms(e_h)
        n_pos += 1
    end

    vel_rmse    = vel_sum / n
    inslip_rmse = inslip_sum / n
    rate_rmse   = rate_sum / n
    smoothness  = smooth_sum / n
    pos_rmse    = n_pos > 0 ? pos_sum / n_pos : 0.0
    heading_rmse= n_pos > 0 ? head_sum / n_pos : 0.0

    score = vel_rmse / v_tol +
            rate_rmse / rate_tol +
            λ_slip * inslip_rmse / v_tol +
            pos_rmse / pos_tol + heading_rmse / heading_tol +
            λ_smooth * smoothness

    return (
        vel_rmse        = vel_rmse,
        inslip_vel_rmse = inslip_rmse,
        rate_rmse       = rate_rmse,
        pos_rmse        = pos_rmse,
        heading_rmse    = heading_rmse,
        smoothness      = smoothness,
        score           = score,
    )
end

end # module
