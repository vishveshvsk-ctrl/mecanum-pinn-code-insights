#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_iae.jl — CLI entry point
# (instructions/estimator-v2-iae-adaptive.md §7)
# =============================================================================
# Tunes `ESKFIAEEstimatorV2`'s 12-dim param space (10 shared v2 dims +
# `tau_iae`, `kappa_iae`) by replaying the candidate filter over pre-simulated
# Arrow trajectories. Mirrors `run_estimator_replay.jl` exactly.
#
# Warm-start from the v2 baseline optimum via `--init-from`:
#   --init-from hybrid_ctrl_v2/runs_estimator_v2_replay/seedS/best_config.json
# The 10 shared dims are seeded at the baseline optimum, the two IAE dims at
# their defaults (τ_iae=0.5, κ_iae=0.5).
#
#   --seed        RNG seed (measurement noise + optimizer start offset)
#   --p1-cap      dxNES eval cap (phase 1)
#   --p2-cap      BOBYQA eval cap (phase 2)
#   --out         output root
#   --smoke       one default-gains eval, no optimization (fast sanity check)
#   --init-from   path to baseline-v2 best_config.json (optional warm start)
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2_iae.jl")); using .EstimatorModV2IAE
include(joinpath(ROOT, "tuning", "param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2_iae.jl")); using .ParamSpaceV2IAEMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2_iae.jl")); using .HarnessV2IAEMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl")); using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "optimizer_stage.jl")); using .StageOptimizerMod
using JSON

function parse_args(argv)
    a = Dict{String,Any}("seed" => 1, "p1-cap" => 250, "p2-cap" => 60,
                         "out" => "hybrid_ctrl_v2/runs_estimator_v2_iae_replay",
                         "smoke" => false, "init-from" => nothing,
                         "rel-tol" => 1e-3, "window" => 20,
                         "refiner" => "bobyqa", "refine-halfwidth" => 0.25)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p1-cap"; a["p1-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p2-cap"; a["p2-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        elseif arg == "--smoke"; a["smoke"] = true; i += 1
        elseif arg == "--init-from"; a["init-from"] = argv[i+1]; i += 2
        else; error("run_estimator_replay_iae.jl: unknown arg $arg"); end
    end
    return a
end

"""
    make_replay_objective_v2_iae(space, trajs; kwargs...)

Identical shape to `EstimatorObjectiveV2Mod.make_replay_objective_v2`, but
decodes via `ParamSpaceV2IAEMod.apply_params_v2_iae!` and replays via
`HarnessV2IAEMod.run_and_log_replay_v2_iae`.
"""
function make_replay_objective_v2_iae(space, trajs;
                                      seed::Int=42, sensor_kind::Symbol=:default,
                                      flow::Bool=true, fix_tier::Symbol=:docking,
                                      v_tol::Real=1e-3, rate_tol::Real=1e-2,
                                      pos_tol::Real=1e-2, heading_tol::Real=1e-2,
                                      λ_slip::Real=1.0, λ_smooth::Real=0.05)
    return function (theta::Vector{Float64})
        est_cfg = Main.ParamSpaceV2IAEMod.apply_params_v2_iae!(theta, space)

        logs = Main.HarnessV2Mod.EstimatorLogV2[]
        per_traj = Dict{String,NamedTuple}()
        n_fail = 0

        for tr in trajs
            key = "$(tr.name)_c$(tr.combo_idx)_mu$(tr.mu)"
            log = try
                suite = Main.SensorModV2.build_suite(sensor_kind; seed=seed, flow=flow, fix_tier=fix_tier)
                Main.HarnessV2IAEMod.run_and_log_replay_v2_iae(est_cfg, tr, suite; seed=seed)
            catch e
                @warn "run_estimator_replay_iae: replay failed for $key" exception = e
                n_fail += 1
                nothing
            end
            if log !== nothing
                push!(logs, log)
                m1 = Main.EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log];
                    v_tol=v_tol, rate_tol=rate_tol, pos_tol=pos_tol,
                    heading_tol=heading_tol, λ_slip=λ_slip, λ_smooth=λ_smooth)
                per_traj[key] = m1
            else
                per_traj[key] = Main.EstimatorObjectiveV2Mod._FAILED_METRICS_V2
            end
        end

        if n_fail > 0
            return merge(Main.EstimatorObjectiveV2Mod._FAILED_METRICS_V2,
                         (per_traj=per_traj, n_fail=n_fail))
        end

        m = Main.EstimatorObjectiveV2Mod.estimator_objective_abs_v2(logs;
            v_tol=v_tol, rate_tol=rate_tol, pos_tol=pos_tol,
            heading_tol=heading_tol, λ_slip=λ_slip, λ_smooth=λ_smooth)
        return merge(m, (per_traj=per_traj, n_fail=n_fail))
    end
end

function run_smoke(space, trajs)
    println("=== SMOKE: default-gains replay eval on $(length(trajs)) trajectories ===")
    obj = make_replay_objective_v2_iae(space, trajs; seed=1)
    lo, hi = space.flat_lower, space.flat_upper
    theta0 = (lo .+ hi) ./ 2.0
    r = obj(theta0)
    println("  score=$(round(r.score,digits=4)) vel_rmse=$(round(r.vel_rmse,digits=6)) " *
            "rate_rmse=$(round(r.rate_rmse,digits=6)) pos_rmse=$(round(r.pos_rmse,digits=6)) " *
            "heading_rmse=$(round(r.heading_rmse,digits=6)) n_fail=$(r.n_fail)")
    r.n_fail == 0 || error("run_estimator_replay_iae.jl --smoke: $(r.n_fail)/$(length(trajs)) trajectories failed")
    println("=== SMOKE OK ===")
end

"""
    warm_start_theta(space, baseline_path)

Build a 12-dim theta0 from a baseline-v2 `best_config.json`. The 10 shared
log-scale dims are converted back to log-space and clamped to the IAE space
bounds; `tau_iae` starts at log(0.5); `kappa_iae` starts at 0.5 (linear).
"""
function warm_start_theta(space, baseline_path)
    gains = JSON.parsefile(baseline_path)["best_gains"]
    base_space = Main.ParamSpaceV2Mod.eskf_param_space_v2()

    theta0 = Vector{Float64}(undef, Main.TuningParamSpaceMod.n_params(space))
    # Map each baseline gain to the corresponding position in the new space.
    name_to_idx = Dict(d.name => i for (i, d) in enumerate(space.dims))
    for (i, d) in enumerate(base_space.dims)
        v = Float64(get(gains, d.name, exp((d.lower + d.upper) / 2)))
        idx = name_to_idx[d.name]
        theta_v = d.scale == :log ? log(v) : v
        theta0[idx] = clamp(theta_v, space.flat_lower[idx], space.flat_upper[idx])
    end

    # tau_iae: log-scale, default 0.5 s
    theta0[name_to_idx["tau_iae"]] = clamp(log(0.5),
        space.flat_lower[name_to_idx["tau_iae"]], space.flat_upper[name_to_idx["tau_iae"]])
    # kappa_iae: linear, default 0.5
    theta0[name_to_idx["kappa_iae"]] = clamp(0.5,
        space.flat_lower[name_to_idx["kappa_iae"]], space.flat_upper[name_to_idx["kappa_iae"]])

    return theta0
end

function main()
    a = parse_args(ARGS)
    space = eskf_iae_param_space_v2()
    trajs = replay_trajset()

    if a["smoke"]
        run_smoke(space, trajs)
        return
    end

    lo, hi = space.flat_lower, space.flat_upper
    obj = make_replay_objective_v2_iae(space, trajs; seed=a["seed"])

    outdir = joinpath(a["out"], "seed$(a["seed"])")
    mkpath(outdir)

    if a["init-from"] !== nothing
        println("[eskf_iae_v2 replay seed$(a["seed"])] warm-start from $(a["init-from"])")
        theta0 = warm_start_theta(space, a["init-from"])
    else
        theta0 = (lo .+ hi) ./ 2.0
    end

    println("\n===== ESKFIAEEstimatorV2 replay tuning / seed $(a["seed"]) — $(length(trajs)) trajectories =====")
    best, best_score, trials, diag = optimize_staged(obj, obj, lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"),
        warm_start=a["init-from"] !== nothing ? theta0 : nothing)

    kw = apply_params_v2_iae!(best, space)
    gains = Dict(string(k) => v for (k, v) in pairs(kw) if v isa Real)

    open(joinpath(outdir, "best_config.json"), "w") do io
        JSON.print(io, Dict(
            "seed" => a["seed"], "best_score" => best_score, "best_gains" => gains,
            "converged" => diag.converged, "stop_reason" => string(diag.stop_reason),
            "phase1_evals" => diag.phase1_evals, "phase2_evals" => diag.phase2_evals,
            "n_trajectories" => length(trajs),
        ), 2)
    end
    open(joinpath(outdir, "trials.json"), "w") do io
        JSON.print(io, [Dict("iter"=>t.iter, "tier"=>string(t.tier), "score"=>t.score,
                             "vel_rmse"=>get(t, :vel_rmse, missing), "pos_rmse"=>get(t, :pos_rmse, missing))
                        for t in trials], 2)
    end
    println("[eskf_iae_v2 replay seed$(a["seed"])] best score=$(round(best_score,digits=4)) " *
            "stop_reason=$(diag.stop_reason) evals=$(diag.phase1_evals + diag.phase2_evals) -> $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
