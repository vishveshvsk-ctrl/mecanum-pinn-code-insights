#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_slipobs.jl — CLI entry point
# =============================================================================
# Tunes `ESKFSlipObsEstimatorV2` by replaying pre-simulated Arrow trajectories.
# Mirrors `run_estimator_replay.jl` and reuses the same two-phase optimizer
# (dxNES global → BOBYQA local).  The 10 shared ESKF dims can be warm-started
# from a baseline `best_config.json`; the observer dims start at their defaults.
#
#   --seed        RNG seed (measurement noise + optimizer start offset)
#   --observer    smo | eso          (selects the param space)
#   --init-from   path to baseline best_config.json (warm start of shared dims)
#   --p1-cap      dxNES eval cap
#   --p2-cap      BOBYQA eval cap
#   --out         output root
#   --smoke       one default-gains eval, no optimization
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl"));            using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl"));     using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));         using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl"));     using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));       using .EstimatorObjectiveV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2_slipobs.jl"));                  using .EstimatorModV2SlipObs
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2_slipobs.jl")); using .HarnessV2SlipObsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2_slipobs.jl")); using .ParamSpaceV2SlipObsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "optimizer_stage.jl")); using .StageOptimizerMod
using JSON

function parse_args(argv)
    a = Dict{String,Any}(
        "seed" => 1, "p1-cap" => 250, "p2-cap" => 60,
        "out" => "hybrid_ctrl_v2/runs_estimator_v2_slipobs_replay",
        "observer" => "smo", "init-from" => "", "smoke" => false,
        "rel-tol" => 1e-3, "window" => 20, "refiner" => "bobyqa",
        "refine-halfwidth" => 0.25)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p1-cap"; a["p1-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p2-cap"; a["p2-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        elseif arg == "--observer"; a["observer"] = argv[i+1]; i += 2
        elseif arg == "--init-from"; a["init-from"] = argv[i+1]; i += 2
        elseif arg == "--smoke"; a["smoke"] = true; i += 1
        else; error("run_estimator_replay_slipobs.jl: unknown arg $arg"); end
    end
    return a
end

"""
    _theta_from_baseline(init_path, space, kind) -> Vector{Float64}

Warm-start vector: the 10 shared dims are taken from the baseline
`best_config.json` (converted back to log-space and clamped to the space
bounds); the observer dims are set to their default values, also in log-space.
"""
function _theta_from_baseline(init_path::AbstractString, space, kind::Symbol)
    isfile(init_path) || error("Init-from file not found: $init_path")
    d = JSON.parsefile(init_path)
    gains = get(d, "best_gains", Dict{String,Any}())
    defaults = Dict(
        "P0_vel" => 1e-2, "P0_yaw" => 1e-2, "P0_heading" => 1e-2,
        "P0_bias_acc" => 1e-2, "P0_bias_gyro" => 1e-2, "P0_slip" => 1e-5,
        "P0_pos" => 1.0, "pose_Qn_heading" => 1e-6, "pose_Qn_pos" => 1e-6,
        "slip_R_inflate" => 10.0,
    )
    g(k) = Float64(get(gains, k, defaults[k]))
    shared_dec = [g("P0_vel"), g("P0_yaw"), g("P0_heading"),
                  g("P0_bias_acc"), g("P0_bias_gyro"), g("P0_slip"),
                  g("P0_pos"), g("pose_Qn_heading"), g("pose_Qn_pos"),
                  g("slip_R_inflate")]
    shared_dec[shared_dec .<= 0.0] .= defaults["P0_vel"]  # safety against bad JSON

    base_space = Main.ParamSpaceV2Mod.eskf_param_space_v2()
    shared_theta = clamp.(log.(shared_dec), base_space.flat_lower, base_space.flat_upper)

    if kind == :smo
        obs_dec = Float64[0.1, 5.0, 0.042]
        obs_lo  = log.(Float64[0.01, 0.5, 1e-3])
        obs_hi  = log.(Float64[2.0, 50.0, 0.2])
    elseif kind == :eso
        obs_dec = Float64[30.0, 0.042]
        obs_lo  = log.(Float64[5.0, 1e-3])
        obs_hi  = log.(Float64[200.0, 0.2])
    else
        error("_theta_from_baseline: unknown observer kind $kind")
    end
    obs_theta = clamp.(log.(obs_dec), obs_lo, obs_hi)
    return vcat(shared_theta, obs_theta)
end

"""
    make_replay_objective_v2_slipobs(space, trajs; observer_kind, kwargs...) -> Function

`theta -> NamedTuple` closure matching `StageOptimizerMod.optimize_staged`.
Decodes with the variant-specific param-space decoder, builds a fresh sensor
suite per trajectory, replays via `run_and_log_replay_v2_slipobs`, and scores
with `estimator_objective_abs_v2`.
"""
function make_replay_objective_v2_slipobs(space, trajs;
                                          observer_kind::Symbol,
                                          seed::Int=42, sensor_kind::Symbol=:default,
                                          flow::Bool=true, fix_tier::Symbol=:docking,
                                          v_tol::Real=1e-3, rate_tol::Real=1e-2,
                                          pos_tol::Real=1e-2, heading_tol::Real=1e-2,
                                          λ_slip::Real=1.0, λ_smooth::Real=0.05)
    apply_fn = if observer_kind == :smo
        ParamSpaceV2SlipObsMod.apply_params_v2_smo!
    elseif observer_kind == :eso
        ParamSpaceV2SlipObsMod.apply_params_v2_eso!
    else
        error("make_replay_objective_v2_slipobs: unknown observer_kind=$observer_kind")
    end

    return function (theta::Vector{Float64})
        est_cfg = apply_fn(theta, space)

        logs = Main.HarnessV2Mod.EstimatorLogV2[]
        per_traj = Dict{String,NamedTuple}()
        n_fail = 0

        for tr in trajs
            key = "$(tr.name)_c$(tr.combo_idx)_mu$(tr.mu)"
            log = try
                suite = Main.SensorModV2.build_suite(sensor_kind; seed=seed, flow=flow, fix_tier=fix_tier)
                Main.HarnessV2SlipObsMod.run_and_log_replay_v2_slipobs(est_cfg, tr, suite; seed=seed)
            catch e
                @warn "EstimatorObjectiveV2SlipObsMod: replay failed for $key" exception = e
                n_fail += 1
                nothing
            end
            if log !== nothing
                push!(logs, log)
                m1 = EstimatorObjectiveV2Mod.estimator_objective_abs_v2(
                    [log]; v_tol=v_tol, rate_tol=rate_tol, pos_tol=pos_tol,
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

        m = EstimatorObjectiveV2Mod.estimator_objective_abs_v2(
            logs; v_tol=v_tol, rate_tol=rate_tol, pos_tol=pos_tol,
            heading_tol=heading_tol, λ_slip=λ_slip, λ_smooth=λ_smooth)
        return merge(m, (per_traj=per_traj, n_fail=n_fail))
    end
end

function run_smoke(space, trajs, kind)
    println("=== SMOKE: default-gains replay eval on $(length(trajs)) trajectories (observer=$kind) ===")
    obj = make_replay_objective_v2_slipobs(space, trajs; observer_kind=kind, seed=1)
    lo, hi = space.flat_lower, space.flat_upper
    theta0 = (lo .+ hi) ./ 2.0
    r = obj(theta0)
    println("  score=$(round(r.score,digits=4)) vel_rmse=$(round(r.vel_rmse,digits=6)) " *
            "rate_rmse=$(round(r.rate_rmse,digits=6)) pos_rmse=$(round(r.pos_rmse,digits=6)) " *
            "heading_rmse=$(round(r.heading_rmse,digits=6)) n_fail=$(r.n_fail)")
    r.n_fail == 0 || error("run_estimator_replay_slipobs.jl --smoke: $(r.n_fail)/$(length(trajs)) trajectories failed")
    println("=== SMOKE OK ===")
end

function main()
    a = parse_args(ARGS)
    kind = Symbol(a["observer"])
    kind in (:smo, :eso) || error("--observer must be smo or eso, got $kind")

    space = ParamSpaceV2SlipObsMod.eskf_slipobs_param_space_v2(kind)
    trajs = replay_trajset()

    if a["smoke"]
        run_smoke(space, trajs, kind)
        return
    end

    obj = make_replay_objective_v2_slipobs(space, trajs; observer_kind=kind, seed=a["seed"])
    outdir = joinpath(a["out"], string(kind), "seed$(a["seed"])")
    mkpath(outdir)

    lo, hi = space.flat_lower, space.flat_upper
    warm = isempty(a["init-from"]) ? nothing : _theta_from_baseline(a["init-from"], space, kind)

    println("\n===== ESKFSlipObsEstimatorV2 replay tuning / $kind / seed $(a["seed"]) — $(length(trajs)) trajectories =====")
    best, best_score, trials, diag = optimize_staged(obj, obj, lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"),
        warm_start=warm)

    kw = kind == :smo ? ParamSpaceV2SlipObsMod.apply_params_v2_smo!(best, space) :
                        ParamSpaceV2SlipObsMod.apply_params_v2_eso!(best, space)
    gains = Dict(string(k) => v for (k, v) in pairs(kw) if v isa Real)

    open(joinpath(outdir, "best_config.json"), "w") do io
        JSON.print(io, Dict(
            "seed" => a["seed"], "observer" => string(kind),
            "best_score" => best_score, "best_gains" => gains,
            "converged" => diag.converged, "stop_reason" => string(diag.stop_reason),
            "phase1_evals" => diag.phase1_evals, "phase2_evals" => diag.phase2_evals,
            "n_trajectories" => length(trajs),
        ), 2)
    end
    open(joinpath(outdir, "trials.json"), "w") do io
        JSON.print(io, [Dict("iter"=>t.iter, "tier"=>string(t.tier), "score"=>t.score,
                             "vel_rmse"=>get(t, :vel_rmse, missing),
                             "pos_rmse"=>get(t, :pos_rmse, missing))
                        for t in trials], 2)
    end
    println("[eskf_slipobs_v2 replay $kind seed$(a["seed"])] best score=$(round(best_score,digits=4)) " *
            "stop_reason=$(diag.stop_reason) evals=$(diag.phase1_evals + diag.phase2_evals) -> $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
