#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5.jl — CLI entry point
# =============================================================================
# ESKFEstimatorV2 replay tuning restricted to the controller-tuning-aligned
# mu=0.5 / chi=0.005 "train12" trajectory set (full trajectories, realistic
# sensor noise). Uses the same 12 trajectories as hybrid_ctrl_v2/controller_tuning/
# trajsets.jl :train12, mapped back to the Arrow filename contract via the
# `profile` field in harness_v2.jl.
#
#   --seed        RNG seed (measurement noise + optimizer start offset)
#   --p1-cap      dxNES eval cap (phase 1)
#   --p2-cap      BOBYQA eval cap (phase 2)
#   --out         output root
#   --smoke       one default-gains eval, no optimization (fast sanity check)
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "trajsets.jl")); using .TrajSetsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "optimizer_stage.jl")); using .StageOptimizerMod
using JSON

function parse_args(argv)
    a = Dict{String,Any}("seed" => 1, "p1-cap" => 250, "p2-cap" => 60,
                         "out" => "hybrid_ctrl_v2/runs_estimator_v2_mu0p5_train12", "smoke" => false,
                         "rel-tol" => 1e-3, "window" => 20, "refiner" => "bobyqa", "refine-halfwidth" => 0.25)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p1-cap"; a["p1-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--p2-cap"; a["p2-cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        elseif arg == "--smoke"; a["smoke"] = true; i += 1
        else; error("run_estimator_replay_mu0p5.jl: unknown arg $arg"); end
    end
    return a
end

function run_smoke(space, trajs)
    println("=== SMOKE: default-gains replay eval on $(length(trajs)) trajectories ===")
    obj = make_replay_objective_v2(space, trajs; seed=1, sensor_kind=:realistic)
    lo, hi = space.flat_lower, space.flat_upper
    theta0 = (lo .+ hi) ./ 2.0
    r = obj(theta0)
    println("  score=$(round(r.score,digits=4)) vel_rmse=$(round(r.vel_rmse,digits=6)) " *
            "rate_rmse=$(round(r.rate_rmse,digits=6)) pos_rmse=$(round(r.pos_rmse,digits=6)) " *
            "heading_rmse=$(round(r.heading_rmse,digits=6)) n_fail=$(r.n_fail)")
    r.n_fail == 0 || error("run_estimator_replay_mu0p5.jl --smoke: $(r.n_fail)/$(length(trajs)) trajectories failed")
    println("=== SMOKE OK ===")
end

function main()
    a = parse_args(ARGS)
    space = eskf_param_space_v2()
    trajs = Main.HarnessV2Mod.mu0p5_train12_replay_trajset()

    if a["smoke"]
        run_smoke(space, trajs)
        return
    end

    lo, hi = space.flat_lower, space.flat_upper
    obj = make_replay_objective_v2(space, trajs; seed=a["seed"], sensor_kind=:realistic)

    outdir = joinpath(a["out"], "seed$(a["seed"])")
    mkpath(outdir)

    println("\n===== ESKFEstimatorV2 mu0.5 train12 replay tuning / seed $(a["seed"]) — $(length(trajs)) trajectories (realistic noise) =====")
    best, best_score, trials, diag = optimize_staged(obj, obj, lo, hi;
        method=:dxnes, refiner=Symbol(a["refiner"]), seed=a["seed"], start_offset=a["seed"],
        p1_cap=a["p1-cap"], p2_cap=a["p2-cap"], rel_tol=a["rel-tol"], window=a["window"],
        refine_halfwidth=a["refine-halfwidth"], trace_path=joinpath(outdir, "trace.csv"))

    kw = apply_params_v2!(best, space)
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
    println("[eskf_v2 mu0p5 train12 replay seed$(a["seed"])] best score=$(round(best_score,digits=4)) " *
            "stop_reason=$(diag.stop_reason) evals=$(diag.phase1_evals + diag.phase2_evals) -> $outdir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
