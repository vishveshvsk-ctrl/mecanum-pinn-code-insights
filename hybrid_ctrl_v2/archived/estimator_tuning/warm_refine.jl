#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/warm_refine.jl — warm-started BOBYQA continuation
# =============================================================================
# NLopt's BOBYQA has no serializable internal state (trust-region/quadratic
# model), so true phase-2-resume-across-processes is impossible (same
# limitation `controller_tuning/run_stage.jl` documents). This instead
# reconstructs the best point each seed's phase-2 already found (every dim in
# eskf_param_space_v2() is :log scale, so theta = log(best_gains[dim.name])
# recovers the exact raw point) and runs a FRESH BOBYQA call seeded there with
# a bigger eval cap -- a practical continuation, not a different search.
# Phase 1 (dxNES) is skipped entirely; we already have its result baked into
# the seed's best point.
#
#   --seeds       comma-separated seed list (default 1,2,3,4,5)
#   --cap         new BOBYQA eval cap (default 150)
#   --in          input root (default hybrid_ctrl_v2/runs_estimator_v2_replay)
#   --out         output root (default same as --in; writes best_config_warm.json)
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl")); using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "controller_tuning", "optimizer_stage.jl")); using .StageOptimizerMod
using JSON

function parse_args(argv)
    a = Dict{String,Any}("seed" => 1, "cap" => 150,
                         "in" => "hybrid_ctrl_v2/runs_estimator_v2_replay",
                         "out" => "hybrid_ctrl_v2/runs_estimator_v2_replay",
                         "refine-halfwidth" => 0.25)
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--cap"; a["cap"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--in"; a["in"] = argv[i+1]; i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        else; error("warm_refine.jl: unknown arg $arg"); end
    end
    return a
end

function main()
    a = parse_args(ARGS)
    space = eskf_param_space_v2()
    trajs = Main.HarnessV2Mod.segmented_replay_trajset(a["seed"])
    lo, hi = space.flat_lower, space.flat_upper

    indir  = joinpath(a["in"],  "seed$(a["seed"])")
    outdir = joinpath(a["out"], "seed$(a["seed"])")
    prev = JSON.parsefile(joinpath(indir, "best_config.json"))
    prev_gains = prev["best_gains"]
    prev_score = prev["best_score"]

    # Recover the exact raw theta point: every dim here is :log scale, so
    # theta = log(decoded_value). Order MUST match space.dims (flat_lower's
    # order), one scalar per dim (all dims in this space have len==1).
    theta0 = Float64[log(Float64(prev_gains[d.name])) for d in space.dims]

    obj = make_replay_objective_v2(space, trajs; seed=a["seed"], sensor_kind=:realistic)

    println("\n===== warm-refine seed $(a["seed"]) — prev best_score=$(round(prev_score,digits=4)), " *
            "cap=$(a["cap"]) =====")

    halfw = a["refine-halfwidth"] .* (hi .- lo)
    lo2 = clamp.(theta0 .- halfw, lo, hi)
    hi2 = clamp.(theta0 .+ halfw, lo, hi)

    mkpath(outdir)
    trace_path = joinpath(outdir, "trace_warm.csv")
    trace_io = open(trace_path, "w")
    println(trace_io, "iter,tier,score,best_so_far"); flush(trace_io)

    p2 = StageOptimizerMod._bobyqa_refine(obj, lo2, hi2, theta0;
        cap=a["cap"], tier_label=:warm, trace_io=trace_io)
    close(trace_io)

    improved = p2.best_score < prev_score
    best_theta = improved ? p2.best : theta0
    best_score = improved ? p2.best_score : prev_score
    gains = Dict(d.name => exp(best_theta[i]) for (i, d) in enumerate(space.dims))
    gains["use_dhat"] = false

    open(joinpath(outdir, "best_config_warm.json"), "w") do io
        JSON.print(io, Dict(
            "seed" => a["seed"], "prev_best_score" => prev_score, "best_score" => best_score,
            "improved" => improved, "best_gains" => gains, "warm_cap" => a["cap"],
            "warm_evals" => p2.evals, "stop_reason" => string(p2.stop_reason),
        ), 2)
    end
    println("[warm seed$(a["seed"])] prev=$(round(prev_score,digits=4)) -> " *
            "new=$(round(best_score,digits=4)) (improved=$improved, evals=$(p2.evals), " *
            "stop_reason=$(p2.stop_reason)) -> $outdir/best_config_warm.json")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
