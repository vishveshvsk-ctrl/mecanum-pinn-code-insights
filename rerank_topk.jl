#!/usr/bin/env julia
# =============================================================================
# rerank_topk.jl — robustness re-rank of top-K tuning trials on held-out seeds
#
# The tuning search runs on a single sensor seed (deterministic objective).
# This script takes the top-K distinct trials from a finished run, re-evaluates
# each on held-out sensor seeds, and re-ranks by mean(score)+std(score) across
# all seeds — guarding against overfitting a lucky minimum to the search seed.
# =============================================================================

using Pkg
Pkg.activate(".")

include("run_one.jl")
using .Profiles, .DataStore

include("hybrid_ctrl/config.jl");    using .HybridConfigMod
include("hybrid_ctrl/bus.jl");       using .BusMod
include("hybrid_ctrl/plant.jl");     using .PlantMod
include("hybrid_ctrl/sensors.jl");   using .SensorMod
include("hybrid_ctrl/estimators.jl"); using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");     using .FuzzyMod
include("hybrid_ctrl/mixer.jl");     using .MixerMod
include("hybrid_ctrl/scheduler.jl"); using .SchedulerMod

include("tuning/subset.jl");      using .TuningSubsetMod
include("tuning/param_space.jl"); using .TuningParamSpaceMod
include("tuning/harness.jl");     using .TuningHarnessMod
include("tuning/objectives.jl");  using .TuningObjectivesMod
include("tuning/executor.jl");    using .TuningExecutorMod
include("tuning/results.jl");     using .TuningResultsMod

using JSON, Arrow, DataFrames, Statistics

const USAGE = """
Usage: rerank_topk.jl [options]

Options:
  --trials PATH          trials.arrow from the finished run
                         (default: runs_estimator_imm/kalman_imm_dxnes/trials.arrow)
  --manifest PATH        subset manifest JSON (default: runs_estimator_imm/subset_manifest.json)
  --estimator NAME       estimator the trials belong to (default: kalman_imm)
  --run-dir DIR          trajectory config dir (default: trajectory_files_run_0p5_main)
  --search-seed SEED     seed the search used; its score is taken from trials (default: 42)
  --seeds LIST           comma-separated held-out seeds (default: 43,44)
  --topk K               number of top trials to re-rank (default: 10)
  --max-parallel P       thread parallelism cap (default: 1)
  --out PATH             results JSON (default: <trials dir>/rerank_results.json)
  --freeze-dir DIR       if given, write the winner's best_config.json here
  --help                 Show this message
"""

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "trials"       => joinpath("runs_estimator_imm", "kalman_imm_dxnes", "trials.arrow"),
        "manifest"     => joinpath("runs_estimator_imm", "subset_manifest.json"),
        "estimator"    => "kalman_imm",
        "run-dir"      => "trajectory_files_run_0p5_main",
        "search-seed"  => 42,
        "seeds"        => "43,44",
        "topk"         => 10,
        "max-parallel" => 1,
        "out"          => nothing,
        "freeze-dir"   => nothing,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE); exit(0)
        elseif arg == "--trials"
            args["trials"] = argv[i+1]; i += 2
        elseif arg == "--manifest"
            args["manifest"] = argv[i+1]; i += 2
        elseif arg == "--estimator"
            args["estimator"] = argv[i+1]; i += 2
        elseif arg == "--run-dir"
            args["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--search-seed"
            args["search-seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--seeds"
            args["seeds"] = argv[i+1]; i += 2
        elseif arg == "--topk"
            args["topk"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--max-parallel"
            args["max-parallel"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"
            args["out"] = argv[i+1]; i += 2
        elseif arg == "--freeze-dir"
            args["freeze-dir"] = argv[i+1]; i += 2
        else
            error("Unknown argument: $arg\n$USAGE")
        end
    end
    if args["out"] === nothing
        args["out"] = joinpath(dirname(args["trials"]), "rerank_results.json")
    end
    return args
end

function load_subset(manifest_path::String)
    data = JSON.parse(read(manifest_path, String))
    entries = [
        (name=Symbol(e["name"]),
         profile_toml=e["profile_toml"],
         ref_type=Symbol(e["ref_type"]),
         mu=Float64(e["mu"]),
         config_dir=e["config_dir"],
         run_mode=Symbol(e["run_mode"]))
        for e in data["entries"]
    ]
    return TuningSubset(entries, data["hash"])
end

function nominal_controller_cfg(seed::Int)
    return HybridConfig(
        tracking       = :velocity,
        estimator      = :kalman,       # overridden per run by run_and_log
        use_dhat       = false,
        use_asmc       = true,
        use_mpc        = false,
        use_pid        = false,
        fuzzy          = false,
        fixed_weights  = (1.0, 0.0, 0.0),
        f_est          = 1000.0,
        f_mpc          = 100.0,
        f_pid          = 100.0,
        f_fuzzy        = 50.0,
        f_mix          = 1000.0,
        sensor_seed    = seed,
        reltol         = 1e-8,
        abstol_bristle = 1e-10,
        dtmax          = 1e-3,
        solver_symbol  = :TRBDF2,
        saveat_hz      = 500.0,
    )
end

function make_objective(nominal_ctrl_cfg, seed::Int)
    return function (est_cfg, subset)
        logs = [run_and_log(est_cfg, entry, nominal_ctrl_cfg; seed=seed)
                for entry in entries(subset)]
        return estimator_objective(logs; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5)
    end
end

function main()
    args = parse_args(ARGS)

    est_sym = Symbol(args["estimator"])
    space = est_sym == :kalman     ? kf_param_space() :
            est_sym == :smo        ? smo_param_space() :
            est_sym == :kalman_imm ? imm_kf_param_space() :
            error("Unknown estimator: $(args["estimator"])")

    subset = load_subset(args["manifest"])
    println("Subset: $(length(entries(subset))) entries, hash=$(subset.hash)")

    # Top-K distinct trials by search-seed score (finite scores only).
    df = DataFrame(Arrow.Table(args["trials"]))
    df = filter(row -> isfinite(row.score), df)
    sort!(df, :score)
    K = min(args["topk"], nrow(df))
    K >= 1 || error("No finite-score trials in $(args["trials"])")
    top = df[1:K, :]
    thetas = [Vector{Float64}(collect(row.theta)) for row in eachrow(top)]
    search_scores = [Float64(row.score) for row in eachrow(top)]
    println("Re-ranking top $K trials on seeds [$(args["seeds"])] " *
            "(search seed $(args["search-seed"]) scores taken from trials)")

    # Re-evaluate every candidate on each held-out seed.
    heldout_seeds = [parse(Int, s) for s in split(args["seeds"], ",")]
    per_seed_scores = Vector{Vector{Float64}}()
    for s in heldout_seeds
        objective = make_objective(nominal_controller_cfg(s), s)
        results = parallel_evaluate(thetas, subset, space, objective;
                                    max_parallel=args["max-parallel"])
        push!(per_seed_scores, [Float64(r.score) for r in results])
        println("  seed $s done: best=$(round(minimum(per_seed_scores[end]), digits=4))")
    end

    # Robust criterion: mean + std across ALL seeds (search + held-out).
    records = []
    for k in 1:K
        all_scores = vcat(search_scores[k], [per_seed_scores[j][k] for j in eachindex(heldout_seeds)])
        robust = mean(all_scores) + std(all_scores)
        push!(records, Dict(
            "rank_input"    => k,
            "search_score"  => search_scores[k],
            "seed_scores"   => Dict(string(args["search-seed"]) => search_scores[k],
                                    [string(heldout_seeds[j]) => per_seed_scores[j][k]
                                     for j in eachindex(heldout_seeds)]...),
            "mean_score"    => mean(all_scores),
            "std_score"     => std(all_scores),
            "robust_score"  => robust,
            "theta"         => thetas[k],
        ))
        println("  cand $k: search=$(round(search_scores[k],digits=4)) " *
                "mean=$(round(mean(all_scores),digits=4)) " *
                "std=$(round(std(all_scores),digits=4)) " *
                "robust=$(round(robust,digits=4))")
    end
    sort!(records, by = r -> r["robust_score"])
    winner = records[1]
    println("Winner: input-rank $(winner["rank_input"]), " *
            "robust=$(round(winner["robust_score"],digits=4))")

    open(args["out"], "w") do io
        JSON.print(io, Dict(
            "trials"        => args["trials"],
            "search_seed"   => args["search-seed"],
            "heldout_seeds" => heldout_seeds,
            "criterion"     => "mean(score)+std(score) over all seeds",
            "ranking"       => records,
        ), 2)
    end
    println("Wrote $(args["out"])")

    if args["freeze-dir"] !== nothing
        path = save_best_config(winner["theta"], winner["robust_score"], space,
                                args["estimator"], args["freeze-dir"])
        println("Froze winner config to $path")
    end
end

main()
