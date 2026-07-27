#!/usr/bin/env julia
# =============================================================================
# tune_estimator.jl — CLI entry point for estimator (KF / SMO) tuning
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg
Pkg.activate(ROOT)

# Load the simulation stack.  run_one.jl defines PlatformParams, Profiles,
# DataStore, and all physics helpers consumed by the hybrid_ctrl modules.
include(joinpath(ROOT, "run_one.jl"))
using .Profiles, .DataStore

# Hybrid-control modules (must be loaded after run_one.jl because they refer to
# Main.Profiles, Main.PlatformParams, etc.).
include(joinpath(ROOT, "hybrid_ctrl/config.jl"));    using .HybridConfigMod
include(joinpath(ROOT, "hybrid_ctrl/bus.jl"));       using .BusMod
include(joinpath(ROOT, "hybrid_ctrl/plant.jl"));     using .PlantMod
include(joinpath(ROOT, "hybrid_ctrl/sensors.jl"));   using .SensorMod
include(joinpath(ROOT, "hybrid_ctrl/estimators.jl")); using .EstimatorMod
include(joinpath(ROOT, "hybrid_ctrl/controllers.jl")); using .ControllerMod
include(joinpath(ROOT, "hybrid_ctrl/fuzzy.jl"));     using .FuzzyMod
include(joinpath(ROOT, "hybrid_ctrl/mixer.jl"));     using .MixerMod
include(joinpath(ROOT, "hybrid_ctrl/scheduler.jl")); using .SchedulerMod

# Reusable tuning spine.
include(joinpath(ROOT, "tuning/subset.jl"));      using .TuningSubsetMod
include(joinpath(ROOT, "tuning/param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "tuning/harness.jl"));     using .TuningHarnessMod
include(joinpath(ROOT, "tuning/objectives.jl"));  using .TuningObjectivesMod
include(joinpath(ROOT, "tuning/optimizer.jl"));   using .TuningOptimizerMod
include(joinpath(ROOT, "tuning/executor.jl"));    using .TuningExecutorMod
include(joinpath(ROOT, "tuning/results.jl"));     using .TuningResultsMod

using JSON
using Statistics: mean
using Base.Threads

const USAGE = """
Usage: tune_estimator.jl [options]

Options:
  --estimator {kalman|smo|both|kalman_imm|eskf}
                                  Estimator to tune (default: smo)
  --optimizer {coarse|dxnes|de|bo}
                                  Optimizer backend (default: coarse)
  --run-dir DIR                   Trajectory config dir (default: trajectory_files_run_0p5_main)
  --seed SEED                     Sensor RNG seed for the search (default: 42)
  --obj-seeds LIST                Comma-separated sensor seeds averaged per objective
                                  evaluation (default: 42; e.g. 42,43 for robustness)
  --exclude-entries LIST          Comma-separated trajectory names to drop from the
                                  curated subset (e.g. ellipse)
  --budget N                      Total optimizer budget (default: 20)
  --max-parallel P                Thread parallelism cap (default: 1)
  --out DIR                       Output root (default: runs_estimator)
  --subset-manifest PATH          Optional manifest to load instead of building
  --warm-start PATH               Optional checkpoint JSON ({"theta": [...]}) to
                                  warm-start the :bo backend (narrows the box
                                  to ±25% around the incumbent)
  --replay                        Replay pre-simulated Arrow data instead of
                                  live closed-loop ODE solves (much faster)
  --replay-data-dir DIR           Directory containing .arrow files and the
                                  accel/ sidecar subdirectory
                                  (default: ../data/Simulation_Data_MecanumSlipSpin_LugreAdamov)
  --help                          Show this message
"""

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "estimator"       => "smo",
        "optimizer"       => "coarse",
        "run-dir"         => "trajectory_files_run_0p5_main",
        "seed"            => 42,
        "obj-seeds"       => "42",
        "exclude-entries" => "",
        "budget"          => 20,
        "max-parallel"    => 1,
        "out"             => "runs_estimator",
        "subset-manifest" => nothing,
        "warm-start"      => nothing,
        "replay"          => false,
        "replay-data-dir" => "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE)
            exit(0)
        elseif arg == "--estimator"
            args["estimator"] = argv[i+1]; i += 2
        elseif arg == "--optimizer"
            args["optimizer"] = argv[i+1]; i += 2
        elseif arg == "--run-dir"
            args["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--seed"
            args["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--obj-seeds"
            args["obj-seeds"] = argv[i+1]; i += 2
        elseif arg == "--exclude-entries"
            args["exclude-entries"] = argv[i+1]; i += 2
        elseif arg == "--budget"
            args["budget"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--max-parallel"
            args["max-parallel"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"
            args["out"] = argv[i+1]; i += 2
        elseif arg == "--subset-manifest"
            args["subset-manifest"] = argv[i+1]; i += 2
        elseif arg == "--warm-start"
            args["warm-start"] = argv[i+1]; i += 2
        elseif arg == "--replay"
            args["replay"] = true; i += 1
        elseif arg == "--replay-data-dir"
            args["replay-data-dir"] = argv[i+1]; i += 2
        else
            error("Unknown argument: $arg\n$USAGE")
        end
    end
    args["optimizer"] in ("coarse", "dxnes", "de", "bo") ||
        error("Unknown optimizer: $(args["optimizer"]) (expected coarse|dxnes|de|bo)")
    return args
end

function build_curated_subset(run_dir::String; exclude::Vector{String}=String[])
    # Curated core list per brief §4.1, combos pinned to whitelist-validated
    # (diagnostics_combined.csv combined_reco="keep") and envelope-feasible
    # picks: ellipse 24 (a=1.5 band, 0.97 m/s, 0.66 m/s²), octagon 2 (0.36 m/s
    # < 0.379 cap at μ=0.3), all others combo 1 (docking has no whitelist row;
    # pinned for determinism).
    core = [
        (name="octagon", profile_toml="octagon_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity, combo_idx=2),
        (name="coupled_vomega", profile_toml="coupled_vomega_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name="spiral_orbit", profile_toml="spiral_orbit_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name="ellipse", profile_toml="ellipse_mu_0p5.toml",
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose, combo_idx=24),
        (name="docking", profile_toml="docking_mu_0p5.toml",
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose, combo_idx=1),
        (name="multisine_75percent_cap", profile_toml="multisine_75percent_cap_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name="spin_creep", profile_toml="spin_creep_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1,
         pose_fix_tier=:docking),
    ]
    if !isempty(exclude)
        core = [e for e in core if !(e.name in exclude)]
    end
    return build_tuning_subset(run_dir, core; include_optional=false)
end

function load_or_build_subset(args::Dict)
    if args["subset-manifest"] !== nothing
        path = args["subset-manifest"]
        data = JSON.parse(read(path, String))
        entries = [
            (name=Symbol(e["name"]),
             profile_toml=e["profile_toml"],
             ref_type=Symbol(e["ref_type"]),
             mu=Float64(e["mu"]),
             config_dir=e["config_dir"],
             run_mode=Symbol(e["run_mode"]),
             combo_idx=get(e, "combo_idx", nothing),
             pose_fix_tier=get(e, "pose_fix_tier", nothing) === nothing ? nothing : Symbol(e["pose_fix_tier"]))
            for e in data["entries"]
        ]
        return TuningSubset(entries, data["hash"])
    else
        excl = isempty(args["exclude-entries"]) ? String[] :
               String.(split(args["exclude-entries"], ","))
        return build_curated_subset(args["run-dir"]; exclude=excl)
    end
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

function _mean_metrics(ms)
    ks = keys(first(ms))
    vals = [mean(Float64(getfield(m, k)) for m in ms) for k in ks]
    return NamedTuple{Tuple(ks)}(Tuple(vals))
end

# Absolute-tolerance objective (physical-unit RMS errors / target tolerances),
# averaged over one or more sensor seeds for robustness.
function make_objective(seeds::Vector{Int}, run_fn)
    return function (est_cfg, subset)
        per_seed = map(seeds) do s
            # Replay each trajectory in the subset in parallel using Julia threads.
            # Each call owns its own estimator/sensor-model state, so this is race-free.
            ens = collect(entries(subset))
            logs = Vector{EstimatorLog}(undef, length(ens))
            Threads.@threads for i in eachindex(ens)
                logs[i] = run_fn(est_cfg, ens[i], s)
            end
            estimator_objective_abs(logs)
        end
        return _mean_metrics(per_seed)
    end
end

function tune_one(est_name::String, subset::TuningSubset, args::Dict)
    est_sym = Symbol(est_name)
    space = est_sym == :kalman     ? kf_param_space() :
            est_sym == :smo        ? smo_param_space() :
            est_sym == :kalman_imm ? imm_kf_param_space() :
            est_sym == :eskf       ? eskf_param_space() :
            error("Unknown estimator: $est_name")

    nominal_ctrl = nominal_controller_cfg(args["seed"])
    obj_seeds = [parse(Int, s) for s in split(args["obj-seeds"], ",")]

    # Choose live simulation or Arrow replay.  Replay skips the plant ODE and
    # regenerates sensor measurements from the saved true state.
    run_fn = if args["replay"]
        rdir = args["replay-data-dir"]
        println("[$est_name] replay mode: data dir = $rdir")
        (est_cfg, entry, seed) -> run_and_log_replay(est_cfg, entry,
                                                     nominal_controller_cfg(seed);
                                                     seed=seed, data_dir=rdir)
    else
        (est_cfg, entry, seed) -> run_and_log(est_cfg, entry,
                                              nominal_controller_cfg(seed);
                                              seed=seed)
    end
    objective = make_objective(obj_seeds, run_fn)

    budget    = args["budget"]
    optimizer = args["optimizer"]

    out_root = args["out"]
    est_out = optimizer == "coarse" ? joinpath(out_root, est_name) :
                                      joinpath(out_root, est_name * "_" * optimizer)
    mkpath(est_out)

    if optimizer == "coarse"
        # Default ask/tell path: no dependencies beyond the tuning spine.
        n_coarse = div(budget, 2)
        n_local  = budget - n_coarse
        opt = CoarseThenLocal(space, n_coarse, n_local)

        trials = NamedTuple[]
        for iter in 1:budget
            theta = ask(opt)
            results = parallel_evaluate([theta], subset, space, objective;
                                        max_parallel=args["max-parallel"])
            res = results[1]
            tell!(opt, theta, res.score)
            push!(trials, merge((iteration=iter,), res))
            println("[$est_name] iter $(iter)/$(budget): score=$(round(res.score,digits=4)) " *
                    "vel_rmse=$(round(get(res, :vel_rmse, NaN),digits=5)) " *
                    "rate_rmse=$(round(get(res, :rate_rmse, NaN),digits=5))")
        end
        best_theta = opt.best_theta
        best_score = opt.best_score
    else
        # Driver-style backends (dxnes / de / bo) own the evaluation loop, and
        # their parallelism lives inside the backend (NThreads), so each
        # theta_objective call evaluates a single theta serially.
        include(joinpath(ROOT, "tuning/optimizer_bbo.jl"))   # defines Main.TuningBBOMod lazily
        theta_objective = function (theta)
            res = parallel_evaluate([theta], subset, space, objective; max_parallel=1)
            return res[1]
        end
        # Optional warm start (incumbent checkpoint JSON from an earlier run).
        x0 = nothing
        if args["warm-start"] !== nothing
            ws = JSON.parse(read(args["warm-start"], String))
            x0 = Float64.(ws["theta"])
            println("[$est_name] warm start from $(args["warm-start"]) " *
                    "(score=$(get(ws, "score", "?")))")
        end
        # invokelatest: the runtime include above defines run_backend in a newer
        # world age than this function's; without it Julia errors "method too new".
        (best_theta, best_score, trials) = Base.invokelatest(
            Main.TuningBBOMod.run_backend,
            Symbol(optimizer), theta_objective, space;
            budget=budget, nthreads=args["max-parallel"], seed=args["seed"],
            trace_path=joinpath(est_out, "fitness_trace.csv"), x0=x0)
    end

    save_trials(trials, est_out)
    save_best_config(best_theta, best_score, space, est_name, est_out)

    # Diagnostics for the best config.
    best_cfg = apply_params!(best_theta, space)
    best_logs = [run_fn(best_cfg, entry, args["seed"])
                 for entry in entries(subset)]
    save_diagnostics(best_logs, joinpath(est_out, "diagnostics"))

    # Held-out validation on a trajectory not in the subset.
    val_entry = (name="long_circle", profile_toml="long_circle_mu_0p5.toml",
                 ref_type=:velref, mu=0.5, config_dir=args["run-dir"],
                 run_mode=:velocity, combo_idx=1)
    val_log = run_fn(best_cfg, val_entry, args["seed"])
    val_obj = estimator_objective_abs([val_log])
    println("[$est_name] validation long_circle: score=$(round(val_obj.score,digits=4)) " *
            "vel_rmse=$(round(val_obj.vel_rmse,digits=5)) " *
            "rate_rmse=$(round(val_obj.rate_rmse,digits=5))")

    val_path = joinpath(est_out, "validation.json")
    open(val_path, "w") do io
        JSON.print(io, Dict(
            "trajectory"      => "long_circle",
            "score"           => val_obj.score,
            "vel_rmse"        => val_obj.vel_rmse,
            "inslip_vel_rmse" => val_obj.inslip_vel_rmse,
            "rate_rmse"       => val_obj.rate_rmse,
            "smoothness"      => val_obj.smoothness,
        ), 2)
    end

    return best_cfg, best_score
end

function main()
    args = parse_args(ARGS)

    subset = load_or_build_subset(args)
    save_subset_manifest(subset, args["out"])
    println("Subset manifest: $(length(entries(subset))) entries, hash=$(subset.hash)")

    est_arg = args["estimator"]
    if est_arg == "both"
        tune_one("kalman", subset, args)
        tune_one("smo",    subset, args)
    else
        tune_one(est_arg, subset, args)
    end

    println("Tuning complete. Outputs in $(args["out"]).")
end

main()
