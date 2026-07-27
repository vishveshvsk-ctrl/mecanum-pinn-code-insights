#!/usr/bin/env julia
# =============================================================================
# tune_estimator.jl — CLI entry point for estimator (KF / SMO) tuning
# =============================================================================

using Pkg
Pkg.activate(".")

# Load the simulation stack.  run_one.jl defines PlatformParams, Profiles,
# DataStore, and all physics helpers consumed by the hybrid_ctrl modules.
include("run_one.jl")
using .Profiles, .DataStore

# Hybrid-control modules (must be loaded after run_one.jl because they refer to
# Main.Profiles, Main.PlatformParams, etc.).
include("hybrid_ctrl/config.jl");    using .HybridConfigMod
include("hybrid_ctrl/bus.jl");       using .BusMod
include("hybrid_ctrl/plant.jl");     using .PlantMod
include("hybrid_ctrl/sensors.jl");   using .SensorMod
include("hybrid_ctrl/estimators.jl"); using .EstimatorMod
include("hybrid_ctrl/controllers.jl"); using .ControllerMod
include("hybrid_ctrl/fuzzy.jl");     using .FuzzyMod
include("hybrid_ctrl/mixer.jl");     using .MixerMod
include("hybrid_ctrl/scheduler.jl"); using .SchedulerMod

# Reusable tuning spine.
include("tuning/subset.jl");      using .TuningSubsetMod
include("tuning/param_space.jl"); using .TuningParamSpaceMod
include("tuning/harness.jl");     using .TuningHarnessMod
include("tuning/objectives.jl");  using .TuningObjectivesMod
include("tuning/optimizer.jl");   using .TuningOptimizerMod
include("tuning/executor.jl");    using .TuningExecutorMod
include("tuning/results.jl");     using .TuningResultsMod

using JSON

const USAGE = """
Usage: tune_estimator.jl [options]

Options:
  --estimator {kalman|smo|both|kalman_imm}
                                  Estimator to tune (default: smo)
  --optimizer {coarse|dxnes|de|bo}
                                  Optimizer backend (default: coarse)
  --run-dir DIR                   Trajectory config dir (default: trajectory_files_run_0p5_main)
  --seed SEED                     Sensor RNG seed (default: 42)
  --budget N                      Total optimizer budget (default: 20)
  --max-parallel P                Thread parallelism cap (default: 1)
  --out DIR                       Output root (default: runs_estimator)
  --subset-manifest PATH          Optional manifest to load instead of building
  --help                          Show this message
"""

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "estimator"       => "smo",
        "optimizer"       => "coarse",
        "run-dir"         => "trajectory_files_run_0p5_main",
        "seed"            => 42,
        "budget"          => 20,
        "max-parallel"    => 1,
        "out"             => "runs_estimator",
        "subset-manifest" => nothing,
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
        elseif arg == "--budget"
            args["budget"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--max-parallel"
            args["max-parallel"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"
            args["out"] = argv[i+1]; i += 2
        elseif arg == "--subset-manifest"
            args["subset-manifest"] = argv[i+1]; i += 2
        else
            error("Unknown argument: $arg\n$USAGE")
        end
    end
    args["optimizer"] in ("coarse", "dxnes", "de", "bo") ||
        error("Unknown optimizer: $(args["optimizer"]) (expected coarse|dxnes|de|bo)")
    return args
end

function build_curated_subset(run_dir::String)
    # Curated core list per brief §4.1.  ellipse runs in :pose mode now that
    # asmc_wrench! :pose and PoseFixModel are wired.
    core = [
        (name="octagon", profile_toml="octagon_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity),
        (name="coupled_vomega", profile_toml="coupled_vomega_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
        (name="spiral_orbit", profile_toml="spiral_orbit_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
        (name="ellipse", profile_toml="ellipse_mu_0p5.toml",
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose),
        (name="multisine_75percent_cap", profile_toml="multisine_75percent_cap_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity),
    ]
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
             run_mode=Symbol(e["run_mode"]))
            for e in data["entries"]
        ]
        return TuningSubset(entries, data["hash"])
    else
        return build_curated_subset(args["run-dir"])
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

function make_objective(nominal_ctrl_cfg, seed::Int)
    return function (est_cfg, subset)
        logs = [run_and_log(est_cfg, entry, nominal_ctrl_cfg; seed=seed)
                for entry in entries(subset)]
        return estimator_objective(logs; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5)
    end
end

function tune_one(est_name::String, subset::TuningSubset, args::Dict)
    est_sym = Symbol(est_name)
    space = est_sym == :kalman     ? kf_param_space() :
            est_sym == :smo        ? smo_param_space() :
            est_sym == :kalman_imm ? imm_kf_param_space() :
            error("Unknown estimator: $est_name")

    nominal_ctrl = nominal_controller_cfg(args["seed"])
    objective = make_objective(nominal_ctrl, args["seed"])

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
                    "overall=$(round(res.overall_nrmse,digits=4)) " *
                    "inslip=$(round(res.inslip_nrmse,digits=4))")
        end
        best_theta = opt.best_theta
        best_score = opt.best_score
    else
        # Driver-style backends (dxnes / de / bo) own the evaluation loop, and
        # their parallelism lives inside the backend (NThreads), so each
        # theta_objective call evaluates a single theta serially.
        include("tuning/optimizer_bbo.jl")   # defines Main.TuningBBOMod lazily
        theta_objective = function (theta)
            res = parallel_evaluate([theta], subset, space, objective; max_parallel=1)
            return res[1]
        end
        # invokelatest: the runtime include above defines run_backend in a newer
        # world age than this function's; without it Julia errors "method too new".
        (best_theta, best_score, trials) = Base.invokelatest(
            Main.TuningBBOMod.run_backend,
            Symbol(optimizer), theta_objective, space;
            budget=budget, nthreads=args["max-parallel"], seed=args["seed"])
    end

    save_trials(trials, est_out)
    save_best_config(best_theta, best_score, space, est_name, est_out)

    # Diagnostics for the best config.
    best_cfg = apply_params!(best_theta, space)
    best_logs = [run_and_log(best_cfg, entry, nominal_ctrl; seed=args["seed"])
                 for entry in entries(subset)]
    save_diagnostics(best_logs, joinpath(est_out, "diagnostics"))

    # Held-out validation on a trajectory not in the subset.
    val_entry = (name="long_circle", profile_toml="long_circle_mu_0p5.toml",
                 ref_type=:velref, mu=0.5, config_dir=args["run-dir"],
                 run_mode=:velocity)
    val_log = run_and_log(best_cfg, val_entry, nominal_ctrl; seed=args["seed"])
    val_obj = estimator_objective([val_log]; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5)
    println("[$est_name] validation long_circle: score=$(round(val_obj.score,digits=4)) " *
            "overall=$(round(val_obj.overall_nrmse,digits=4))")

    val_path = joinpath(est_out, "validation.json")
    open(val_path, "w") do io
        JSON.print(io, Dict(
            "trajectory"    => "long_circle",
            "score"         => val_obj.score,
            "overall_nrmse" => val_obj.overall_nrmse,
            "inslip_nrmse"  => val_obj.inslip_nrmse,
            "pose_drift"    => val_obj.pose_drift,
            "smoothness"    => val_obj.smoothness,
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
