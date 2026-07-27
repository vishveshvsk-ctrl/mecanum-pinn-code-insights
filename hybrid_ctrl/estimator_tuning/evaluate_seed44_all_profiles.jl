#!/usr/bin/env julia
# =============================================================================
# evaluate_seed44_all_profiles.jl — per-profile RMSE report for the best ESKF
# from the mu=0.5, chi=0.005 10-traj replay ensemble (seed 44).
# =============================================================================

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

using Pkg
Pkg.activate(ROOT)

include(joinpath(ROOT, "run_one.jl"))
using .Profiles, .DataStore

include(joinpath(ROOT, "hybrid_ctrl/config.jl"));    using .HybridConfigMod
include(joinpath(ROOT, "hybrid_ctrl/bus.jl"));       using .BusMod
include(joinpath(ROOT, "hybrid_ctrl/plant.jl"));     using .PlantMod
include(joinpath(ROOT, "hybrid_ctrl/sensors.jl"));   using .SensorMod
include(joinpath(ROOT, "hybrid_ctrl/estimators.jl")); using .EstimatorMod
include(joinpath(ROOT, "hybrid_ctrl/controllers.jl")); using .ControllerMod
include(joinpath(ROOT, "hybrid_ctrl/fuzzy.jl"));     using .FuzzyMod
include(joinpath(ROOT, "hybrid_ctrl/mixer.jl"));     using .MixerMod
include(joinpath(ROOT, "hybrid_ctrl/scheduler.jl")); using .SchedulerMod

include(joinpath(ROOT, "tuning/subset.jl"));      using .TuningSubsetMod
include(joinpath(ROOT, "tuning/param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "tuning/harness.jl"));     using .TuningHarnessMod
include(joinpath(ROOT, "tuning/objectives.jl"));  using .TuningObjectivesMod

using JSON
using Statistics: mean
using Base.Threads

const BEST_CONFIG_PATH = joinpath(ROOT,
    "runs_estimator_posfix_velref_mu05chi005_10traj_replay",
    "seed_44", "eskf_dxnes", "best_config.json")
const MANIFEST_PATH = joinpath(ROOT, "hybrid_ctrl", "estimator_tuning", "manifests", "subset_manifest_mu05_chi005_10traj.json")
const DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
const SEED = 44

function nominal_controller_cfg(seed::Int)
    return HybridConfig(
        tracking       = :velocity,
        estimator      = :kalman,
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

function load_manifest(path::String)
    data = JSON.parse(read(path, String))
    ens = [
        (name          = Symbol(e["name"]),
         profile_toml  = e["profile_toml"],
         ref_type      = Symbol(e["ref_type"]),
         mu            = Float64(e["mu"]),
         config_dir    = e["config_dir"],
         run_mode      = Symbol(e["run_mode"]),
         combo_idx     = get(e, "combo_idx", nothing),
         pose_fix_tier = get(e, "pose_fix_tier", nothing) === nothing ? nothing : Symbol(e["pose_fix_tier"]))
        for e in data["entries"]
    ]
    return ens
end

function evaluate_entry(est_cfg, entry; seed=SEED)
    log = run_and_log_replay(est_cfg, entry, nominal_controller_cfg(seed);
                             seed=seed, data_dir=DATA_DIR)
    obj = estimator_objective_abs([log])
    return (
        name      = String(entry.name),
        profile   = entry.profile_toml,
        combo     = entry.combo_idx,
        mu        = entry.mu,
        vel_rmse  = obj.vel_rmse,
        rate_rmse = obj.rate_rmse,
        pos_rmse  = obj.pos_rmse,
        head_rmse = obj.heading_rmse,
        score     = obj.score,
    )
end

function main()
    cfg_data = JSON.parse(read(BEST_CONFIG_PATH, String))
    theta = Float64.(cfg_data["theta"])
    space = eskf_param_space()
    est_cfg = apply_params!(theta, space)

    println("Loaded best ESKF config from seed 44 ($(length(theta)) params)")
    println("Data dir: $DATA_DIR")
    println("Evaluation seed: $SEED")
    println()

    entries = load_manifest(MANIFEST_PATH)

    # Held-out validation entry (not in the 10-traj manifest).
    val_entry = (name=:long_circle, profile_toml="long_circle_mu_0p5.toml",
                 ref_type=:velref, mu=0.5, config_dir="trajectory_files_run_0p5_main",
                 run_mode=:velocity, combo_idx=1)

    all_entries = [entries; [val_entry]]

    results = Vector{NamedTuple}(undef, length(all_entries))
    println("Evaluating $(length(all_entries)) trajectories ...")
    Threads.@threads for i in eachindex(all_entries)
        results[i] = evaluate_entry(est_cfg, all_entries[i])
    end

    println()
    println("| profile                 | combo |   vel_rmse (m/s) | rate_rmse (rad/s) |  pos_rmse (m) | head_rmse (rad) | score |")
    println("|-------------------------|------:|-----------------:|------------------:|--------------:|----------------:|------:|")
    for r in results
        @printf("| %-23s | %5d | %16.6f | %17.6f | %13.6f | %15.6f | %5.2f |\n",
                r.name * " (" * splitext(r.profile)[1] * ")", r.combo,
                r.vel_rmse, r.rate_rmse, r.pos_rmse, r.head_rmse, r.score)
    end

    # Aggregate line over training entries only.
    train_res = results[1:end-1]
    println()
    println("Training-set averages ($(length(train_res)) trajectories):")
    @printf("  vel_rmse  = %.6f m/s\n", mean(r.vel_rmse for r in train_res))
    @printf("  rate_rmse = %.6f rad/s\n", mean(r.rate_rmse for r in train_res))
    @printf("  pos_rmse  = %.6f m\n", mean(r.pos_rmse for r in train_res))
    @printf("  head_rmse = %.6f rad\n", mean(r.head_rmse for r in train_res))
    @printf("  score     = %.4f\n", mean(r.score for r in train_res))

    val_res = results[end]
    println()
    println("Validation long_circle combo 1:")
    @printf("  vel_rmse  = %.6f m/s\n", val_res.vel_rmse)
    @printf("  rate_rmse = %.6f rad/s\n", val_res.rate_rmse)
    @printf("  pos_rmse  = %.6f m\n", val_res.pos_rmse)
    @printf("  head_rmse = %.6f rad\n", val_res.head_rmse)
    @printf("  score     = %.4f\n", val_res.score)
end

main()
