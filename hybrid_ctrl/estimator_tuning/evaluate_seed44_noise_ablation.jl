#!/usr/bin/env julia
# =============================================================================
# evaluate_seed44_noise_ablation.jl — baseline vs realistic sensor-noise ESKF
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

function evaluate_entry(est_cfg, entry; seed=SEED, sensor_kind=:default, pose_fix_kind=:default)
    log = run_and_log_replay(est_cfg, entry, nominal_controller_cfg(seed);
                             seed=seed, data_dir=DATA_DIR,
                             sensor_kind=sensor_kind, pose_fix_kind=pose_fix_kind)
    obj = estimator_objective_abs([log])
    return (
        name      = String(entry.name),
        profile   = entry.profile_toml,
        combo     = entry.combo_idx,
        vel_rmse  = obj.vel_rmse,
        rate_rmse = obj.rate_rmse,
        pos_rmse  = obj.pos_rmse,
        head_rmse = obj.heading_rmse,
        score     = obj.score,
    )
end

function evaluate_all(est_cfg, entries; sensor_kind=:default, pose_fix_kind=:default)
    results = Vector{NamedTuple}(undef, length(entries))
    Threads.@threads for i in eachindex(entries)
        results[i] = evaluate_entry(est_cfg, entries[i];
                                    sensor_kind=sensor_kind, pose_fix_kind=pose_fix_kind)
    end
    return results
end

function print_table(results)
    println("| profile                 | combo |   vel_rmse (m/s) | rate_rmse (rad/s) |  pos_rmse (m) | head_rmse (rad) | score |")
    println("|-------------------------|------:|-----------------:|------------------:|--------------:|----------------:|------:|")
    for r in results
        @printf("| %-23s | %5d | %16.6f | %17.6f | %13.6f | %15.6f | %5.2f |\n",
                r.name, r.combo, r.vel_rmse, r.rate_rmse, r.pos_rmse, r.head_rmse, r.score)
    end
end

function main()
    cfg_data = JSON.parse(read(BEST_CONFIG_PATH, String))
    theta = Float64.(cfg_data["theta"])
    space = eskf_param_space()
    est_cfg = apply_params!(theta, space)

    entries = load_manifest(MANIFEST_PATH)
    val_entry = (name=:long_circle, profile_toml="long_circle_mu_0p5.toml",
                 ref_type=:velref, mu=0.5, config_dir="trajectory_files_run_0p5_main",
                 run_mode=:velocity, combo_idx=1)
    all_entries = [entries; [val_entry]]

    println("="^80)
    println("BASELINE — default sensor / docking-tier pose fix")
    println("="^80)
    base_res = evaluate_all(est_cfg, all_entries;
                            sensor_kind=:default, pose_fix_kind=:default)
    print_table(base_res)

    println()
    println("="^80)
    println("REALISTIC NOISE — MEMS/encoder/SLAM-grade noise")
    println("  odo: 10 mm/s white + 2 % SF + ~5 mm/s bias")
    println("  gyro: 3 mrad/s white + 0.5 % SF + 3 mrad/s bias")
    println("  pose fix: 2 cm white + 1 cm bias")
    println("  heading fix: 10 mrad white + 5 mrad bias")
    println("="^80)
    noise_res = evaluate_all(est_cfg, all_entries;
                             sensor_kind=:realistic, pose_fix_kind=:realistic)
    print_table(noise_res)

    println()
    println("="^80)
    println("DEGRADATION (noisy - baseline)")
    println("="^80)
    println("| profile                 | combo | Δvel_rmse (m/s) | Δrate_rmse (rad/s) | Δpos_rmse (m) | Δhead_rmse (rad) | Δscore |")
    println("|-------------------------|------:|----------------:|-------------------:|--------------:|-----------------:|-------:|")
    for (b, n) in zip(base_res, noise_res)
        @printf("| %-23s | %5d | %15.6f | %18.6f | %13.6f | %16.6f | %6.2f |\n",
                b.name, b.combo,
                n.vel_rmse - b.vel_rmse,
                n.rate_rmse - b.rate_rmse,
                n.pos_rmse - b.pos_rmse,
                n.head_rmse - b.head_rmse,
                n.score - b.score)
    end

    train_base = base_res[1:end-1]
    val_base   = base_res[end]
    train_noise = noise_res[1:end-1]
    val_noise   = noise_res[end]

    function avg(rs, k)
        return mean(getfield(r, k) for r in rs)
    end

    println()
    println("Training-set averages")
    println("-"^60)
    @printf("%-22s %12s %12s %12s\n", "metric", "baseline", "noisy", "degradation")
    for k in (:vel_rmse, :rate_rmse, :pos_rmse, :head_rmse, :score)
        b = avg(train_base, k)
        n = avg(train_noise, k)
        unit = k == :score ? "" : (k == :vel_rmse ? "m/s" : (k == :rate_rmse ? "rad/s" : (k == :pos_rmse ? "m" : "rad")))
        @printf("%-22s %12.6f %12.6f %12.6f %s\n", string(k), b, n, n - b, unit)
    end

    println()
    println("Validation long_circle combo 1")
    println("-"^60)
    @printf("%-22s %12.6f %12.6f %12.6f\n", "vel_rmse m/s", val_base.vel_rmse, val_noise.vel_rmse, val_noise.vel_rmse - val_base.vel_rmse)
    @printf("%-22s %12.6f %12.6f %12.6f\n", "rate_rmse rad/s", val_base.rate_rmse, val_noise.rate_rmse, val_noise.rate_rmse - val_base.rate_rmse)
    @printf("%-22s %12.6f %12.6f %12.6f\n", "pos_rmse m", val_base.pos_rmse, val_noise.pos_rmse, val_noise.pos_rmse - val_base.pos_rmse)
    @printf("%-22s %12.6f %12.6f %12.6f\n", "head_rmse rad", val_base.head_rmse, val_noise.head_rmse, val_noise.head_rmse - val_base.head_rmse)
    @printf("%-22s %12.4f %12.4f %12.4f\n", "score", val_base.score, val_noise.score, val_noise.score - val_base.score)
end

main()
