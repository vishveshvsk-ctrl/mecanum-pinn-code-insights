#!/usr/bin/env julia
# =============================================================================
# evaluate_seed44_noise_multiseed.jl — locked seed-44 ESKF, realistic noise over
# 5 sensor seeds (42-46).  Reports mean +/- std over bias/noise realizations.
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
using Statistics: mean, std
using Base.Threads

const BEST_CONFIG_PATH = joinpath(ROOT,
    "runs_estimator_posfix_velref_mu05chi005_10traj_replay",
    "seed_44", "eskf_dxnes", "best_config.json")
const MANIFEST_PATH = joinpath(ROOT, "hybrid_ctrl", "estimator_tuning", "manifests", "subset_manifest_mu05_chi005_10traj.json")
const DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
const SEEDS = [42, 43, 44, 45, 46]

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

function evaluate_one(est_cfg, entry, seed::Int; sensor_kind=:default, pose_fix_kind=:default)
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

function evaluate_all(est_cfg, entries, seeds::Vector{Int}; sensor_kind=:default, pose_fix_kind=:default)
    # Flatten (seed, entry) pairs for parallel evaluation.
    pairs = [(s, i) for s in seeds for i in eachindex(entries)]
    results = Vector{NamedTuple}(undef, length(pairs))
    println("  -> evaluating $(length(pairs)) (seed, entry) pairs with $sensor_kind / $pose_fix_kind ...")
    Threads.@threads for k in eachindex(pairs)
        s, i = pairs[k]
        results[k] = evaluate_one(est_cfg, entries[i], s;
                                  sensor_kind=sensor_kind, pose_fix_kind=pose_fix_kind)
    end
    # Reshape to [seed][entry]
    per_seed = [results[(s-1)*length(entries)+1 : s*length(entries)] for s in 1:length(seeds)]
    return per_seed
end

function per_profile_stats(per_seed_results, metric::Symbol)
    n_entries = length(first(per_seed_results))
    n_seeds = length(per_seed_results)
    means = Float64[]
    sigmas = Float64[]
    for i in 1:n_entries
        vals = [per_seed_results[s][i][metric] for s in 1:n_seeds]
        push!(means, mean(vals))
        push!(sigmas, std(vals; corrected=false))
    end
    return means, sigmas
end

function print_table(entries, base_means, base_sigs, noise_means, noise_sigs, delta_means, delta_sigs)
    println("| profile          | c | vel_rmse (m/s)        | rate_rmse (rad/s)     | pos_rmse (m)          | head_rmse (rad)       | score        |")
    println("|------------------|---|-----------------------|-----------------------|-----------------------|-----------------------|--------------|")
    for i in eachindex(entries)
        e = entries[i]
        bm, bs = base_means[i], base_sigs[i]
        nm, ns = noise_means[i], noise_sigs[i]
        dm, ds = delta_means[i], delta_sigs[i]
        @printf("| %-16s | %1d | %7.4f±%-6.4f → %7.4f±%-6.4f | %7.4f±%-6.4f → %7.4f±%-6.4f | %7.4f±%-6.4f → %7.4f±%-6.4f | %7.4f±%-6.4f → %7.4f±%-6.4f | %6.2f±%-5.2f |\n",
                e.name, e.combo,
                bm, bs, nm, ns,
                bm, bs, nm, ns,
                bm, bs, nm, ns,
                bm, bs, nm, ns,
                dm, ds)
    end
end

function print_metric_table(entries, metrics, base_stats, noise_stats, delta_stats)
    println("| profile          | c | metric | baseline           | realistic noise    | degradation        |")
    println("|------------------|---|--------|--------------------|--------------------|--------------------|")
    for i in eachindex(entries)
        e = entries[i]
        for m in metrics
            bm, bs = base_stats[m].means[i], base_stats[m].stds[i]
            nm, ns = noise_stats[m].means[i], noise_stats[m].stds[i]
            dm, ds = delta_stats[m].means[i], delta_stats[m].stds[i]
            unit = m == :score ? "" : (m == :vel_rmse ? "m/s" : (m == :rate_rmse ? "rad/s" : (m == :pos_rmse ? "m" : "rad")))
            @printf("| %-16s | %1d | %-6s | %10.6f ± %-8.6f %s | %10.6f ± %-8.6f %s | %10.6f ± %-8.6f %s |\n",
                    e.name, e.combo_idx, string(m),
                    bm, bs, unit, nm, ns, unit, dm, ds, unit)
        end
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
    println("Multi-seed noise ablation: estimator locked to seed-44 best config")
    println("Sensor seeds: $(SEEDS)")
    println("="^80)

    println()
    println("BASELINE — default sensor / docking-tier pose fix")
    base_per_seed = evaluate_all(est_cfg, all_entries, SEEDS;
                                 sensor_kind=:default, pose_fix_kind=:default)

    println()
    println("REALISTIC NOISE — MEMS/encoder/SLAM-grade noise")
    noise_per_seed = evaluate_all(est_cfg, all_entries, SEEDS;
                                  sensor_kind=:realistic, pose_fix_kind=:realistic)

    metrics = (:vel_rmse, :rate_rmse, :pos_rmse, :head_rmse, :score)
    base_stats = Dict(m => (means=Float64[], stds=Float64[]) for m in metrics)
    noise_stats = Dict(m => (means=Float64[], stds=Float64[]) for m in metrics)
    delta_stats = Dict(m => (means=Float64[], stds=Float64[]) for m in metrics)

    for m in metrics
        bm, bs = per_profile_stats(base_per_seed, m)
        nm, ns = per_profile_stats(noise_per_seed, m)
        dm = nm .- bm
        # Propagate std of difference: sqrt(sb² + sn²)
        ds = sqrt.(bs.^2 .+ ns.^2)
        base_stats[m] = (means=bm, stds=bs)
        noise_stats[m] = (means=nm, stds=ns)
        delta_stats[m] = (means=dm, stds=ds)
    end

    println()
    println("PER-PROFILE MEAN ± STD OVER $(length(SEEDS)) SEEDS")
    println("-"^80)
    print_metric_table(all_entries, metrics, base_stats, noise_stats, delta_stats)

    println()
    println("TRAINING-SET AVERAGES (mean ± std over seeds, 10 trajectories)")
    println("-"^80)
    n_train = length(entries)
    for m in metrics
        # Average over training trajectories per seed, then mean/std over seeds.
        base_seed_avgs = [mean(base_per_seed[s][i][m] for i in 1:n_train) for s in 1:length(SEEDS)]
        noise_seed_avgs = [mean(noise_per_seed[s][i][m] for i in 1:n_train) for s in 1:length(SEEDS)]
        unit = m == :score ? "" : (m == :vel_rmse ? "m/s" : (m == :rate_rmse ? "rad/s" : (m == :pos_rmse ? "m" : "rad")))
        @printf("%-12s baseline: %.6f ± %.6f %s | noisy: %.6f ± %.6f %s | Δ: %.6f ± %.6f %s\n",
                string(m),
                mean(base_seed_avgs), std(base_seed_avgs), unit,
                mean(noise_seed_avgs), std(noise_seed_avgs), unit,
                mean(noise_seed_avgs) - mean(base_seed_avgs),
                sqrt(std(base_seed_avgs)^2 + std(noise_seed_avgs)^2), unit)
    end

    println()
    println("VALIDATION long_circle combo 1 (mean ± std over seeds)")
    println("-"^80)
    val_idx = length(all_entries)
    for m in metrics
        base_vals = [base_per_seed[s][val_idx][m] for s in 1:length(SEEDS)]
        noise_vals = [noise_per_seed[s][val_idx][m] for s in 1:length(SEEDS)]
        unit = m == :score ? "" : (m == :vel_rmse ? "m/s" : (m == :rate_rmse ? "rad/s" : (m == :pos_rmse ? "m" : "rad")))
        @printf("%-12s baseline: %.6f ± %.6f %s | noisy: %.6f ± %.6f %s | Δ: %.6f ± %.6f %s\n",
                string(m),
                mean(base_vals), std(base_vals), unit,
                mean(noise_vals), std(noise_vals), unit,
                mean(noise_vals) - mean(base_vals),
                sqrt(std(base_vals)^2 + std(noise_vals)^2), unit)
    end

    # Save JSON report.
    report = Dict(
        "estimator" => "eskf",
        "estimator_seed" => 44,
        "sensor_seeds" => SEEDS,
        "noise_model" => Dict(
            "odo" => "10 mm/s white + 2 % scale-factor + ~5 mm/s per-axis bias",
            "gyro" => "3 mrad/s white + 0.5 % scale-factor + 3 mrad/s constant bias",
            "pose_fix" => "2 cm white + 1 cm constant bias",
            "heading_fix" => "10 mrad white + 5 mrad constant bias"
        ),
        "per_profile" => [
            Dict("name" => String(all_entries[i].name),
                 "combo" => all_entries[i].combo_idx,
                 "is_validation" => i > n_train,
                 "baseline" => Dict(string(m) => Dict("mean" => base_stats[m].means[i], "std" => base_stats[m].stds[i]) for m in metrics),
                 "realistic_noise" => Dict(string(m) => Dict("mean" => noise_stats[m].means[i], "std" => noise_stats[m].stds[i]) for m in metrics),
                 "degradation" => Dict(string(m) => Dict("mean" => delta_stats[m].means[i], "std" => delta_stats[m].stds[i]) for m in metrics))
            for i in eachindex(all_entries)
        ]
    )
    out_path = joinpath(ROOT,
        "hybrid_ctrl", "estimator_tuning", "reports", "noise_ablation_multiseed_report.json")
    open(out_path, "w") do io
        JSON.print(io, report, 2)
    end
    println()
    println("Saved multi-seed report to:")
    println(out_path)
end

main()
