#!/usr/bin/env julia
# =============================================================================
# compare_estimators.jl — side-by-side KF vs SMO evaluation on VelRef + PosRef
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
include(joinpath(ROOT, "tuning/metrics.jl"));     using .TuningMetricsMod

using DataFrames
using Arrow
using CSV
using JSON
using TOML
using Statistics

const USAGE = """
Usage: compare_estimators.jl [options]

Options:
  --estimator-dirs DIRS    Comma-separated estimator run dirs
                           (default: runs_estimator/kalman,runs_estimator/smo)
  --run-dir DIR            Trajectory config dir (default: trajectory_files_run_0p5_main)
  --seeds SEEDS            Comma-separated seeds (default: 42,43)
  --out DIR                Output root (default: runs_estimator/compare)
  --help                   Show this message
"""

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "estimator-dirs" => ["runs_estimator/kalman", "runs_estimator/smo"],
        "run-dir"        => "trajectory_files_run_0p5_main",
        "seeds"          => [42, 43],
        "out"            => "runs_estimator/compare",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE); exit(0)
        elseif arg == "--estimator-dirs"
            args["estimator-dirs"] = String.(split(argv[i+1], ',')); i += 2
        elseif arg == "--run-dir"
            args["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--seeds"
            args["seeds"] = [parse(Int, s) for s in split(argv[i+1], ',')]; i += 2
        elseif arg == "--out"
            args["out"] = argv[i+1]; i += 2
        else
            error("Unknown argument: $arg\n$USAGE")
        end
    end
    return args
end

function load_frozen_estimator(est_dir::String)
    path = joinpath(est_dir, "best_config.json")
    isfile(path) || error("Frozen estimator config not found: $path")
    data = JSON.parse(read(path, String))
    cfg = data["config"]
    est_name = cfg["estimator"]
    if est_name == "kalman"
        return (
            estimator       = :kalman,
            Qn              = Diagonal(SVector{3}(Float64.(cfg["Qn"]))),
            Rn_base         = Diagonal(SVector{3}(Float64.(cfg["Rn_base"]))),
            bias_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "bias_Qn", [1e-4, 1e-4])))),
            P0_scale        = Float64(cfg["P0_scale"]),
            slip_R_inflate  = Float64(cfg["slip_R_inflate"]),
            slip_threshold  = Float64(cfg["slip_threshold"]),
            zupt_threshold  = Float64(get(cfg, "zupt_threshold", 0.02)),
            rate_hz         = Float64(cfg["rate_hz"]),
            use_dhat        = false,
        )
    elseif est_name == "kalman_imm"
        return (
            estimator       = :kalman_imm,
            Qn              = Diagonal(SVector{3}(Float64.(cfg["Qn"]))),
            Rn_base         = Diagonal(SVector{3}(Float64.(cfg["Rn_base"]))),
            bias_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "bias_Qn", [1e-4, 1e-4])))),
            slip_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "slip_Qn", [1e-2, 1e-2])))),
            P0_scale        = Float64(cfg["P0_scale"]),
            slip_R_inflate  = Float64(cfg["slip_R_inflate"]),
            slip_threshold  = Float64(cfg["slip_threshold"]),
            zupt_threshold  = Float64(get(cfg, "zupt_threshold", 0.02)),
            alpha_acc       = Float64(get(cfg, "alpha_acc", 1.0)),
            alpha_yaw       = Float64(get(cfg, "alpha_yaw", 0.5)),
            r_boost         = Float64(get(cfg, "r_boost", 10.0)),
            p_stay_grip     = Float64(get(cfg, "p_stay_grip", 0.95)),
            p_stay_slip     = Float64(get(cfg, "p_stay_slip", 0.9)),
            gyro_bias_Qn    = Float64(get(cfg, "gyro_bias_Qn", 1e-6)),
            pose_Qn         = Float64(get(cfg, "pose_Qn", 1e-6)),
            grip_slip_scale = Float64(get(cfg, "grip_slip_scale", 1e-3)),
            nis_thresh      = Float64(get(cfg, "nis_thresh", 9.21)),
            rate_hz         = Float64(cfg["rate_hz"]),
            use_dhat        = false,
        )
    elseif est_name == "eskf"
        return (
            estimator       = :eskf,
            Qn              = Diagonal(SVector{3}(Float64.(cfg["Qn"]))),
            Rn_base         = Diagonal(SVector{3}(Float64.(cfg["Rn_base"]))),
            bias_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "bias_Qn", [1e-4, 1e-4])))),
            slip_Qn         = Diagonal(SVector{2}(Float64.(get(cfg, "slip_Qn", [1e-2, 1e-2])))),
            gyro_bias_Qn    = Float64(get(cfg, "gyro_bias_Qn", 1e-6)),
            pose_Qn         = Float64(get(cfg, "pose_Qn", 1e-6)),
            pose_slip_gain  = Float64(get(cfg, "pose_slip_gain", 10.0)),
            P0_scale        = Float64(cfg["P0_scale"]),
            slip_R_inflate  = Float64(cfg["slip_R_inflate"]),
            slip_threshold  = Float64(cfg["slip_threshold"]),
            zupt_threshold  = Float64(get(cfg, "zupt_threshold", 0.02)),
            alpha_acc       = Float64(get(cfg, "alpha_acc", 1.0)),
            alpha_yaw       = Float64(get(cfg, "alpha_yaw", 0.5)),
            r_boost         = Float64(get(cfg, "r_boost", 10.0)),
            nis_thresh      = Float64(get(cfg, "nis_thresh", 9.21)),
            grip_slip_scale = Float64(get(cfg, "grip_slip_scale", 1e-3)),
            rate_hz         = Float64(cfg["rate_hz"]),
            use_dhat        = false,
        )
    elseif est_name == "smo"
        return (
            estimator       = :smo,
            L               = SVector{3}(Float64.(cfg["L"])),
            K               = SVector{3}(Float64.(cfg["K"])),
            delta           = Float64(cfg["delta"]),
            slip_gate_thresh= Float64(cfg["slip_gate_thresh"]),
            zupt_threshold  = Float64(cfg["zupt_threshold"]),
            bias_gain       = SVector{2}(Float64.(get(cfg, "bias_gain", [0.5, 0.5]))),
            rate_hz         = Float64(cfg["rate_hz"]),
            use_dhat        = true,
        )
    else
        error("Unknown frozen estimator: $est_name")
    end
end

function build_cfg(est_cfg::NamedTuple, entry::NamedTuple, seed::Int)
    tracking = entry.run_mode
    use_pose_fix = tracking == :pose || haskey(entry, :pose_fix_tier)
    pose_fix_tier = get(entry, :pose_fix_tier,
                        tracking == :pose ? :docking : :transit)
    return HybridConfig(
        tracking       = tracking,
        estimator      = est_cfg.estimator,
        use_dhat       = est_cfg.estimator == :smo,
        use_asmc       = true,
        use_mpc        = false,
        use_pid        = false,
        fuzzy          = false,
        fixed_weights  = (1.0, 0.0, 0.0),
        use_pose_fix   = use_pose_fix,
        pose_fix_tier  = pose_fix_tier,
        f_est          = est_cfg.rate_hz,
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

function run_one_eval(est_cfg::NamedTuple, entry::NamedTuple, seed::Int)
    cfg = build_cfg(est_cfg, entry, seed)
    est = TuningHarnessMod._build_estimator(est_cfg)

    base = Profiles.load_base(entry.config_dir)
    chi = get(base, "physics", Dict())["chi"]
    params = PlatformParams(base; mu_friction=Float64(entry.mu))

    prof = TOML.parsefile(joinpath(entry.config_dir, "profiles", entry.profile_toml))["profile"]
    pcfg = Profiles.resolve_profile(prof; combo_idx=get(entry, :combo_idx, nothing))
    ref = Profiles.build(String(prof["builder"]), pcfg)

    sol, df, bus = SchedulerMod.run_hybrid(
        cfg, params, entry.name;
        chi=chi, friction_case=1, ref=ref, est=est,
        config_dir=entry.config_dir, profile_toml=entry.profile_toml,
        return_bus=true)

    slip = [TuningHarnessMod.slip_indicator(sol.u[k], params) for k in eachindex(sol.t)]
    m = compute_metrics(df, ref, cfg.tracking;
                        controller=string(est_cfg.estimator),
                        regime=string(entry.name),
                        seed=seed,
                        slip=slip)

    probe = SchedulerMod.build_estimator_log(bus, params; decimation=1)
    est_log = TuningHarnessMod.EstimatorLog(
        probe.time, probe.v_true, probe.v_hat, probe.pose_true, probe.pose_hat,
        probe.d_hat, probe.slip, string(entry.name), entry.ref_type,
        entry.run_mode, seed)

    return m, est_log
end

function build_subset(run_dir::String)
    core = [
        (name=:octagon, profile_toml="octagon_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity, combo_idx=2),
        (name=:coupled_vomega, profile_toml="coupled_vomega_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name=:spiral_orbit, profile_toml="spiral_orbit_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name=:ellipse, profile_toml="ellipse_mu_0p5.toml",
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose, combo_idx=24),
        (name=:multisine_75percent_cap, profile_toml="multisine_75percent_cap_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name=:spin_creep, profile_toml="spin_creep_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1,
         pose_fix_tier=:docking),
    ]
    return build_tuning_subset(run_dir, core; include_optional=false)
end

_skipnan_mean(x) = mean(filter(!isnan, x))
_skipnan_std(x) = std(filter(!isnan, x))

function main()
    args = parse_args(ARGS)
    run_dir = args["run-dir"]
    out = args["out"]
    mkpath(out)

    subset = build_subset(run_dir)
    est_configs = [load_frozen_estimator(d) for d in args["estimator-dirs"]]

    # Held-out trajectories not in the tuning subset
    held_out = [
        (name=:long_circle, profile_toml="long_circle_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity, combo_idx=1),
        (name=:docking, profile_toml="docking_mu_0p5.toml",
         ref_type=:posref, mu=0.5, config_dir=run_dir, run_mode=:pose, combo_idx=1),
    ]
    all_entries = [entries(subset); held_out]

    rows = NamedTuple[]
    all_logs = TuningHarnessMod.EstimatorLog[]

    for est_cfg in est_configs
        est_name = string(est_cfg.estimator)
        for seed in args["seeds"]
            for entry in all_entries
                println("[eval] $est_name / $(entry.name) / seed=$seed ...")
                m, est_log = run_one_eval(est_cfg, entry, seed)
                push!(rows, (
                    estimator        = est_name,
                    traj_name        = string(entry.name),
                    mode             = string(entry.run_mode),
                    ref_type         = string(entry.ref_type),
                    seed             = seed,
                    pose_rmse        = m.pose_rmse,
                    velocity_nrmse   = m.velocity_nrmse,
                    inslip_nrmse     = m.inslip_nrmse,
                    ce               = m.ce,
                    emax             = m.emax,
                    settling_time    = m.settling_time,
                    overshoot        = m.overshoot,
                    pose_drift_rate  = m.pose_drift_rate,
                    dropout_drift    = m.dropout_drift,
                ))
                push!(all_logs, est_log)
            end
        end
    end

    df = DataFrame(rows)
    Arrow.write(joinpath(out, "estimator_runs.arrow"), df)
    CSV.write(joinpath(out, "estimator_runs.csv"), df)

    # Per-estimator mode-stratified objective on the tuning subset only
    subset_logs_by_est = Dict{String, Vector{TuningHarnessMod.EstimatorLog}}()
    for est_cfg in est_configs
        est_name = string(est_cfg.estimator)
        subset_logs_by_est[est_name] = TuningHarnessMod.EstimatorLog[]
    end
    for (m, est_log) in zip(eachrow(df), all_logs)
        # all_logs order matches rows order
    end
    # Build map by matching index
    for i in eachindex(all_logs)
        log = all_logs[i]
        row = rows[i]
        # Only include logs that belong to the tuning subset
        if any(e -> string(e.name) == log.traj_name, entries(subset))
            push!(subset_logs_by_est[row.estimator], log)
        end
    end

    obj_rows = []
    for est_cfg in est_configs
        est_name = string(est_cfg.estimator)
        logs = subset_logs_by_est[est_name]
        obj = estimator_objective_by_mode(logs; λ_slip=2.0, λ_smooth=0.1, λ_pose=0.5)
        push!(obj_rows, (
            estimator           = est_name,
            score               = obj.score,
            overall_nrmse       = obj.overall_nrmse,
            inslip_nrmse        = obj.inslip_nrmse,
            pose_drift          = obj.pose_drift,
            smoothness          = obj.smoothness,
            velref_overall      = obj.velref_overall,
            velref_inslip       = obj.velref_inslip,
            posref_overall      = obj.posref_overall,
            posref_pose_rmse    = obj.posref_pose_rmse,
            posref_pose_drift   = obj.posref_pose_drift,
        ))
    end
    obj_df = DataFrame(obj_rows)
    Arrow.write(joinpath(out, "objective_by_mode.arrow"), obj_df)
    CSV.write(joinpath(out, "objective_by_mode.csv"), obj_df)

    # Summary mean ± std per estimator / mode / traj_name
    summary = combine(
        groupby(df, [:estimator, :traj_name, :mode]),
        :pose_rmse       => _skipnan_mean => :pose_rmse_mean,
        :pose_rmse       => _skipnan_std  => :pose_rmse_std,
        :velocity_nrmse  => _skipnan_mean => :velocity_nrmse_mean,
        :velocity_nrmse  => _skipnan_std  => :velocity_nrmse_std,
        :inslip_nrmse    => _skipnan_mean => :inslip_nrmse_mean,
        :inslip_nrmse    => _skipnan_std  => :inslip_nrmse_std,
        :ce              => mean => :ce_mean,
        :ce              => std => :ce_std,
        :pose_drift_rate => _skipnan_mean => :pose_drift_rate_mean,
        nrow             => :n,
    )
    Arrow.write(joinpath(out, "summary_table.arrow"), summary)
    CSV.write(joinpath(out, "summary_table.csv"), summary)

    println("Estimator comparison complete. Outputs in $out")
    println("Objective by mode:")
    println(obj_df)
end

main()
