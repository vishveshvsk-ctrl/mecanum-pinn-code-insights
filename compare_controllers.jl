#!/usr/bin/env julia
# =============================================================================
# compare_controllers.jl — {ASMC,MPC,PID} × {transit,docking} comparison driver
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
include("tuning/optimizer.jl");   using .TuningOptimizerMod
include("tuning/executor.jl");    using .TuningExecutorMod
include("tuning/results.jl");     using .TuningResultsMod
include("tuning/metrics.jl");     using .TuningMetricsMod

using DataFrames
using Arrow
using CSV
using JSON
using Statistics

const USAGE = """
Usage: compare_controllers.jl [options]

Options:
  --run-dir DIR            Trajectory config dir (default: trajectory_files_run_0p5_main)
  --estimator-dir DIR      Directory with frozen best_config.json (default: runs_estimator/kalman)
  --seeds SEEDS            Comma-separated seeds (default: 42,43)
  --out DIR                Output root (default: runs_comparison)
  --help                   Show this message
"""

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "run-dir"       => "trajectory_files_run_0p5_main",
        "estimator-dir" => "runs_estimator/kalman",
        "seeds"         => [42, 43],
        "out"           => "runs_comparison",
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help"
            println(USAGE); exit(0)
        elseif arg == "--run-dir"
            args["run-dir"] = argv[i+1]; i += 2
        elseif arg == "--estimator-dir"
            args["estimator-dir"] = argv[i+1]; i += 2
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
    # Decode small vectors back to the shapes _build_estimator expects
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

function make_base_cfg(est_cfg::NamedTuple, tracking::Symbol, use_pose_fix::Bool, seed::Int; pose_fix_tier::Symbol=:transit)
    return HybridConfig(
        tracking       = tracking,
        estimator      = est_cfg.estimator,
        use_dhat       = est_cfg.estimator == :smo,
        use_asmc       = true,
        use_mpc        = true,
        use_pid        = true,
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

function run_one_controller(est_cfg, cfg, params, ref, traj_name, controller::Symbol, regime::String, seed::Int;
                            fix=nothing, dropout_window=nothing)
    # Build the specific controller-enabled config
    ctrl_cfg = HybridConfig(
        tracking       = cfg.tracking,
        estimator      = cfg.estimator,
        use_dhat       = cfg.use_dhat,
        use_asmc       = controller == :asmc,
        use_mpc        = controller == :mpc,
        use_pid        = controller == :pid,
        fuzzy          = false,
        fixed_weights  = (1.0, 0.0, 0.0),
        use_pose_fix   = cfg.use_pose_fix,
        pose_fix_tier  = cfg.pose_fix_tier,
        f_est          = cfg.f_est,
        f_mpc          = cfg.f_mpc,
        f_pid          = cfg.f_pid,
        f_fuzzy        = cfg.f_fuzzy,
        f_mix          = cfg.f_mix,
        sensor_seed    = cfg.sensor_seed,
        reltol         = cfg.reltol,
        abstol_bristle = cfg.abstol_bristle,
        dtmax          = cfg.dtmax,
        solver_symbol  = cfg.solver_symbol,
        saveat_hz      = cfg.saveat_hz,
    )

    est = TuningHarnessMod._build_estimator(est_cfg)
    base = Profiles.load_base("trajectory_files_run_0p5_main")
    chi = get(base, "physics", Dict())["chi"]

    sol, df = SchedulerMod.run_hybrid(ctrl_cfg, params, Symbol(traj_name);
                                      chi=chi, friction_case=1, ref=ref, est=est,
                                      fix_override=fix)

    # Compute ground-truth slip from saved state for in-slip metric
    slip = [TuningHarnessMod.slip_indicator(sol.u[k], params) for k in eachindex(sol.t)]
    return compute_metrics(df, ref, cfg.tracking;
                           controller=string(controller), regime=regime, seed=seed, slip=slip,
                           dropout_window=dropout_window)
end

function build_transit_entries(run_dir::String)
    return [
        (name="octagon", profile_toml="octagon_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity),
        (name="coupled_vomega", profile_toml="coupled_vomega_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
        (name="spiral_orbit", profile_toml="spiral_orbit_mu_0p5.toml",
         ref_type=:velref, mu=0.5, config_dir=run_dir, run_mode=:velocity),
        (name="multisine_75percent_cap", profile_toml="multisine_75percent_cap_mu_0p3.toml",
         ref_type=:velref, mu=0.3, config_dir=run_dir, run_mode=:velocity),
    ]
end

function build_docking_ref(run_dir::String)
    ref, _cfg, _name = Profiles.pick_and_build(run_dir, ["docking_mu_0p5.toml"])
    return ref
end

function make_pose_fix(tier::Symbol; seed::Int=42, dropout_start::Float64=Inf, dropout_duration::Float64=0.0)
    fix = EstimatorMod.PoseFixModel(tier; seed=seed)
    fix.dropout_start = dropout_start
    fix.dropout_duration = dropout_duration
    return fix
end

function main()
    args = parse_args(ARGS)
    est_cfg = load_frozen_estimator(args["estimator-dir"])

    run_dir = args["run-dir"]
    base = Profiles.load_base(run_dir)
    chi = get(base, "physics", Dict())["chi"]

    out = args["out"]
    mkpath(out)

    all_metrics = MetricSet[]
    transit_entries = build_transit_entries(run_dir)
    docking_ref = build_docking_ref(run_dir)
    T_dock = docking_ref.T_total
    # Controlled fix dropout in the middle of the approach/hold phase
    drop_start = max(1.0, T_dock * 0.3)
    drop_duration = min(2.0, T_dock * 0.25)

    for seed in args["seeds"]
        params_base = PlatformParams(base; mu_friction=0.5)

        # ---- Transit (velocity tracking, intermittent pose fix) -------------
        cfg_transit = make_base_cfg(est_cfg, :velocity, true, seed; pose_fix_tier=:transit)
        # Construct a shared transit-tier fix so all transit controllers see
        # the same dropout/outlier stream for this seed.
        fix_transit = make_pose_fix(:transit; seed=seed)
        for entry in transit_entries
            params_e = PlatformParams(base; mu_friction=Float64(entry.mu))
            ref = Profiles.pick_and_build(entry.config_dir, [entry.profile_toml])[1]
            for ctrl in (:asmc, :mpc, :pid)
                m = run_one_controller(est_cfg, cfg_transit, params_e, ref, entry.name, ctrl, "transit", seed; fix=fix_transit)
                push!(all_metrics, m)
                println("[transit] $(entry.name) $(ctrl) seed=$(seed): v_nrmse=$(round(m.velocity_nrmse,digits=4)) CE=$(round(m.ce,digits=2))")
            end
        end

        # ---- Docking (pose tracking, continuous precise fix) ----------------
        cfg_dock = make_base_cfg(est_cfg, :pose, true, seed; pose_fix_tier=:docking)
        fix_dock = make_pose_fix(:docking; seed=seed)
        for ctrl in (:asmc, :mpc, :pid)
            m = run_one_controller(est_cfg, cfg_dock, params_base, docking_ref, "docking", ctrl, "docking", seed; fix=fix_dock)
            push!(all_metrics, m)
            println("[docking] $(ctrl) seed=$(seed): pose_rmse=$(round(m.pose_rmse,digits=4)) CE=$(round(m.ce,digits=2))")
        end

        # ---- Dropout (pose tracking, fix disabled for $(drop_duration) s) ---
        fix_dropout = make_pose_fix(:docking; seed=seed,
                                    dropout_start=drop_start,
                                    dropout_duration=drop_duration)
        for ctrl in (:asmc, :mpc, :pid)
            m = run_one_controller(est_cfg, cfg_dock, params_base, docking_ref, "docking", ctrl, "dropout", seed;
                                   fix=fix_dropout, dropout_window=(drop_start, drop_start + drop_duration))
            push!(all_metrics, m)
            println("[dropout] $(ctrl) seed=$(seed): pose_rmse=$(round(m.pose_rmse,digits=4)) dropout_drift=$(round(m.dropout_drift,digits=4)) CE=$(round(m.ce,digits=2))")
        end
    end

    # ---- Aggregate and save -------------------------------------------------
    df = DataFrame([
        (controller = m.controller,
         regime = m.regime,
         seed = m.seed,
         pose_rmse = m.pose_rmse,
         velocity_nrmse = m.velocity_nrmse,
         inslip_nrmse = m.inslip_nrmse,
         ce = m.ce,
         emax = m.emax,
         settling_time = m.settling_time,
         overshoot = m.overshoot,
         pose_drift_rate = m.pose_drift_rate,
         dropout_drift = m.dropout_drift)
        for m in all_metrics
    ])
    Arrow.write(joinpath(out, "comparison_runs.arrow"), df)

    # Mean ± std tables per controller/regime
    groups = Dict{Tuple{String,String}, Vector{MetricSet}}()
    for m in all_metrics
        k = (m.controller, m.regime)
        push!(get!(groups, k, MetricSet[]), m)
    end
    rows = []
    for ((ctrl, regime), ms) in groups
        pose_rmse = [m.pose_rmse for m in ms if !isnan(m.pose_rmse)]
        vel_nrmse = [m.velocity_nrmse for m in ms if !isnan(m.velocity_nrmse)]
        inslip = [m.inslip_nrmse for m in ms if !isnan(m.inslip_nrmse)]
        dropout_drift = [m.dropout_drift for m in ms if !isnan(m.dropout_drift)]
        ce = [m.ce for m in ms]
        push!(rows, (
            controller = ctrl,
            regime = regime,
            pose_rmse_mean = isempty(pose_rmse) ? NaN : mean(pose_rmse),
            pose_rmse_std = isempty(pose_rmse) ? NaN : std(pose_rmse),
            velocity_nrmse_mean = isempty(vel_nrmse) ? NaN : mean(vel_nrmse),
            velocity_nrmse_std = isempty(vel_nrmse) ? NaN : std(vel_nrmse),
            inslip_nrmse_mean = isempty(inslip) ? NaN : mean(inslip),
            inslip_nrmse_std = isempty(inslip) ? NaN : std(inslip),
            dropout_drift_mean = isempty(dropout_drift) ? NaN : mean(dropout_drift),
            dropout_drift_std = isempty(dropout_drift) ? NaN : std(dropout_drift),
            ce_mean = mean(ce),
            ce_std = std(ce),
            n = length(ms),
        ))
    end
    summary = DataFrame(rows)
    Arrow.write(joinpath(out, "summary_table.arrow"), summary)
    CSV.write(joinpath(out, "summary_table.csv"), summary)
    println("Comparison complete. Outputs in $out")
end

main()
