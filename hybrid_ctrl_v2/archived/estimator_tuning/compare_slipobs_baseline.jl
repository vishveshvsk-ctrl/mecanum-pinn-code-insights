#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/compare_slipobs_baseline.jl
# =============================================================================
# First experiment for the slip-observer channel: fixed default observer gains,
# seed-specific baseline optimum for the shared 10 ESKF params.
#
# Decision rule (brief §6): the hypothesis holds if in-slip velocity RMSE and
# slip RMSE improve on the high-slip entries (spin_creep, μ=0.3 cases) while
# grip-phase metrics stay within the eval-noise floor (~5%).  If either variant
# holds → run §7 tuning for that variant.  If neither → the binding constraint
# is proprioceptive information content, not observer design.
#
# CLI: --seed S (default 1)  --out runs_estimator_v2_slipobs_replay/compare_seedS/
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl"));            using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl"));     using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));         using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl"));     using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));       using .EstimatorObjectiveV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2_slipobs.jl"));                  using .EstimatorModV2SlipObs
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2_slipobs.jl")); using .HarnessV2SlipObsMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2_slipobs.jl")); using .ParamSpaceV2SlipObsMod
using JSON
using Printf
using Statistics: mean, median
using StaticArrays
using LinearAlgebra

function parse_args(argv)
    a = Dict{String,Any}("seed" => 1,
                         "out" => "hybrid_ctrl_v2/runs_estimator_v2_slipobs_replay/compare_seed1")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        else; error("compare_slipobs_baseline.jl: unknown arg $arg"); end
    end
    # Re-default the output directory if the user only changed the seed.
    if a["out"] == "hybrid_ctrl_v2/runs_estimator_v2_slipobs_replay/compare_seed1"
        a["out"] = "hybrid_ctrl_v2/runs_estimator_v2_slipobs_replay/compare_seed$(a["seed"])"
    end
    return a
end

function _default_gains()
    return Dict(
        "P0_vel" => 1e-2, "P0_yaw" => 1e-2, "P0_heading" => 1e-2,
        "P0_bias_acc" => 1e-2, "P0_bias_gyro" => 1e-2, "P0_slip" => 1e-5,
        "P0_pos" => 1.0, "pose_Qn_heading" => 1e-6, "pose_Qn_pos" => 1e-6,
        "slip_R_inflate" => 10.0,
    )
end

function load_base_cfg(seed::Int)
    path = joinpath(ROOT, "hybrid_ctrl_v2", "runs_estimator_v2_replay",
                    "seed$(seed)", "best_config.json")
    isfile(path) || error("Baseline best config not found: $path")
    d = JSON.parsefile(path)
    gains = get(d, "best_gains", Dict{String,Any}())
    defaults = _default_gains()
    g(k) = get(gains, k, defaults[k])
    return (
        P0_vel          = g("P0_vel"),
        P0_yaw          = g("P0_yaw"),
        P0_heading      = g("P0_heading"),
        P0_bias_acc     = g("P0_bias_acc"),
        P0_bias_gyro    = g("P0_bias_gyro"),
        P0_slip         = g("P0_slip"),
        P0_pos          = g("P0_pos"),
        pose_Qn_heading = g("pose_Qn_heading"),
        pose_Qn_pos     = g("pose_Qn_pos"),
        slip_R_inflate  = g("slip_R_inflate"),
        use_dhat        = false,
    )
end

function slip_rmse_axes(log, data, Hw)
    ts = data.ts
    N = length(log.time)
    sx_true = Vector{Float64}(undef, N)
    sy_true = Vector{Float64}(undef, N)
    for i in 1:N
        t = log.time[i]
        omega = SVector(
            HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[1], t),
            HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[2], t),
            HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[3], t),
            HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[4], t),
        )
        v_wheel = Hw \ omega
        sx_true[i] = v_wheel[1] - log.v_true[1, i]
        sy_true[i] = v_wheel[2] - log.v_true[2, i]
    end
    ex = log.d_hat[1, :] .- sx_true
    ey = log.d_hat[2, :] .- sy_true
    return sqrt(mean(ex .^ 2)), sqrt(mean(ey .^ 2))
end

function load_traj_data(tr, data_dir)
    base = Main.Profiles.load_base(tr.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    meta = (profile=String(tr.name), combo_idx=Int(tr.combo_idx),
            mu=Float64(tr.mu), friction_case=1, friction_model=:lugre_adamov, chi=chi)
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("Data not found for $(tr.name) c$(tr.combo_idx) mu$(tr.mu): $arrow_path")
    return HarnessV2Mod._load_replay_data_v2(arrow_path, data_dir)
end

function main()
    a = parse_args(ARGS)
    seed = a["seed"]
    outdir = a["out"]
    mkpath(outdir)
    data_dir = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"

    base_cfg = load_base_cfg(seed)
    trajs = replay_trajset()

    # Common observer defaults for the fixed-gain comparison.
    smo_cfg = merge(base_cfg, (
        observer_kind = :smo,
        use_slipobs   = true,
        smo_k1        = 0.1,
        smo_k2        = 5.0,
        smo_delta     = 1e-2,
        eso_omega_o   = 30.0,
        rho_s         = 0.042,
    ))
    eso_cfg = merge(base_cfg, (
        observer_kind = :eso,
        use_slipobs   = true,
        smo_k1        = 0.1,
        smo_k2        = 5.0,
        smo_delta     = 1e-2,
        eso_omega_o   = 30.0,
        rho_s         = 0.042,
    ))

    rows = NamedTuple[]

    for (idx, tr) in enumerate(trajs)
        key = "$(tr.name)_c$(tr.combo_idx)_mu$(tr.mu)"
        # Fresh suite per estimator so each replay starts from the same seeded
        # RNG state (the suite object is mutated during simulation).
        suite_base = SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)

        # --- regression gate on first trajectory --------------------------------
        if idx == 1
            suite_off = SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)
            log_off = HarnessV2SlipObsMod.run_and_log_replay_v2_slipobs(
                merge(base_cfg, (observer_kind=:smo, use_slipobs=false,
                                 smo_k1=0.1, smo_k2=5.0, smo_delta=1e-2,
                                 eso_omega_o=30.0, rho_s=0.042)),
                tr, suite_off; seed=seed, data_dir=data_dir)
            log_base_gate = HarnessV2Mod.run_and_log_replay_v2(base_cfg, tr, suite_base; seed=seed, data_dir=data_dir)
            dv = maximum(abs.(log_off.v_hat .- log_base_gate.v_hat))
            dp = maximum(abs.(log_off.pose_hat .- log_base_gate.pose_hat))
            println("REGRESSION gate on first trajectory ($key): max |Δv|=$dv, max |Δpose|=$dp")
            if dv >= 1e-12 || dp >= 1e-12
                error("REGRESSION FAILED: use_slipobs=false differs from ESKFEstimatorV2 (dv=$dv, dp=$dp)")
            end
            log_base = log_base_gate
        else
            log_base = HarnessV2Mod.run_and_log_replay_v2(base_cfg, tr, suite_base; seed=seed, data_dir=data_dir)
        end

        suite_smo = SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)
        suite_eso = SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)
        log_smo = HarnessV2SlipObsMod.run_and_log_replay_v2_slipobs(smo_cfg, tr, suite_smo; seed=seed, data_dir=data_dir)
        log_eso = HarnessV2SlipObsMod.run_and_log_replay_v2_slipobs(eso_cfg, tr, suite_eso; seed=seed, data_dir=data_dir)

        m_base = EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log_base])
        m_smo  = EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log_smo])
        m_eso  = EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log_eso])

        params = Main.PlatformParams(Main.Profiles.load_base(tr.config_dir); mu_friction=Float64(tr.mu))
        Hw = Main.EstimatorMod._wheel_jacobian(params)
        data = load_traj_data(tr, data_dir)
        slip_base = slip_rmse_axes(log_base, data, Hw)
        slip_smo  = slip_rmse_axes(log_smo,  data, Hw)
        slip_eso  = slip_rmse_axes(log_eso,  data, Hw)

        push!(rows, (
            traj=key, variant="baseline",
            vel_rmse=m_base.vel_rmse, inslip_vel_rmse=m_base.inslip_vel_rmse,
            slip_rmse_x=slip_base[1], slip_rmse_y=slip_base[2],
            pos_rmse=m_base.pos_rmse, heading_rmse=m_base.heading_rmse, score=m_base.score))
        push!(rows, (
            traj=key, variant="smo",
            vel_rmse=m_smo.vel_rmse, inslip_vel_rmse=m_smo.inslip_vel_rmse,
            slip_rmse_x=slip_smo[1], slip_rmse_y=slip_smo[2],
            pos_rmse=m_smo.pos_rmse, heading_rmse=m_smo.heading_rmse, score=m_smo.score))
        push!(rows, (
            traj=key, variant="eso",
            vel_rmse=m_eso.vel_rmse, inslip_vel_rmse=m_eso.inslip_vel_rmse,
            slip_rmse_x=slip_eso[1], slip_rmse_y=slip_eso[2],
            pos_rmse=m_eso.pos_rmse, heading_rmse=m_eso.heading_rmse, score=m_eso.score))
    end

    # --- write CSV ------------------------------------------------------------
    csv_path = joinpath(outdir, "compare_table.csv")
    open(csv_path, "w") do io
        println(io, "traj,variant,vel_rmse,inslip_vel_rmse,slip_rmse_x,slip_rmse_y,pos_rmse,heading_rmse,score")
        for r in rows
            println(io, "$(r.traj),$(r.variant),$(r.vel_rmse),$(r.inslip_vel_rmse)," *
                        "$(r.slip_rmse_x),$(r.slip_rmse_y),$(r.pos_rmse),$(r.heading_rmse),$(r.score)")
        end
        # Summary rows: mean and median per variant across the manifest.
        for variant in ("baseline", "smo", "eso")
            subset = [r for r in rows if r.variant == variant]
            for stat in ("mean", "median")
                agg(v::AbstractVector{<:Real}) = stat == "mean" ? mean(v) : median(v)
                println(io, "$(stat)_$(variant),$(variant)," *
                            "$(agg([r.vel_rmse for r in subset])),$(agg([r.inslip_vel_rmse for r in subset]))," *
                            "$(agg([r.slip_rmse_x for r in subset])),$(agg([r.slip_rmse_y for r in subset]))," *
                            "$(agg([r.pos_rmse for r in subset])),$(agg([r.heading_rmse for r in subset])),$(agg([r.score for r in subset]))")
            end
        end
    end

    # --- stdout table ---------------------------------------------------------
    println("\n===== Slip-observer channel comparison (seed $seed) =====")
    println(rpad("traj", 34), rpad("variant", 10),
            rpad("vel_rmse", 12), rpad("inslip_rmse", 14),
            rpad("slip_x", 12), rpad("slip_y", 12),
            rpad("pos_rmse", 12), rpad("head_rmse", 12), rpad("score", 12))
    println("-" ^ 134)
    for r in rows
        println(rpad(r.traj, 34), rpad(r.variant, 10),
                @sprintf("%.6f  ", r.vel_rmse), @sprintf("%.6f    ", r.inslip_vel_rmse),
                @sprintf("%.6f  ", r.slip_rmse_x), @sprintf("%.6f  ", r.slip_rmse_y),
                @sprintf("%.6f  ", r.pos_rmse), @sprintf("%.6f    ", r.heading_rmse),
                @sprintf("%.4f", r.score))
    end
    println("-" ^ 134)
    for variant in ("baseline", "smo", "eso")
        subset = [r for r in rows if r.variant == variant]
        println("MEAN  ", rpad(variant, 10),
                @sprintf("%.6f  ", mean(r->r.vel_rmse, subset)),
                @sprintf("%.6f    ", mean(r->r.inslip_vel_rmse, subset)),
                @sprintf("%.6f  ", mean(r->r.slip_rmse_x, subset)),
                @sprintf("%.6f  ", mean(r->r.slip_rmse_y, subset)),
                @sprintf("%.6f  ", mean(r->r.pos_rmse, subset)),
                @sprintf("%.6f    ", mean(r->r.heading_rmse, subset)),
                @sprintf("%.4f", mean(r->r.score, subset)))
        println("MEDIAN", rpad(variant, 10),
                @sprintf("%.6f  ", median([r.vel_rmse for r in subset])),
                @sprintf("%.6f    ", median([r.inslip_vel_rmse for r in subset])),
                @sprintf("%.6f  ", median([r.slip_rmse_x for r in subset])),
                @sprintf("%.6f  ", median([r.slip_rmse_y for r in subset])),
                @sprintf("%.6f  ", median([r.pos_rmse for r in subset])),
                @sprintf("%.6f    ", median([r.heading_rmse for r in subset])),
                @sprintf("%.4f", median([r.score for r in subset])))
    end
    println("\nWrote: $csv_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
