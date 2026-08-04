#!/usr/bin/env julia
# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/compare_iae_baseline.jl
# (instructions/estimator-v2-iae-adaptive.md §6)
# =============================================================================
# Fixed-default comparison of `ESKFEstimatorV2` (baseline) vs
# `ESKFIAEEstimatorV2` with `iae_kind=:nis_ema`, `tau_iae=0.5`, `kappa_iae=0.5`
# across the frozen 11-trajectory replay manifest.
#
# Decision rule (from brief §6):
#   The hypothesis holds if in-slip velocity RMSE and slip RMSE improve on
#   high-slip entries while grip-phase metrics stay within the ~5% eval-noise
#   floor, AND the γ diagnostic confirms regime-tracking (γ rises in slip,
#   ≈1 in grip). If γ never moves, the innovation carries no usable signal.
#
#   CLI: --seed S (default 1), --out runs_estimator_v2_iae_replay/compare_seedS/
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2_iae.jl")); using .EstimatorModV2IAE
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2_iae.jl")); using .ParamSpaceV2IAEMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2_iae.jl")); using .HarnessV2IAEMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl")); using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
using JSON
using CSV
using DataFrames
using Statistics: mean, median

const DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"

function parse_args(argv)
    a = Dict{String,Any}("seed" => 1,
                         "out" => "hybrid_ctrl_v2/runs_estimator_v2_iae_replay/compare_seed1")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--seed"; a["seed"] = parse(Int, argv[i+1]); i += 2
        elseif arg == "--out"; a["out"] = argv[i+1]; i += 2
        else; error("compare_iae_baseline.jl: unknown arg $arg"); end
    end
    return a
end

"""
    load_baseline_config(seed) -> Dict

Read the baseline-v2 optimized gains from
`runs_estimator_v2_replay/seed<S>/best_config.json` and return a plain Dict of
numeric gains. Falls back to the struct defaults if the file is missing, but
emits a clear warning because the comparison is meant to start from the v2
optimum.
"""
function load_baseline_config(seed::Int)
    path = joinpath(ROOT, "hybrid_ctrl_v2", "runs_estimator_v2_replay",
                    "seed$(seed)", "best_config.json")
    if !isfile(path)
        @warn "compare_iae_baseline.jl: baseline config not found at $path; using defaults"
        return Dict{String,Float64}()
    end
    return JSON.parsefile(path)["best_gains"]
end

"""
    gains_to_est_cfg(gains; use_iae=false, tau_iae=0.5, kappa_iae=0.5) -> NamedTuple

Convert the JSON "best_gains" Dict into a NamedTuple that
`build_estimator_v2` / `build_estimator_v2_iae` accept. All 10 shared v2 dims
are passed through; IAE-specific fields are added when `use_iae=true`.
"""
function gains_to_est_cfg(gains::Dict; use_iae::Bool=false,
                          tau_iae::Float64=0.5, kappa_iae::Float64=0.5)
    base = (
        P0_vel          = Float64(get(gains, "P0_vel",          1e-2)),
        P0_yaw          = Float64(get(gains, "P0_yaw",          1e-2)),
        P0_heading      = Float64(get(gains, "P0_heading",      1e-2)),
        P0_bias_acc     = Float64(get(gains, "P0_bias_acc",     1e-2)),
        P0_bias_gyro    = Float64(get(gains, "P0_bias_gyro",    1e-2)),
        P0_slip         = Float64(get(gains, "P0_slip",         1e-5)),
        P0_pos          = Float64(get(gains, "P0_pos",          1.0)),
        pose_Qn_heading = Float64(get(gains, "pose_Qn_heading", 1e-6)),
        pose_Qn_pos     = Float64(get(gains, "pose_Qn_pos",     1e-6)),
        slip_R_inflate  = Float64(get(gains, "slip_R_inflate",  10.0)),
        use_dhat        = false,
    )
    if use_iae
        return merge(base, (use_iae=true, iae_kind=:nis_ema,
                            tau_iae=tau_iae, kappa_iae=kappa_iae))
    end
    return base
end

"""
    regression_gate!(base_cfg, suite, first_traj)

Run `ESKFEstimatorV2` and `ESKFIAEEstimatorV2(use_iae=false)` on the first
manifest trajectory and assert bit-for-bit agreement (brief §8 gate 2).
"""
function regression_gate(base_cfg, first_traj, seed)
    println("\n=== REGRESSION GATE: ESKFIAEEstimatorV2(use_iae=false) vs ESKFEstimatorV2 ===")

    suite_base = Main.SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)
    suite_iae  = Main.SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)

    log_base = Main.HarnessV2Mod.run_and_log_replay_v2(base_cfg, first_traj, suite_base; seed=seed)

    iae_off_cfg = merge(base_cfg, (use_iae=false, iae_kind=:nis_ema,
                                   tau_iae=0.5, kappa_iae=0.5))
    log_iae_off = Main.HarnessV2IAEMod.run_and_log_replay_v2_iae(iae_off_cfg, first_traj, suite_iae; seed=seed)

    dv = maximum(abs.(log_iae_off.v_hat .- log_base.v_hat))
    dpose = maximum(abs.(log_iae_off.pose_hat .- log_base.pose_hat))
    println("  max |Δv_hat|    = $dv")
    println("  max |Δpose_hat| = $dpose")

    dv < 1e-12 && dpose < 1e-12 ||
        error("compare_iae_baseline.jl: regression gate FAILED " *
              "(max |Δv_hat|=$dv, max |Δpose_hat|=$dpose)")
    println("  REGRESSION GATE PASSED")
end

"""
    slip_rmse_per_axis(log, Hw, data, data_dir)

Compute per-axis signed slip RMSE between the estimated slip `d_hat[1:2]` and
the true body-frame slip `Hw \ ω_true − v_true` interpolated to the estimator
ticks. Returns `(rmse_x, rmse_y)`.
"""
function slip_rmse_per_axis(log, Hw, data, data_dir)
    ts = data.ts
    N = length(log.time)
    ex = Float64[]; ey = Float64[]
    sizehint!(ex, N); sizehint!(ey, N)

    for (i, t) in enumerate(log.time)
        omega = SVector(
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[1], t),
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[2], t),
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[3], t),
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.omega_arrs[4], t))
        v_body = SVector(
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.Vx_arr, t),
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.Vy_arr, t),
            Main.HarnessV2Mod._interp_scalar_v2(ts, data.psidot_arr, t))
        slip_true = Hw \ omega - v_body
        slip_est_x = log.d_hat[1, i]
        slip_est_y = log.d_hat[2, i]
        push!(ex, slip_est_x - slip_true[1])
        push!(ey, slip_est_y - slip_true[2])
    end

    rms(x) = sqrt(sum(xi -> xi^2, x) / max(length(x), 1))
    return rms(ex), rms(ey)
end

function run_one(tr, base_cfg, iae_cfg, seed)
    key = "$(tr.name)_c$(tr.combo_idx)_mu$(tr.mu)"

    # Fresh suites with the same seed for both estimators (the suite object is
    # mutated during replay; reusing it would give baseline and IAE different
    # noise realizations on the same trajectory).
    suite_base = Main.SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)
    suite_iae  = Main.SensorModV2.build_suite(:default; seed=seed, flow=true, fix_tier=:docking)

    log_base = Main.HarnessV2Mod.run_and_log_replay_v2(base_cfg, tr, suite_base; seed=seed)
    gamma_trace = Vector{Float64}(undef, length(log_base.time))
    log_iae  = Main.HarnessV2IAEMod.run_and_log_replay_v2_iae(iae_cfg, tr, suite_iae;
                                                              seed=seed, gamma_trace=gamma_trace)

    m_base = Main.EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log_base])
    m_iae  = Main.EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log_iae])

    # Per-axis slip RMSE requires the true wheel speeds from the cached data.
    base = Main.Profiles.load_base(tr.config_dir)
    params = Main.PlatformParams(base; mu_friction=Float64(tr.mu))
    Hw = Main.EstimatorMod._wheel_jacobian(params)
    meta = (profile=String(tr.name), combo_idx=Int(tr.combo_idx),
            mu=Float64(tr.mu), friction_case=1, friction_model=:lugre_adamov,
            chi=Float64(get(base, "physics", Dict())["chi"]))
    arrow_path = Main.DataStore.expected_output(DATA_DIR, meta)
    data = Main.HarnessV2Mod._load_replay_data_v2(arrow_path, DATA_DIR)

    slip_x_base, slip_y_base = slip_rmse_per_axis(log_base, Hw, data, DATA_DIR)
    slip_x_iae,  slip_y_iae  = slip_rmse_per_axis(log_iae,  Hw, data, DATA_DIR)

    final_gamma = gamma_trace[end]
    frac_gamma_gt_10 = count(g -> g > 10.0, gamma_trace) / length(gamma_trace)

    row_base = (
        traj=key, role=string(tr.role), mu=tr.mu, kind="baseline",
        vel_rmse=m_base.vel_rmse, rate_rmse=m_base.rate_rmse,
        pos_rmse=m_base.pos_rmse, heading_rmse=m_base.heading_rmse,
        inslip_vel_rmse=m_base.inslip_vel_rmse, smoothness=m_base.smoothness,
        score=m_base.score, slip_x_rmse=slip_x_base, slip_y_rmse=slip_y_base,
        final_gamma=NaN, frac_gamma_gt_10=NaN
    )
    row_iae = (
        traj=key, role=string(tr.role), mu=tr.mu, kind="iae",
        vel_rmse=m_iae.vel_rmse, rate_rmse=m_iae.rate_rmse,
        pos_rmse=m_iae.pos_rmse, heading_rmse=m_iae.heading_rmse,
        inslip_vel_rmse=m_iae.inslip_vel_rmse, smoothness=m_iae.smoothness,
        score=m_iae.score, slip_x_rmse=slip_x_iae, slip_y_rmse=slip_y_iae,
        final_gamma=final_gamma, frac_gamma_gt_10=frac_gamma_gt_10
    )
    return row_base, row_iae
end

function main()
    a = parse_args(ARGS)
    seed = a["seed"]
    outdir = a["out"]
    mkpath(outdir)

    trajs = replay_trajset()
    gains = load_baseline_config(seed)
    base_cfg = gains_to_est_cfg(gains; use_iae=false)
    iae_cfg  = gains_to_est_cfg(gains; use_iae=true, tau_iae=0.5, kappa_iae=0.5)

    regression_gate(base_cfg, trajs[1], seed)

    println("\n===== ESKF-v2 baseline vs IAE adaptive-Q / seed $seed — $(length(trajs)) trajectories =====")
    rows = NamedTuple[]
    for tr in trajs
        rbase, riae = run_one(tr, base_cfg, iae_cfg, seed)
        push!(rows, rbase); push!(rows, riae)
        println("  $(rbase.traj) [$(rbase.role)] mu=$(rbase.mu)")
        println("    baseline  score=$(round(rbase.score,digits=4)) vel=$(round(rbase.vel_rmse,digits=6)) " *
                "inslip=$(round(rbase.inslip_vel_rmse,digits=6)) slip_x=$(round(rbase.slip_x_rmse,digits=6)) " *
                "slip_y=$(round(rbase.slip_y_rmse,digits=6))")
        println("    iae       score=$(round(riae.score,digits=4)) vel=$(round(riae.vel_rmse,digits=6)) " *
                "inslip=$(round(riae.inslip_vel_rmse,digits=6)) slip_x=$(round(riae.slip_x_rmse,digits=6)) " *
                "slip_y=$(round(riae.slip_y_rmse,digits=6)) γ_final=$(round(riae.final_gamma,digits=4)) " *
                "γ>10=$(round(riae.frac_gamma_gt_10*100,digits=2))%")
    end

    # Summary rows: mean and median across the manifest.
    numeric_cols = [:vel_rmse, :rate_rmse, :pos_rmse, :heading_rmse,
                    :inslip_vel_rmse, :smoothness, :score,
                    :slip_x_rmse, :slip_y_rmse, :final_gamma, :frac_gamma_gt_10]
    base_rows = filter(r -> r.kind == "baseline", rows)
    iae_rows  = filter(r -> r.kind == "iae", rows)

    _summarize(subset, label, kind) = begin
        fn = label == "mean" ? mean : median
        vals = Dict(c => fn(getproperty.(subset, c)) for c in numeric_cols)
        (
            traj=label, role="summary", mu=NaN, kind=kind,
            vel_rmse=vals[:vel_rmse], rate_rmse=vals[:rate_rmse],
            pos_rmse=vals[:pos_rmse], heading_rmse=vals[:heading_rmse],
            inslip_vel_rmse=vals[:inslip_vel_rmse], smoothness=vals[:smoothness],
            score=vals[:score], slip_x_rmse=vals[:slip_x_rmse],
            slip_y_rmse=vals[:slip_y_rmse],
            final_gamma=kind == "iae" ? vals[:final_gamma] : NaN,
            frac_gamma_gt_10=kind == "iae" ? vals[:frac_gamma_gt_10] : NaN
        )
    end

    for label in ("mean", "median")
        push!(rows, _summarize(base_rows, label, "baseline"))
        push!(rows, _summarize(iae_rows,  label, "iae"))
    end

    df = DataFrame(rows)
    csv_path = joinpath(outdir, "compare_table.csv")
    CSV.write(csv_path, df)
    println("\nWrote comparison table to $csv_path")

    # Stability sanity (brief §8 gate 5): γ should not be clipped >5% of any
    # single trajectory's ticks.
    bad = filter(r -> r.kind == "iae" && r.frac_gamma_gt_10 > 0.05, rows)
    if !isempty(bad)
        @warn "IAE γ clipped above 10 for >5% of ticks on $(length(bad)) trajectories — " *
              "persistent clipping indicates the adaptation is fighting the model"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
