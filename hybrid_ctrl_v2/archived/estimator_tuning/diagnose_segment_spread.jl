#!/usr/bin/env julia
# =============================================================================
# diagnose_segment_spread.jl
# =============================================================================
# Per-segment diagnostics for the realistic segmented ESKF v2 re-tune.
# For each seed, loads the best gains from
# runs_estimator_v2_replay_realistic_seg/seed<N>/best_config.json, then:
#   1. Replays each of the 22 segments individually and computes metrics.
#   2. Replays each of the 11 full trajectories and computes metrics.
# Writes a CSV for spreadsheet/ plotting inspection.
# =============================================================================
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
cd(ROOT)

include(joinpath(ROOT, "hybrid_ctrl_v2", "tune_controller_v2.jl"))
include(joinpath(ROOT, "hybrid_ctrl_v2", "sensors_v2.jl"));    using .SensorModV2
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimators_v2.jl")); using .EstimatorModV2
include(joinpath(ROOT, "tuning", "param_space.jl")); using .TuningParamSpaceMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "param_space_v2.jl")); using .ParamSpaceV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "harness_v2.jl"));     using .HarnessV2Mod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "replay_trajset.jl")); using .ReplayTrajSetMod
include(joinpath(ROOT, "hybrid_ctrl_v2", "estimator_tuning", "objective_v2.jl"));   using .EstimatorObjectiveV2Mod
using JSON
using CSV
using DataFrames

const DATA_DIR = "../data/Simulation_Data_MecanumSlipSpin_LugreAdamov"
const IN_ROOT  = "hybrid_ctrl_v2/runs_estimator_v2_replay_realistic_seg"
const OUT_CSV  = "hybrid_ctrl_v2/runs_estimator_v2_replay_realistic_seg/segment_diagnostics.csv"

function gains_to_cfg(gains::Dict)
    return (
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
end

function eval_segment(cfg, seg; seed::Int)
    suite = SensorModV2.build_suite(:realistic; seed=seed, flow=true, fix_tier=:docking)
    log = HarnessV2Mod.run_and_log_replay_v2(cfg, seg, suite; seed=seed, t_window=seg.t_window, data_dir=DATA_DIR)
    m = EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log])
    return m
end

function eval_full(cfg, tr; seed::Int)
    suite = SensorModV2.build_suite(:realistic; seed=seed, flow=true, fix_tier=:docking)
    log = HarnessV2Mod.run_and_log_replay_v2(cfg, tr, suite; seed=seed, data_dir=DATA_DIR)
    m = EstimatorObjectiveV2Mod.estimator_objective_abs_v2([log])
    return m
end

function main()
    rows = NamedTuple[]
    full_trajs = replay_trajset()

    for seed in (1, 2, 3)
        path = joinpath(ROOT, IN_ROOT, "seed$(seed)", "best_config.json")
        gains = JSON.parsefile(path)["best_gains"]
        cfg = gains_to_cfg(gains)

        # --- segmented replay ---
        segs = HarnessV2Mod.segmented_replay_trajset(seed)
        for seg in segs
            m = eval_segment(cfg, seg; seed=seed)
            push!(rows, (
                seed=seed, mode="segment", name=seg.name, combo_idx=seg.combo_idx,
                mu=seg.mu, role=string(seg.role), t_start=seg.t_window[1], t_end=seg.t_window[2],
                score=m.score, vel_rmse=m.vel_rmse, rate_rmse=m.rate_rmse,
                pos_rmse=m.pos_rmse, heading_rmse=m.heading_rmse,
                inslip_vel_rmse=m.inslip_vel_rmse, smoothness=m.smoothness
            ))
        end

        # --- full-trajectory replay (same gains, realistic noise) ---
        for tr in full_trajs
            m = eval_full(cfg, tr; seed=seed)
            push!(rows, (
                seed=seed, mode="full", name=tr.name, combo_idx=tr.combo_idx,
                mu=tr.mu, role=string(tr.role), t_start=0.0, t_end=0.0,
                score=m.score, vel_rmse=m.vel_rmse, rate_rmse=m.rate_rmse,
                pos_rmse=m.pos_rmse, heading_rmse=m.heading_rmse,
                inslip_vel_rmse=m.inslip_vel_rmse, smoothness=m.smoothness
            ))
        end
    end

    df = DataFrame(rows)
    mkpath(dirname(OUT_CSV))
    CSV.write(OUT_CSV, df)
    println("Wrote $(nrow(df)) rows to $OUT_CSV")

    # --- quick console summary: worst segments per seed ---
    println("\nWorst 5 segments per seed:")
    for seed in (1, 2, 3)
        sub = filter(r -> r.seed == seed && r.mode == "segment", df)
        println("  seed $seed (n=$(nrow(sub))):")
        sub_sorted = sort(sub, :score, rev=true)
        for r in eachrow(sub_sorted[1:min(5, nrow(sub_sorted)), :])
            println("    $(r.name)_c$(r.combo_idx) mu=$(r.mu) [$(round(r.t_start,digits=2))-$(round(r.t_end,digits=2))] " *
                    "score=$(round(r.score,digits=2)) vel=$(round(r.vel_rmse,digits=5)) pos=$(round(r.pos_rmse,digits=5))")
        end
    end

    # --- full-trajectory summary per seed ---
    println("\nFull-trajectory scores per seed:")
    for seed in (1, 2, 3)
        sub = filter(r -> r.seed == seed && r.mode == "full", df)
        total = sum(sub.score)
        println("  seed $seed: total=$(round(total,digits=2)) mean=$(round(total/nrow(sub),digits=2)) over $(nrow(sub)) trajectories")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
