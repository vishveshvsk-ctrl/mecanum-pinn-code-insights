# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/harness_v2_slipobs.jl — HarnessV2SlipObsMod
# =============================================================================
# Thin additive harness for `ESKFSlipObsEstimatorV2`.  Mirrors
# `HarnessV2Mod.run_and_log_replay_v2` exactly except for estimator construction
# and the flow-update callback name (`apply_flow_sobs!`).
#
# Include-after: harness_v2.jl + estimators_v2_slipobs.jl.
# =============================================================================
module HarnessV2SlipObsMod

using StaticArrays
using LinearAlgebra
using TOML
using DataFrames
using Arrow

export build_estimator_v2_slipobs, run_and_log_replay_v2_slipobs

"""
    build_estimator_v2_slipobs(est_cfg::NamedTuple, suite) -> ESKFSlipObsEstimatorV2

Pass-through constructor for the shared 10 ESKF tuning params plus the
slip-observer-specific fields (`observer_kind`, `smo_k1`, `smo_k2`,
`smo_delta`, `eso_omega_o`, `rho_s`, `use_slipobs`).  Physical constants
(`tau_slip`, `sigma_slip`, `sigma_gyro_bias_rw`) are forwarded if present.
"""
function build_estimator_v2_slipobs(est_cfg::NamedTuple, suite)
    kw = Dict{Symbol,Any}(:imu=>suite.imu, :enc=>suite.enc, :flow=>suite.flow)
    for k in (:P0_vel, :P0_yaw, :P0_heading, :P0_bias_acc, :P0_bias_gyro, :P0_slip, :P0_pos,
              :pose_Qn_heading, :pose_Qn_pos, :slip_R_inflate,
              :observer_kind, :smo_k1, :smo_k2, :smo_delta, :eso_omega_o, :rho_s, :use_slipobs,
              :tau_slip, :sigma_slip, :sigma_gyro_bias_rw, :use_dhat, :rate_hz)
        haskey(est_cfg, k) && (kw[k] = getfield(est_cfg, k))
    end
    return Main.EstimatorModV2SlipObs.ESKFSlipObsEstimatorV2(; kw...)
end

"""
    run_and_log_replay_v2_slipobs(est_cfg, traj_entry, suite; seed=42, rate_hz=1000.0,
                                  data_dir="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov") -> EstimatorLogV2

Replay a pre-simulated Arrow trajectory through `ESKFSlipObsEstimatorV2`.
Identical to `HarnessV2Mod.run_and_log_replay_v2` except for the estimator
builder and the flow callback (`apply_flow_sobs!`), which re-anchors the
observer's open-loop IMU velocity on acceptance.
"""
function run_and_log_replay_v2_slipobs(est_cfg::NamedTuple,
                                       traj_entry::NamedTuple,
                                       suite;
                                       seed::Int=42,
                                       rate_hz::Float64=1000.0,
                                       data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    est = build_estimator_v2_slipobs(est_cfg, suite)
    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = false

    meta = (profile=String(traj_entry.name), combo_idx=Int(traj_entry.combo_idx),
            mu=Float64(traj_entry.mu), friction_case=1, friction_model=:lugre_adamov, chi=chi)
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("run_and_log_replay_v2_slipobs: data not found for $(traj_entry.name) " *
                                "combo $(traj_entry.combo_idx) mu $(traj_entry.mu) chi $chi: $arrow_path")

    data = Main.HarnessV2Mod._load_replay_data_v2(arrow_path, data_dir)
    ts, Vx_arr, Vy_arr, psidot_arr, psi_arr = data.ts, data.Vx_arr, data.Vy_arr, data.psidot_arr, data.psi_arr
    theta_arrs, omega_arrs, Xo_arr, Yo_arr = data.theta_arrs, data.omega_arrs, data.Xo_arr, data.Yo_arr
    dVx_arr, dVy_arr, dpsidot_arr = data.dVx_arr, data.dVy_arr, data.dpsidot_arr

    dt = 1.0 / rate_hz
    T = ts[end]
    ticks = collect(range(0.0, T; step=dt))
    !isempty(ticks) && ticks[end] > T + 1e-12 && pop!(ticks)
    N = length(ticks)

    v_true = Matrix{Float64}(undef, 3, N); v_hat = Matrix{Float64}(undef, 3, N)
    pose_true = Matrix{Float64}(undef, 3, N); pose_hat = Matrix{Float64}(undef, 3, N)
    d_hat_log = Matrix{Float64}(undef, 3, N); slip = Vector{Float64}(undef, N)

    u = zeros(30); du = zeros(30)

    for (i, t) in enumerate(ticks)
        u[1] = Main.HarnessV2Mod._interp_scalar_v2(ts, Vx_arr, t)
        u[2] = Main.HarnessV2Mod._interp_scalar_v2(ts, Vy_arr, t)
        u[3] = Main.HarnessV2Mod._interp_scalar_v2(ts, psidot_arr, t)
        u[4] = Main.HarnessV2Mod._interp_scalar_v2(ts, psi_arr, t)
        for j in 1:4
            u[4+j] = Main.HarnessV2Mod._interp_scalar_v2(ts, theta_arrs[j], t)
            u[8+j] = Main.HarnessV2Mod._interp_scalar_v2(ts, omega_arrs[j], t)
        end
        u[17] = Main.HarnessV2Mod._interp_scalar_v2(ts, Xo_arr, t)
        u[18] = Main.HarnessV2Mod._interp_scalar_v2(ts, Yo_arr, t)

        du[1] = Main.HarnessV2Mod._interp_scalar_v2(ts, dVx_arr, t)
        du[2] = Main.HarnessV2Mod._interp_scalar_v2(ts, dVy_arr, t)
        du[3] = Main.HarnessV2Mod._interp_scalar_v2(ts, dpsidot_arr, t)

        a_x, a_y, g_z = Main.SensorModV2.simulate_imu(suite.imu, u, du, t)
        _theta, omega = Main.SensorModV2.simulate_encoders(suite.enc, u)
        y = (a_x=a_x, a_y=a_y, g_z=g_z, ω=omega)
        Main.EstimatorMod.estimator_update!(bus, y, est, params, dt)

        if suite.flow !== nothing
            z_flow = Main.SensorModV2.simulate_flow(suite.flow, u, t)
            z_flow !== nothing && Main.EstimatorModV2SlipObs.apply_flow_sobs!(bus, est, suite.flow, z_flow, params)
        end
        if suite.fix !== nothing
            z_fix = Main.SensorModV2.sample_pose_fix(suite.fix, u, t)
            z_fix !== nothing && Main.EstimatorMod.apply_pose_fix!(bus, est, suite.fix, z_fix)
        end

        v_true[:, i]    .= u[1], u[2], u[3]
        pose_true[:, i] .= u[17], u[18], u[4]
        v_hat[:, i]     .= bus.xhat[1], bus.xhat[2], bus.xhat[3]
        pose_hat[:, i]  .= bus.xhat[5], bus.xhat[6], bus.xhat[4]
        d_hat_log[:, i] .= bus.d_hat
        Hw = est.wheel_H
        slip[i] = norm(Hw \ SVector(u[9], u[10], u[11], u[12]) - SVector(u[1], u[2], u[3]))
    end

    return Main.HarnessV2Mod.EstimatorLogV2(ticks, v_true, v_hat, pose_true, pose_hat, d_hat_log, slip,
                                            string(traj_entry.name), traj_entry.ref_type,
                                            traj_entry.run_mode, seed)
end

end # module
