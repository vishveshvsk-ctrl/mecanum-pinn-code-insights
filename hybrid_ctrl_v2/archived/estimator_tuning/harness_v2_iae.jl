# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/harness_v2_iae.jl
# (instructions/estimator-v2-iae-adaptive.md)
# =============================================================================
# `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl` is NEVER edited. This module
# adds a thin replay harness for `ESKFIAEEstimatorV2`, mirroring
# `HarnessV2Mod.run_and_log_replay_v2` exactly.
#
# Must be `include`d after `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl`
# (reuses `EstimatorLogV2`, `_load_replay_data_v2`, `_interp_scalar_v2`,
# `REPLAY_CACHE_V2`) and `hybrid_ctrl_v2/estimators_v2_iae.jl`.
# =============================================================================
module HarnessV2IAEMod

using StaticArrays
using LinearAlgebra
using TOML
using DataFrames
using Arrow

export build_estimator_v2_iae, run_and_log_replay_v2_iae

"""
    build_estimator_v2_iae(est_cfg, suite) -> ESKFIAEEstimatorV2

`est_cfg` is the decoded NamedTuple from `ParamSpaceV2IAEMod.apply_params_v2_iae!`
(10 shared ESKF-v2 dims + `tau_iae`, `kappa_iae`, `use_iae`). `iae_kind`,
`gamma_min`, `gamma_max` default to the struct values; `tau_slip`/`sigma_slip`/
`sigma_gyro_bias_rw` come from the struct's measured defaults unless
`est_cfg` explicitly overrides them.
"""
function build_estimator_v2_iae(est_cfg::NamedTuple, suite)
    kw = Dict{Symbol,Any}(:imu=>suite.imu, :enc=>suite.enc, :flow=>suite.flow)
    for k in (:P0_vel, :P0_yaw, :P0_heading, :P0_bias_acc, :P0_bias_gyro, :P0_slip, :P0_pos,
             :pose_Qn_heading, :pose_Qn_pos, :slip_R_inflate,
             :tau_slip, :sigma_slip, :sigma_gyro_bias_rw, :use_dhat, :rate_hz,
             :use_iae, :iae_kind, :tau_iae, :kappa_iae, :gamma_min, :gamma_max)
        haskey(est_cfg, k) && (kw[k] = getfield(est_cfg, k))
    end
    return Main.EstimatorModV2IAE.ESKFIAEEstimatorV2(; kw...)
end

"""
    run_and_log_replay_v2_iae(est_cfg, traj_entry, suite; seed=42, rate_hz=1000.0,
                              data_dir="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
                              gamma_trace=nothing) -> EstimatorLogV2

Replay a pre-simulated Arrow trajectory through `ESKFIAEEstimatorV2`. Identical
contract to `HarnessV2Mod.run_and_log_replay_v2` except the estimator is built
via `build_estimator_v2_iae` and the optical-flow update is routed through
`EstimatorModV2IAE.apply_flow_iae!`.

`gamma_trace` is an optional pre-allocated `Vector{Float64}` of length N (the
number of estimator ticks). When provided, the adaptive scale γ is recorded at
every tick so downstream diagnostics can report e.g. the fraction of time spent
above a threshold. The estimator log struct itself is unchanged.
"""
function run_and_log_replay_v2_iae(est_cfg::NamedTuple,
                                   traj_entry::NamedTuple,
                                   suite;
                                   seed::Int=42,
                                   rate_hz::Float64=1000.0,
                                   data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
                                   gamma_trace::Union{Nothing,Vector{Float64}}=nothing)
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    est = build_estimator_v2_iae(est_cfg, suite)
    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = false

    meta = (profile=String(traj_entry.name), combo_idx=Int(traj_entry.combo_idx),
            mu=Float64(traj_entry.mu), friction_case=1, friction_model=:lugre_adamov, chi=chi)
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("run_and_log_replay_v2_iae: data not found for $(traj_entry.name) " *
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

    if gamma_trace !== nothing
        length(gamma_trace) == N || error("run_and_log_replay_v2_iae: gamma_trace length $(length(gamma_trace)) != $N")
    end

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
            z_flow !== nothing && Main.EstimatorModV2IAE.apply_flow_iae!(bus, est, suite.flow, z_flow, params)
        end
        if suite.fix !== nothing
            z_fix = Main.SensorModV2.sample_pose_fix(suite.fix, u, t)
            z_fix !== nothing && Main.EstimatorMod.apply_pose_fix!(bus, est, suite.fix, z_fix)
        end

        gamma_trace !== nothing && (gamma_trace[i] = est.gamma_q)

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
