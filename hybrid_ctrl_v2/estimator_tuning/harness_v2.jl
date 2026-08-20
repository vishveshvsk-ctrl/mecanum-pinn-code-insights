# =============================================================================
# hybrid_ctrl_v2/estimator_tuning/harness_v2.jl
# (instructions/sensors-suite-consolidation-and-physical-noise.md)
# =============================================================================
# `tuning/harness.jl` is NEVER edited — every existing caller keeps using the
# original `run_and_log`/`_build_estimator` (22-dim ESKF) unaffected. This is
# a NEW, additive harness for ESKFEstimatorV2, mirroring `run_and_log`'s
# structure but wired through `SchedulerModV2.run_hybrid_v2` +
# `SensorModV2.build_suite`.
#
# Must be `include`d after tune_controller.jl, sensors_v2.jl, estimators_v2.jl,
# scheduler_v2.jl, and param_space_v2.jl.
# =============================================================================
module HarnessV2Mod

using StaticArrays
using LinearAlgebra
using TOML
using DataFrames
using Arrow
using Random

export EstimatorLogV2, run_and_log_v2, build_estimator_v2, build_estimator_v3, run_and_log_replay_v2,
       segmented_replay_trajset, mu0p5_train12_replay_trajset

"""
    EstimatorLogV2

Same shape as `TuningHarnessMod.EstimatorLog` (drop-in compatible with any
downstream metric code expecting that field set).
"""
struct EstimatorLogV2
    time::Vector{Float64}
    v_true::Matrix{Float64}
    v_hat::Matrix{Float64}
    pose_true::Matrix{Float64}
    pose_hat::Matrix{Float64}
    d_hat::Matrix{Float64}
    slip::Vector{Float64}
    traj_name::String
    ref_type::Symbol
    run_mode::Symbol
    seed::Int
end

"""
    build_estimator_v2(est_cfg, suite) -> ESKFEstimatorV2

`est_cfg` is `ParamSpaceV2Mod.apply_params_v2!`'s / `ParamSpaceV3Mod.apply_params_v3!`'s
decoded NamedTuple. The v3 space additionally passes the update-rate knobs
`alpha_acc`/`alpha_yaw`/`grip_slip_scale`/`r_boost` (all guarded by `haskey`,
so v2 callers are unaffected). `tau_slip`/`sigma_slip`/
`sigma_gyro_bias_rw` come from the struct's own measured defaults unless
`est_cfg` explicitly overrides them.
"""
function build_estimator_v2(est_cfg::NamedTuple, suite)
    kw = Dict{Symbol,Any}(:imu=>suite.imu, :enc=>suite.enc, :flow=>suite.flow)
    for k in (:P0_vel, :P0_yaw, :P0_heading, :P0_bias_acc, :P0_bias_gyro, :P0_slip, :P0_pos,
             :pose_Qn_heading, :pose_Qn_pos, :slip_R_inflate,
             :alpha_acc, :alpha_yaw, :grip_slip_scale, :r_boost,
             :tau_slip, :sigma_slip, :sigma_gyro_bias_rw, :use_dhat, :rate_hz)
        haskey(est_cfg, k) && (kw[k] = getfield(est_cfg, k))
    end
    return Main.EstimatorModV2.ESKFEstimatorV2(; kw...)
end

"""
    build_estimator_v3(est_cfg, suite) -> ESKFEstimatorV3

ESKFEstimatorV3 (13-dim, yaw-accel state) builder — same pass-through as
`build_estimator_v2` plus the V3-specific `q_alpha` (tunable) and `P0_alpha`
(pinned) keys. `est_cfg` is `ParamSpaceV4Mod.apply_params_v4!`'s decoded
NamedTuple. Requires `estimators_v3.jl` to be included.
"""
function build_estimator_v3(est_cfg::NamedTuple, suite)
    kw = Dict{Symbol,Any}(:imu=>suite.imu, :enc=>suite.enc, :flow=>suite.flow)
    for k in (:P0_vel, :P0_yaw, :P0_heading, :P0_bias_acc, :P0_bias_gyro, :P0_slip, :P0_pos,
             :P0_alpha, :pose_Qn_heading, :pose_Qn_pos, :slip_R_inflate,
             :alpha_acc, :alpha_yaw, :grip_slip_scale, :r_boost, :q_alpha,
             :tau_slip, :sigma_slip, :sigma_gyro_bias_rw, :use_dhat, :rate_hz)
        haskey(est_cfg, k) && (kw[k] = getfield(est_cfg, k))
    end
    return Main.EstimatorModV3.ESKFEstimatorV3(; kw...)
end

"""
    run_and_log_v2(est_cfg, traj_entry, nominal_ctrl_cfg; seed=42, flow=true, fix_tier=:docking) -> EstimatorLogV2

Run one closed-loop sim via `SchedulerModV2.run_hybrid_v2` with a fixed
nominal controller and an `ESKFEstimatorV2` candidate. `traj_entry` follows
the same NamedTuple contract as `TuningHarnessMod.run_and_log`'s
(name, profile_toml, combo_idx, config_dir, mu, ref_type, run_mode).
"""
function run_and_log_v2(est_cfg::NamedTuple,
                        traj_entry::NamedTuple,
                        nominal_ctrl_cfg::Main.HybridConfigMod.HybridConfig;
                        seed::Int=42, flow::Bool=true, fix_tier::Symbol=:docking)
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = get(base, "physics", Dict())["chi"]
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    suite = Main.SensorModV2.build_suite(:default; seed=seed, flow=flow, fix_tier=fix_tier)
    est = build_estimator_v2(est_cfg, suite)

    cfg = Main.HybridConfigMod.HybridConfig(
        tracking=traj_entry.run_mode, use_dhat=nominal_ctrl_cfg.use_dhat,
        use_asmc=nominal_ctrl_cfg.use_asmc, use_mpc=nominal_ctrl_cfg.use_mpc,
        use_pid=nominal_ctrl_cfg.use_pid, fuzzy=nominal_ctrl_cfg.fuzzy,
        fixed_weights=nominal_ctrl_cfg.fixed_weights, use_pose_fix=true,
        pose_fix_tier=fix_tier, f_est=nominal_ctrl_cfg.f_est, f_mpc=nominal_ctrl_cfg.f_mpc,
        f_pid=nominal_ctrl_cfg.f_pid, f_fuzzy=nominal_ctrl_cfg.f_fuzzy, f_mix=nominal_ctrl_cfg.f_mix,
        sensor_seed=seed, reltol=nominal_ctrl_cfg.reltol, abstol_bristle=nominal_ctrl_cfg.abstol_bristle,
        dtmax=nominal_ctrl_cfg.dtmax, solver_symbol=nominal_ctrl_cfg.solver_symbol,
        saveat_hz=nominal_ctrl_cfg.saveat_hz)

    prof = TOML.parsefile(joinpath(traj_entry.config_dir, "profiles", traj_entry.profile_toml))["profile"]
    pcfg = Main.Profiles.resolve_profile(prof; combo_idx=get(traj_entry, :combo_idx, nothing))
    ref = Main.Profiles.build(String(prof["builder"]), pcfg)
    if get(traj_entry, :adapt, false)
        ref = Main.Profiles.velref_to_posref(ref)
    end

    sol, bus = Main.SchedulerModV2.run_hybrid_v2(
        cfg, params, Symbol(traj_entry.name); chi=chi, suite=suite, est=est,
        config_dir=traj_entry.config_dir, profile_toml=traj_entry.profile_toml,
        ref=ref, return_bus=true)

    T_total = ref.T_total
    if sol.t[end] < 0.9 * T_total
        error("run_and_log_v2: $(traj_entry.name) sim aborted early " *
              "(t_end=$(round(sol.t[end],digits=4))s < 90% of T_total=$(T_total)s)")
    end

    probe = get(Main.SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])
    Main.SchedulerMod.clear_probe_log!(bus)
    isempty(probe) && error("run_and_log_v2: no per-tick probe log for $(traj_entry.name)")

    ticks = [p.t for p in probe]
    v_true = hcat([Vector(sol(t)[1:3]) for t in ticks]...)
    pose_true = hcat([[sol(t)[17], sol(t)[18], sol(t)[4]] for t in ticks]...)
    v_hat = hcat([Vector(p.xhat[1:3]) for p in probe]...)
    pose_hat = hcat([[p.xhat[5], p.xhat[6], p.xhat[4]] for p in probe]...)
    d_hat = hcat([Vector(p.d_hat) for p in probe]...)
    Hw = Main.EstimatorMod._wheel_jacobian(params)
    slip = [norm(Hw \ SVector(p.u[9:12]...) - SVector(p.u[1:3]...)) for p in probe]

    return EstimatorLogV2(ticks, v_true, v_hat, pose_true, pose_hat, d_hat, slip,
                          string(traj_entry.name), traj_entry.ref_type,
                          traj_entry.run_mode, seed)
end

# =============================================================================
# Replay-based estimator evaluation (no plant ODE solve) — mirrors
# `TuningHarnessMod.run_and_log_replay` (tuning/harness.jl), ported to
# SensorModV2's per-modality synthesis + ESKFEstimatorV2. Reads a
# PRE-SIMULATED Arrow file from the existing data/ sweep (per-user
# direction: tune the estimator by replaying it over trajectories that
# already exist, not by re-solving the plant ODE per candidate).
# =============================================================================
const REPLAY_CACHE_V2 = Dict{String,NamedTuple}()
const REPLAY_CACHE_V2_LOCK = ReentrantLock()

function _interp_scalar_v2(ts::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, t::Real)
    n = length(ts)
    n == 0 && error("_interp_scalar_v2: empty time vector")
    t <= ts[1] && return ys[1]
    t >= ts[end] && return ys[end]
    i = searchsortedfirst(ts, t)
    ts[i] == t && return ys[i]
    t0, t1 = ts[i-1], ts[i]
    y0, y1 = ys[i-1], ys[i]
    dt = t1 - t0
    iszero(dt) && return y0
    return y0 + (y1 - y0) * (t - t0) / dt
end

function _finite_diff_v2(y::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    n = length(y)
    n == length(t) || error("_finite_diff_v2: length mismatch")
    dy = similar(y, Float64)
    n == 1 && (dy[1] = 0.0; return dy)
    dy[1] = (y[2] - y[1]) / (t[2] - t[1])
    for i in 2:n-1
        dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
    end
    dy[n] = (y[n] - y[n-1]) / (t[n] - t[n-1])
    return dy
end

function _load_replay_data_v2(arrow_path::String, data_dir::String)
    cached = lock(REPLAY_CACHE_V2_LOCK) do
        get(REPLAY_CACHE_V2, arrow_path, nothing)
    end
    cached !== nothing && return cached

    df = DataFrame(Arrow.Table(arrow_path))
    _col(v) = Float64.(collect(v))

    ts         = _col(df.time)
    Vx_arr     = _col(df.Vx)
    Vy_arr     = _col(df.Vy)
    psidot_arr = _col(df.psi_dot)
    psi_arr    = _col(df.psi)
    theta_arrs = (_col(df.theta1), _col(df.theta2), _col(df.theta3), _col(df.theta4))
    omega_arrs = (_col(df.w1), _col(df.w2), _col(df.w3), _col(df.w4))
    Xo_arr     = _col(df.Xo)
    Yo_arr     = _col(df.Yo)

    stem, _ = splitext(basename(arrow_path))
    accel_path = joinpath(data_dir, "accel", "$(stem)_accel.arrow")
    if isfile(accel_path)
        adf = DataFrame(Arrow.Table(accel_path))
        dVx_arr = _col(adf.dVx); dVy_arr = _col(adf.dVy); dpsidot_arr = _col(adf.dpsidot)
    else
        dVx_arr = _finite_diff_v2(Vx_arr, ts)
        dVy_arr = _finite_diff_v2(Vy_arr, ts)
        dpsidot_arr = _finite_diff_v2(psidot_arr, ts)
    end

    data = (ts=ts, Vx_arr=Vx_arr, Vy_arr=Vy_arr, psidot_arr=psidot_arr, psi_arr=psi_arr,
            theta_arrs=theta_arrs, omega_arrs=omega_arrs, Xo_arr=Xo_arr, Yo_arr=Yo_arr,
            dVx_arr=dVx_arr, dVy_arr=dVy_arr, dpsidot_arr=dpsidot_arr)
    lock(REPLAY_CACHE_V2_LOCK) do
        REPLAY_CACHE_V2[arrow_path] = data
    end
    return data
end

"""
    run_and_log_replay_v2(est_cfg, traj_entry, suite; seed=42, rate_hz=1000.0,
                          data_dir="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov") -> EstimatorLogV2

Replay a PRE-SIMULATED Arrow trajectory (from the existing `data/` sweep — no
plant/controller ODE solve) through a candidate `ESKFEstimatorV2`. The saved
true state is interpolated to the estimator tick rate; IMU/encoder/flow
measurements are resynthesized per tick via `SensorModV2`; pose fixes are
sampled from the true pose via `suite.fix`. Mirrors
`TuningHarnessMod.run_and_log_replay`, generalized to the 4-modality suite.

`traj_entry.name` MUST be the literal profile string used in the Arrow
filename contract (e.g. "ellipse", "octagon"), NOT a descriptive label —
`Main.DataStore.expected_output` resolves the path from it directly.

The candidate estimator is fully decoupled from the trajectory here: since
the true state comes from a FIXED pre-recorded run, changing `est_cfg` only
changes the filter's own P0/Q/R-derived parameters, never the excitation —
this is what makes many candidate evaluations cheap (one Arrow read, cached,
then reused across every candidate).
"""
function run_and_log_replay_v2(est_cfg::NamedTuple,
                               traj_entry::NamedTuple,
                               suite;
                               seed::Int=42,
                               rate_hz::Float64=1000.0,
                               data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
                               t_window::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                               builder=build_estimator_v2)
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    est = builder(est_cfg, suite)
    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = false

    profile_name = get(traj_entry, :profile, traj_entry.name)
    meta = (profile=String(profile_name), combo_idx=Int(traj_entry.combo_idx),
            mu=Float64(traj_entry.mu), friction_case=1, friction_model=:lugre_adamov, chi=chi)
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("run_and_log_replay_v2: data not found for $(traj_entry.name) " *
                                "combo $(traj_entry.combo_idx) mu $(traj_entry.mu) chi $chi: $arrow_path")

    data = _load_replay_data_v2(arrow_path, data_dir)
    ts, Vx_arr, Vy_arr, psidot_arr, psi_arr = data.ts, data.Vx_arr, data.Vy_arr, data.psidot_arr, data.psi_arr
    theta_arrs, omega_arrs, Xo_arr, Yo_arr = data.theta_arrs, data.omega_arrs, data.Xo_arr, data.Yo_arr
    dVx_arr, dVy_arr, dpsidot_arr = data.dVx_arr, data.dVy_arr, data.dpsidot_arr

    dt = 1.0 / rate_hz
    T = ts[end]
    t0, t1 = something(t_window, (0.0, T))
    t0 = clamp(t0, 0.0, T)
    t1 = clamp(t1, t0, T)
    ticks = collect(range(t0, t1; step=dt))
    !isempty(ticks) && ticks[end] > t1 + 1e-12 && pop!(ticks)
    N = length(ticks)

    v_true = Matrix{Float64}(undef, 3, N); v_hat = Matrix{Float64}(undef, 3, N)
    pose_true = Matrix{Float64}(undef, 3, N); pose_hat = Matrix{Float64}(undef, 3, N)
    d_hat_log = Matrix{Float64}(undef, 3, N); slip = Vector{Float64}(undef, N)

    u = zeros(30); du = zeros(30)

    for (i, t) in enumerate(ticks)
        u[1] = _interp_scalar_v2(ts, Vx_arr, t)
        u[2] = _interp_scalar_v2(ts, Vy_arr, t)
        u[3] = _interp_scalar_v2(ts, psidot_arr, t)
        u[4] = _interp_scalar_v2(ts, psi_arr, t)
        for j in 1:4
            u[4+j] = _interp_scalar_v2(ts, theta_arrs[j], t)
            u[8+j] = _interp_scalar_v2(ts, omega_arrs[j], t)
        end
        u[17] = _interp_scalar_v2(ts, Xo_arr, t)
        u[18] = _interp_scalar_v2(ts, Yo_arr, t)

        du[1] = _interp_scalar_v2(ts, dVx_arr, t)
        du[2] = _interp_scalar_v2(ts, dVy_arr, t)
        du[3] = _interp_scalar_v2(ts, dpsidot_arr, t)

        a_x, a_y, g_z = Main.SensorModV2.simulate_imu(suite.imu, u, du, t)
        _theta, omega = Main.SensorModV2.simulate_encoders(suite.enc, u)
        y = (a_x=a_x, a_y=a_y, g_z=g_z, ω=omega)
        Main.EstimatorMod.estimator_update!(bus, y, est, params, dt)

        if suite.flow !== nothing
            z_flow = Main.SensorModV2.simulate_flow(suite.flow, u, t)
            z_flow !== nothing && Main.EstimatorModV2.apply_flow!(bus, est, suite.flow, z_flow, params)
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

    return EstimatorLogV2(ticks, v_true, v_hat, pose_true, pose_hat, d_hat_log, slip,
                          string(traj_entry.name), traj_entry.ref_type,
                          traj_entry.run_mode, seed)
end

# =============================================================================
# Segmented replay manifest — two random T/4 windows per trajectory
# =============================================================================
"""
    _trajectory_duration(traj_entry; data_dir) -> Float64

Read only the `time` column of the Arrow file for `traj_entry` and return the
final timestamp. Used once per seed to place random replay windows.
"""
function _trajectory_duration(traj_entry::NamedTuple;
                              data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    meta = (profile=String(traj_entry.name), combo_idx=Int(traj_entry.combo_idx),
            mu=Float64(traj_entry.mu), friction_case=1, friction_model=:lugre_adamov, chi=chi)
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("_trajectory_duration: data not found for $(traj_entry.name): $arrow_path")
    return Float64(Arrow.Table(arrow_path).time[end])
end

"""
    segmented_replay_trajset(seed::Int; segment_frac=0.25, data_dir)

Return a 22-entry manifest (2 segments × 11 trajectories). For each trajectory
of duration T, two non-overlapping windows of length segment_frac·T are drawn
deterministically from `seed`:
  - window 1 starts in [0, T/2 - L]   (lies inside the first half)
  - window 2 starts in [T/2, T - L]   (lies inside the second half)
where L = segment_frac·T. The windows are stored in each entry as
`t_window = (t_start, t_end)` for `run_and_log_replay_v2`.
"""
function segmented_replay_trajset(seed::Int; segment_frac::Float64=0.25,
                                  data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    base = Main.ReplayTrajSetMod.replay_trajset()
    rng = MersenneTwister(hash((seed, :replay_segments)))
    segs = []
    for tr in base
        T = _trajectory_duration(tr; data_dir=data_dir)
        L = segment_frac * T
        # First half window: start in [0, T/2 - L], end in [L, T/2]
        t1_start = rand(rng) * (T/2 - L)
        # Second half window: start in [T/2, T - L], end in [T/2 + L, T]
        t2_start = T/2 + rand(rng) * (T/2 - L)
        push!(segs, merge(tr, (t_window=(t1_start, t1_start + L),)))
        push!(segs, merge(tr, (t_window=(t2_start, t2_start + L),)))
    end
    return segs
end

# =============================================================================
# Controller-tuning-aligned mu=0.5 replay manifest
# =============================================================================
"""
    _toml_to_profile(toml::String) -> String

Strip the `_mu_0p5.toml` suffix from a controller-tuning profile TOML name to
recover the base profile string used in the Arrow filename contract.
"""
function _toml_to_profile(toml::String)
    suffix = "_mu_0p5.toml"
    endswith(toml, suffix) || error("_toml_to_profile: expected '*$suffix', got '$toml'")
    return String(toml[1:end-length(suffix)])
end

"""
    mu0p5_train12_replay_trajset(run_dir::AbstractString="trajectory_files_run_0p5_main")

Return the 12-trajectory `train12` set from `TrajSetsMod`, remapped for replay:
- `name` stays descriptive (e.g. "octagon_easy") for logging.
- `profile` is added as the base Arrow profile string (e.g. "octagon").
- `mu=0.5` and `combo_idx` are preserved.
All entries are `run_mode=:pose` in controller tuning, but replay is reference-
agnostic: it only needs the recorded state Arrow file.
"""
function mu0p5_train12_replay_trajset(run_dir::AbstractString="trajectory_files_run_0p5_main")
    base = Main.TrajSetsMod.trajset(:train12, run_dir)
    return [merge(tr, (profile=_toml_to_profile(tr.profile_toml),)) for tr in base]
end

end # module
