# =============================================================================
# tuning/harness.jl — closed-loop run + per-tick estimator log
# =============================================================================
module TuningHarnessMod

using StaticArrays
using LinearAlgebra
using TOML
using DataFrames
using Arrow

export EstimatorLog, slip_indicator, run_and_log, run_and_log_replay

# In-memory cache for replay data: one entry per Arrow file.
# This avoids re-reading the same compressed Arrow file on every evaluation.
const REPLAY_CACHE = Dict{String, NamedTuple}()
const REPLAY_CACHE_LOCK = ReentrantLock()

"""
    EstimatorLog

Aligned per-run time series produced by `run_and_log`.

  time      :: Vector{Float64}          estimator tick instants
  v_true    :: Matrix{Float64}          3×N  [Vx, Vy, ψ̇] true
  v_hat     :: Matrix{Float64}          3×N  [V̂x, V̂y, ψ̂̇]
  pose_true :: Matrix{Float64}          3×N  [Xo, Yo, ψ]
  pose_hat  :: Matrix{Float64}          3×N  [X̂o, Ŷo, ψ̂]
  d_hat     :: Matrix{Float64}          3×N  disturbance estimate
  slip      :: Vector{Float64}          ground-truth slip magnitude
  traj_name :: String
  ref_type  :: Symbol
  run_mode  :: Symbol
  seed      :: Int
"""
struct EstimatorLog
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
    slip_indicator(u, params) -> Float64

Ground-truth slip magnitude `‖ Hω \\ ω_true − [Vx, Vy, ψ̇]_true ‖` at one
timestep.  This quantifies the no-slip wheel-map corruption the estimator must
survive.  Evaluation-only; never fed to the estimator.
"""
function slip_indicator(u::AbstractVector, params)
    Hω = Main.EstimatorMod._wheel_jacobian(params)      # 4×3
    ω_true = SVector(u[9], u[10], u[11], u[12])
    v_body = SVector(u[1], u[2], u[3])
    v_from_wheels = Hω \ ω_true
    return norm(v_from_wheels - v_body)
end

function _build_cfg(est_cfg::NamedTuple, traj_entry, nominal_ctrl_cfg::Main.HybridConfigMod.HybridConfig, seed::Int)
    tracking = traj_entry.run_mode
    # Exteroceptive pose fix: always on for :pose runs (docking tier); velref
    # entries may opt in by carrying pose_fix_tier (e.g. spin_creep, where the
    # wheel channel is corrupt for the whole maneuver and the 100 Hz absolute
    # anchor is what bounds drift).
    use_pose_fix = tracking == :pose || haskey(traj_entry, :pose_fix_tier)
    pose_fix_tier = get(traj_entry, :pose_fix_tier,
                        tracking == :pose ? :docking : :transit)
    return Main.HybridConfigMod.HybridConfig(
        tracking       = tracking,
        estimator      = est_cfg.estimator,
        use_dhat       = nominal_ctrl_cfg.use_dhat,
        use_asmc       = nominal_ctrl_cfg.use_asmc,
        use_mpc        = nominal_ctrl_cfg.use_mpc,
        use_pid        = nominal_ctrl_cfg.use_pid,
        fuzzy          = nominal_ctrl_cfg.fuzzy,
        fixed_weights  = nominal_ctrl_cfg.fixed_weights,
        use_pose_fix   = use_pose_fix,
        pose_fix_tier  = pose_fix_tier,
        f_est          = est_cfg.rate_hz,
        f_mpc          = nominal_ctrl_cfg.f_mpc,
        f_pid          = nominal_ctrl_cfg.f_pid,
        f_fuzzy        = nominal_ctrl_cfg.f_fuzzy,
        f_mix          = nominal_ctrl_cfg.f_mix,
        sensor_seed    = seed,
        reltol         = nominal_ctrl_cfg.reltol,
        abstol_bristle = nominal_ctrl_cfg.abstol_bristle,
        dtmax          = nominal_ctrl_cfg.dtmax,
        solver_symbol  = nominal_ctrl_cfg.solver_symbol,
        saveat_hz      = nominal_ctrl_cfg.saveat_hz,
    )
end

function _build_estimator(est_cfg::NamedTuple)
    if est_cfg.estimator == :kalman
        return Main.EstimatorMod.KalmanEstimator(
            Qn              = est_cfg.Qn,
            Rn_base         = est_cfg.Rn_base,
            bias_Qn         = get(est_cfg, :bias_Qn, Diagonal(SVector(1e-4, 1e-4))),
            rate_hz         = est_cfg.rate_hz,
            P0_scale        = est_cfg.P0_scale,
            slip_R_inflate  = est_cfg.slip_R_inflate,
            slip_threshold  = est_cfg.slip_threshold,
            zupt_threshold  = get(est_cfg, :zupt_threshold, 0.02),
            use_dhat        = get(est_cfg, :use_dhat, false),
        )
    elseif est_cfg.estimator == :kalman_imm
        return Main.EstimatorMod.IMMKalmanEstimator(
            Qn              = est_cfg.Qn,
            Rn_base         = est_cfg.Rn_base,
            bias_Qn         = get(est_cfg, :bias_Qn, Diagonal(SVector(1e-4, 1e-4))),
            slip_Qn         = get(est_cfg, :slip_Qn, Diagonal(SVector(1e-2, 1e-2))),
            rate_hz         = est_cfg.rate_hz,
            P0_scale        = est_cfg.P0_scale,
            slip_R_inflate  = est_cfg.slip_R_inflate,
            slip_threshold  = est_cfg.slip_threshold,
            zupt_threshold  = get(est_cfg, :zupt_threshold, 0.02),
            alpha_acc       = get(est_cfg, :alpha_acc, 1.0),
            alpha_yaw       = get(est_cfg, :alpha_yaw, 0.5),
            r_boost         = get(est_cfg, :r_boost, 10.0),
            p_stay_grip     = get(est_cfg, :p_stay_grip, 0.95),
            p_stay_slip     = get(est_cfg, :p_stay_slip, 0.9),
            gyro_bias_Qn    = get(est_cfg, :gyro_bias_Qn, 1e-6),
            pose_Qn         = get(est_cfg, :pose_Qn, 1e-6),
            grip_slip_scale = get(est_cfg, :grip_slip_scale, 1e-3),
            nis_thresh      = get(est_cfg, :nis_thresh, 9.21),
            use_dhat        = get(est_cfg, :use_dhat, false),
        )
    elseif est_cfg.estimator == :eskf
        return Main.EstimatorMod.ESKFEstimator(
            Qn              = est_cfg.Qn,
            Rn_base         = est_cfg.Rn_base,
            bias_Qn         = get(est_cfg, :bias_Qn, Diagonal(SVector(1e-4, 1e-4))),
            slip_Qn         = get(est_cfg, :slip_Qn, Diagonal(SVector(1e-2, 1e-2))),
            gyro_bias_Qn    = get(est_cfg, :gyro_bias_Qn, 1e-6),
            pose_Qn         = get(est_cfg, :pose_Qn, 1e-6),
            pose_slip_gain  = get(est_cfg, :pose_slip_gain, 10.0),
            rate_hz         = est_cfg.rate_hz,
            P0_scale        = est_cfg.P0_scale,
            slip_R_inflate  = est_cfg.slip_R_inflate,
            slip_threshold  = est_cfg.slip_threshold,
            zupt_threshold  = get(est_cfg, :zupt_threshold, 0.02),
            alpha_acc       = get(est_cfg, :alpha_acc, 1.0),
            alpha_yaw       = get(est_cfg, :alpha_yaw, 0.5),
            r_boost         = get(est_cfg, :r_boost, 10.0),
            nis_thresh      = get(est_cfg, :nis_thresh, 9.21),
            grip_slip_scale = get(est_cfg, :grip_slip_scale, 1e-3),
            use_dhat        = get(est_cfg, :use_dhat, false),
        )
    elseif est_cfg.estimator == :smo
        return Main.EstimatorMod.SMOEstimator(
            L               = est_cfg.L,
            K               = est_cfg.K,
            δ               = est_cfg.delta,
            rate_hz         = est_cfg.rate_hz,
            slip_gate_thresh= est_cfg.slip_gate_thresh,
            zupt_threshold  = est_cfg.zupt_threshold,
            bias_gain       = get(est_cfg, :bias_gain, SVector(0.5, 0.5)),
            use_dhat        = get(est_cfg, :use_dhat, true),
        )
    else
        error("_build_estimator: unknown estimator $(est_cfg.estimator)")
    end
end

"""
    run_and_log(est_cfg, traj_entry, nominal_ctrl_cfg; seed=42) -> EstimatorLog

Run one closed-loop sim via `SchedulerMod.run_hybrid` with a fixed nominal
controller and the candidate estimator.  Uses per-tick probe logging when
available; otherwise falls back to the final bus state as a constant placeholder
(TODO(rewrite): remove fallback once per-tick probe logging is mandatory).

  est_cfg          :: NamedTuple     decoder output (estimator + kwargs)
  traj_entry       :: NamedTuple     one TuningSubset entry
  nominal_ctrl_cfg :: HybridConfig   fixed nominal controller configuration
  seed             :: Int            sensor RNG seed
"""
# Convert a PosRef to a VelRef by rotating global velocity into the body frame.
# Accel feedforwards are set to zero (placeholder until the pose-mode controller
# is wired; the velocity controller still runs safely without them).
function _posref_to_velref(ref::Main.Profiles.PosRef)
    fVx(t::Float64)  = ref.Vxo(t) * cos(ref.psi(t)) + ref.Vyo(t) * sin(ref.psi(t))
    fVy(t::Float64)  = -ref.Vxo(t) * sin(ref.psi(t)) + ref.Vyo(t) * cos(ref.psi(t))
    fpsi(t::Float64) = ref.psi(t)
    fWz(t::Float64)  = ref.om(t)
    zero_acc(t::Float64) = 0.0
    return Main.Profiles.VelRef(
        Main.Profiles.Getter(fVx),  Main.Profiles.Getter(fVy),
        Main.Profiles.Getter(fpsi), Main.Profiles.Getter(fWz),
        Main.Profiles.Getter(zero_acc), Main.Profiles.Getter(zero_acc),
        Main.Profiles.Getter(zero_acc),
        ref.tstops, ref.T_total)
end

function run_and_log(est_cfg::NamedTuple,
                     traj_entry::NamedTuple,
                     nominal_ctrl_cfg::Main.HybridConfigMod.HybridConfig;
                     seed::Int=42)
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = get(base, "physics", Dict())["chi"]
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    cfg = _build_cfg(est_cfg, traj_entry, nominal_ctrl_cfg, seed)
    est = _build_estimator(est_cfg)

    # PosRef fallback: when the pose-mode controller is not wired, run the
    # trajectory velocity-tracked by converting the PosRef to a VelRef.
    # Combo-pinned build: entries may carry combo_idx (whitelist-pinned
    # trajectories); nothing falls back to resolve_profile's random pick.
    prof = TOML.parsefile(joinpath(traj_entry.config_dir, "profiles",
                                   traj_entry.profile_toml))["profile"]
    pcfg = Main.Profiles.resolve_profile(prof;
                                         combo_idx=get(traj_entry, :combo_idx, nothing))
    ref = Main.Profiles.build(String(prof["builder"]), pcfg)
    if traj_entry.ref_type == :posref && traj_entry.run_mode == :velocity && ref isa Main.Profiles.PosRef
        ref = _posref_to_velref(ref)
    end

    # Run closed-loop simulation; request the bus so we can retrieve per-tick probes.
    sol, _df, bus = Main.SchedulerMod.run_hybrid(
        cfg, params, Symbol(traj_entry.name);
        chi=chi, friction_case=1, config_dir=traj_entry.config_dir,
        profile_toml=traj_entry.profile_toml, return_bus=true, est=est, ref=ref)

    # Truncation guard: a config that destabilises the closed loop makes the
    # solver abort early (dt_epsilon), and scoring the handful of near-zero
    # ticks that remain would REWARD instability.  Treat any run that did not
    # cover (almost) the full reference duration as a failed evaluation so the
    # executor returns the sentinel penalty instead.
    T_total = ref.T_total
    if sol.t[end] < 0.9 * T_total
        error("run_and_log: $(traj_entry.name) sim aborted early " *
              "(t_end=$(round(sol.t[end], digits=4)) s < 90% of T_total=$(T_total) s); " *
              "candidate config is unstable")
    end

    probe = get(Main.SchedulerMod.ESTIMATOR_PROBE_LOG, objectid(bus), NamedTuple[])

    if isempty(probe)
        # TODO(rewrite): per-tick probe logging unavailable; use final bus state.
        @warn "No per-tick probe log found for $(traj_entry.name); using constant placeholder."
        f_est = cfg.f_est
        T = sol.t[end]
        ticks = collect(range(0.0, T; step=1.0 / f_est))
        N = length(ticks)
        v_true = hcat([Vector(sol(t)[1:3]) for t in ticks]...)
        pose_true = hcat([[sol(t)[17], sol(t)[18], sol(t)[4]] for t in ticks]...)
        v_hat = fill(bus.xhat[1], 3, N)  # constant placeholder
        pose_hat = repeat([bus.xhat[5], bus.xhat[6], bus.xhat[4]], 1, N)  # [X, Y, ψ]
        d_hat = fill(bus.d_hat[1], 3, N)
        slip = [slip_indicator(sol(t), params) for t in ticks]
        return EstimatorLog(ticks, v_true, v_hat, pose_true, pose_hat, d_hat, slip,
                            string(traj_entry.name), traj_entry.ref_type,
                            traj_entry.run_mode, seed)
    end

    ticks = [p.t for p in probe]
    N = length(ticks)
    v_true = hcat([Vector(sol(t)[1:3]) for t in ticks]...)
    pose_true = hcat([[sol(t)[17], sol(t)[18], sol(t)[4]] for t in ticks]...)
    v_hat = hcat([Vector(p.xhat[1:3]) for p in probe]...)
    # bus.xhat layout is [Vx, Vy, ψ̇, ψ, X, Y]; pose_hat expects [X, Y, ψ].
    pose_hat = hcat([[p.xhat[5], p.xhat[6], p.xhat[4]] for p in probe]...)
    d_hat = hcat([Vector(p.d_hat) for p in probe]...)
    slip = [slip_indicator(p.u, params) for p in probe]

    return EstimatorLog(ticks, v_true, v_hat, pose_true, pose_hat, d_hat, slip,
                        string(traj_entry.name), traj_entry.ref_type,
                        traj_entry.run_mode, seed)
end

# =============================================================================
# Replay-based estimator evaluation (no plant ODE solve)
# =============================================================================

function _interp_scalar(ts::AbstractVector{<:Real},
                        ys::AbstractVector{<:Real},
                        t::Real)
    n = length(ts)
    n == 0 && error("_interp_scalar: empty time vector")
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

function _finite_diff(y::AbstractVector{<:Real},
                      t::AbstractVector{<:Real})
    n = length(y)
    n == length(t) || error("_finite_diff: length mismatch")
    dy = similar(y, Float64)
    n == 1 && (dy[1] = 0.0; return dy)
    dy[1] = (y[2] - y[1]) / (t[2] - t[1])
    for i in 2:n-1
        dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
    end
    dy[n] = (y[n] - y[n-1]) / (t[n] - t[n-1])
    return dy
end

"""
    run_and_log_replay(est_cfg, traj_entry, nominal_ctrl_cfg; seed=42, data_dir)

Replay a pre-simulated Arrow trajectory through the candidate estimator without
running the plant/controller ODE.  The saved true state is interpolated to the
estimator tick rate, sensor measurements are regenerated with `SensorModel`, and
pose fixes are sampled from the true pose.  Returns an `EstimatorLog` compatible
with `estimator_objective_abs`.
"""
function run_and_log_replay(est_cfg::NamedTuple,
                            traj_entry::NamedTuple,
                            nominal_ctrl_cfg::Main.HybridConfigMod.HybridConfig;
                            seed::Int=42,
                            data_dir::AbstractString="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov",
                            sensor_kind::Symbol=:default,
                            pose_fix_kind::Symbol=:default)
    base = Main.Profiles.load_base(traj_entry.config_dir)
    chi = Float64(get(base, "physics", Dict())["chi"])
    params = Main.PlatformParams(base; mu_friction=Float64(traj_entry.mu))

    cfg = _build_cfg(est_cfg, traj_entry, nominal_ctrl_cfg, seed)
    est = _build_estimator(est_cfg)

    bus = Main.BusMod.ControllerBus()
    bus.use_dhat = cfg.use_dhat

    # Resolve the Arrow file via the canonical filename contract.
    meta = (profile       = String(traj_entry.name),
            combo_idx     = Int(traj_entry.combo_idx),
            mu            = Float64(traj_entry.mu),
            friction_case = 1,
            friction_model = :lugre_adamov,
            chi           = Float64(chi))
    arrow_path = Main.DataStore.expected_output(data_dir, meta)
    isfile(arrow_path) || error("run_and_log_replay: data not found for $(traj_entry.name) " *
                                "combo $(traj_entry.combo_idx) mu $(traj_entry.mu) chi $chi: $arrow_path")

    # Load trajectory data, with a process-wide cache so repeated evaluations of
    # the same trajectory do not re-read the compressed Arrow file.
    function _load_replay_data(arrow_path::String, data_dir::String)
        cached = lock(REPLAY_CACHE_LOCK) do
            get(REPLAY_CACHE, arrow_path, nothing)
        end
        cached !== nothing && return cached

        df = DataFrame(Arrow.Table(arrow_path))

        # Materialise the columns we need as plain Float64 vectors (the Arrow-backed
        # columns may be Float32 and/or allow Missing, which our interpolator rejects).
        function _col(v)
            return Float64.(collect(v))
        end
        ts        = _col(df.time)
        Vx_arr    = _col(df.Vx)
        Vy_arr    = _col(df.Vy)
        psidot_arr= _col(df.psi_dot)
        psi_arr   = _col(df.psi)
        theta_arrs= (_col(df.theta1), _col(df.theta2), _col(df.theta3), _col(df.theta4))
        omega_arrs= (_col(df.w1), _col(df.w2), _col(df.w3), _col(df.w4))
        Xo_arr    = _col(df.Xo)
        Yo_arr    = _col(df.Yo)

        # Body/wheel accelerations: exact-dynamics sidecar if available, else finite diff.
        stem, _ = splitext(basename(arrow_path))
        accel_path = joinpath(data_dir, "accel", "$(stem)_accel.arrow")
        if isfile(accel_path)
            adf = DataFrame(Arrow.Table(accel_path))
            # The acceleration sidecar has no time column; its rows align with df.time.
            a_times = ts
            dVx_arr = _col(adf.dVx)
            dVy_arr = _col(adf.dVy)
            dpsidot_arr = _col(adf.dpsidot)
        else
            a_times = ts
            dVx_arr = _finite_diff(Vx_arr, a_times)
            dVy_arr = _finite_diff(Vy_arr, a_times)
            dpsidot_arr = _finite_diff(psidot_arr, a_times)
        end

        data = (ts=ts, Vx_arr=Vx_arr, Vy_arr=Vy_arr, psidot_arr=psidot_arr,
                psi_arr=psi_arr, theta_arrs=theta_arrs, omega_arrs=omega_arrs,
                Xo_arr=Xo_arr, Yo_arr=Yo_arr, a_times=a_times, dVx_arr=dVx_arr,
                dVy_arr=dVy_arr, dpsidot_arr=dpsidot_arr)
        lock(REPLAY_CACHE_LOCK) do
            REPLAY_CACHE[arrow_path] = data
        end
        return data
    end

    data = _load_replay_data(arrow_path, data_dir)
    ts        = data.ts
    Vx_arr    = data.Vx_arr
    Vy_arr    = data.Vy_arr
    psidot_arr= data.psidot_arr
    psi_arr   = data.psi_arr
    theta_arrs= data.theta_arrs
    omega_arrs= data.omega_arrs
    Xo_arr    = data.Xo_arr
    Yo_arr    = data.Yo_arr
    a_times   = data.a_times
    dVx_arr   = data.dVx_arr
    dVy_arr   = data.dVy_arr
    dpsidot_arr = data.dpsidot_arr

    sm = Main.SensorMod.SensorModel(sensor_kind; seed=seed)

    fix = nothing
    if cfg.use_pose_fix
        tier = pose_fix_kind == :default ? cfg.pose_fix_tier : pose_fix_kind
        fix = Main.EstimatorMod.PoseFixModel(tier; seed=seed)
        Main.EstimatorMod.reset_pose_fix!(fix)
    end

    dt = 1.0 / cfg.f_est
    T = ts[end]
    ticks = collect(range(0.0, T; step=dt))
    !isempty(ticks) && ticks[end] > T + 1e-12 && pop!(ticks)
    N = length(ticks)

    v_true    = Matrix{Float64}(undef, 3, N)
    v_hat     = Matrix{Float64}(undef, 3, N)
    pose_true = Matrix{Float64}(undef, 3, N)
    pose_hat  = Matrix{Float64}(undef, 3, N)
    d_hat_log = Matrix{Float64}(undef, 3, N)
    slip      = Vector{Float64}(undef, N)

    # Scratch vectors long enough for the sensor/pose-fix indices used.
    u  = zeros(30)
    du = zeros(30)

    for (i, t) in enumerate(ticks)
        u[1]  = _interp_scalar(ts, Vx_arr, t)
        u[2]  = _interp_scalar(ts, Vy_arr, t)
        u[3]  = _interp_scalar(ts, psidot_arr, t)
        u[4]  = _interp_scalar(ts, psi_arr, t)
        for j in 1:4
            u[4+j] = _interp_scalar(ts, theta_arrs[j], t)
            u[8+j] = _interp_scalar(ts, omega_arrs[j], t)
        end
        u[17] = _interp_scalar(ts, Xo_arr, t)
        u[18] = _interp_scalar(ts, Yo_arr, t)

        du[1] = _interp_scalar(a_times, dVx_arr, t)
        du[2] = _interp_scalar(a_times, dVy_arr, t)
        du[3] = _interp_scalar(a_times, dpsidot_arr, t)

        y = Main.SensorMod.simulate_measurement(u, du, sm, t)
        Main.EstimatorMod.estimator_update!(bus, y, est, params, dt)

        if fix !== nothing
            z_fix = Main.EstimatorMod.sample_pose_fix(u, fix, t)
            if z_fix !== nothing
                Main.EstimatorMod.apply_pose_fix!(bus, est, fix, z_fix)
            end
        end

        v_true[:, i]   .= u[1], u[2], u[3]
        pose_true[:, i].= u[17], u[18], u[4]
        v_hat[:, i]    .= bus.xhat[1], bus.xhat[2], bus.xhat[3]
        pose_hat[:, i] .= bus.xhat[5], bus.xhat[6], bus.xhat[4]
        d_hat_log[:, i].= bus.d_hat
        slip[i] = slip_indicator(u, params)
    end

    return EstimatorLog(ticks, v_true, v_hat, pose_true, pose_hat, d_hat_log, slip,
                        string(traj_entry.name), traj_entry.ref_type,
                        traj_entry.run_mode, seed)
end

end # module
