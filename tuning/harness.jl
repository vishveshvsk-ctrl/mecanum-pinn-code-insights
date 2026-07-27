# =============================================================================
# tuning/harness.jl — closed-loop run + per-tick estimator log
# =============================================================================
module TuningHarnessMod

using StaticArrays
using LinearAlgebra

export EstimatorLog, slip_indicator, run_and_log

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
    # Enable exteroceptive pose fix for pose-tracking runs (docking tier by default)
    use_pose_fix = tracking == :pose
    pose_fix_tier = tracking == :pose ? :docking : :transit
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
    ref = Main.Profiles.pick_and_build(traj_entry.config_dir, [traj_entry.profile_toml])[1]
    if traj_entry.ref_type == :posref && traj_entry.run_mode == :velocity && ref isa Main.Profiles.PosRef
        ref = _posref_to_velref(ref)
    end

    # Run closed-loop simulation; request the bus so we can retrieve per-tick probes.
    sol, _df, bus = Main.SchedulerMod.run_hybrid(
        cfg, params, Symbol(traj_entry.name);
        chi=chi, friction_case=1, config_dir=traj_entry.config_dir,
        profile_toml=traj_entry.profile_toml, return_bus=true, est=est, ref=ref)

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
        pose_hat = fill(bus.xhat[4], 3, N)
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
    pose_hat = hcat([Vector(p.xhat[4:6]) for p in probe]...)
    d_hat = hcat([Vector(p.d_hat) for p in probe]...)
    slip = [slip_indicator(p.u, params) for p in probe]

    return EstimatorLog(ticks, v_true, v_hat, pose_true, pose_hat, d_hat, slip,
                        string(traj_entry.name), traj_entry.ref_type,
                        traj_entry.run_mode, seed)
end

end # module
