# =============================================================================
# hybrid_ctrl_v2/estimators_v2_iae.jl — EstimatorModV2IAE
# (instructions/estimator-v2-iae-adaptive.md)
# =============================================================================
# `hybrid_ctrl_v2/estimators_v2.jl` is NEVER edited. This module adds an
# adaptive-Q variant of `ESKFEstimatorV2` that scales the slip process-noise
# block online from the wheel+gyro innovation sequence.
#
# Must be `include`d after `hybrid_ctrl_v2/estimators_v2.jl` (needs
# `Main.EstimatorModV2.gauss_markov_q`, `derive_process_noise`, the v1 generic
# `Main.EstimatorMod.estimator_update!`/`apply_pose_fix!`, and
# `Main.EstimatorMod._wheel_jacobian`/`slip_detect`/`_wheel_body_velocity`).
#
# Extension pattern is identical to `estimators_v2.jl`: new struct
# `ESKFIAEEstimatorV2` with its own `Main.EstimatorMod.estimator_update!` and
# `Main.EstimatorMod.apply_pose_fix!` methods. The v2 estimator remains fully
# untouched and bit-for-bit reproducible (`use_iae=false` is the regression
# gate in `compare_iae_baseline.jl`).
# =============================================================================
module EstimatorModV2IAE

using StaticArrays
using LinearAlgebra

export ESKFIAEEstimatorV2, init_eskf_iae_v2!, apply_flow_iae!

const ESKF_DIM_IAE = Main.EstimatorModV2.ESKF_DIM

# Pinned STF fading bound (brief §2 — λ_max is discipline, not tuned).
const LAMBDA_MAX_STF = 100.0

"""
    ESKFIAEEstimatorV2

Field-identical copy of `ESKFEstimatorV2` with the addition of an innovation-
based adaptive estimation (IAE) loop on the slip process-noise block only
(brief §2). When `use_iae=false` the tick is byte-for-byte identical to
`ESKFEstimatorV2`.

The slip-Q scale γ is driven by the running EMA of the wheel+gyro NIS:

    ν̄ ← (1 − λ)·ν̄ + λ·ν,    λ = dt/τ_iae
    γ ← clip( γ · (ν̄/3)^κ , γ_min , γ_max )

`E[ν] = 3` for the 3-D wheel+gyro update, so `ν̄/3` is the measured-to-
expected innovation power ratio. Adaptation is intentionally slow
(τ_iae ≫ τ_slip) so γ tracks slip regimes, not per-tick noise.
"""
Base.@kwdef mutable struct ESKFIAEEstimatorV2
    imu::Any    # SensorModV2.ImuModel
    enc::Any    # SensorModV2.EncoderModel
    flow::Any = nothing   # Union{SensorModV2.FlowModel,Nothing}

    # Physical specification for derive_process_noise (brief §7.3).
    # MEASURED (not assumed) from a closed-loop trace: spin_creep combo 178,
    # mu=0.5, ASMC default gains, 25s/25000 ticks. Ground-truth slip
    # s = Hw\omega_true - v_true (signed, per-axis), autocorrelation of sx/sy:
    # ACF decays cleanly (0.98 -> 0.13) over the first ~8ms, then goes
    # NEGATIVE and oscillates (a resonance coupled to spin_creep's own
    # periodic excitation, not intrinsic slip relaxation) -- this oscillatory
    # tail biases a naive multi-lag log-linear fit low (tau=0.006s, R^2=0.70).
    # The lag-1 AR(1) estimator tau=-dt/log(rho_1) is far less sensitive to
    # that tail (uses only the first step) and gives tau_sx=0.061s,
    # tau_sy=0.064s (rho_1~0.984) -- squarely inside the brief's own a-priori
    # order-of-magnitude expectation (0.05-0.2s) -- so THAT estimate is used.
    # sigma_slip = std(s) = 0.0132(sx)/0.0139(sy) m/s on this trajectory;
    # single-trajectory point estimate, likely trajectory-dependent -- worth
    # broadening to more traces in a follow-up (brief §11 leaves re-tuning
    # out of scope for THIS iteration).
    tau_slip::Float64             = 0.06     # s -- measured (lag-1 AR(1), see above)
    sigma_slip::Float64           = 0.014    # m/s -- measured stationary slip std (same measurement)
    sigma_gyro_bias_rw::Float64   = 1e-4    # rad/s/sqrt(s) -- MUST match imu.gyro_bias_rw (shared source)

    # Retained tunable — split by physically distinct quantity/unit (struct
    # docstring above). Defaults reproduce the old single-scalar draft's
    # numbers per-group so behaviour is unchanged until the optimizer moves
    # them apart.
    P0_vel::Float64        = 1e-2   # Vx,Vy initial variance [(m/s)^2]
    P0_yaw::Float64        = 1e-2   # psidot initial variance [(rad/s)^2]
    P0_heading::Float64    = 1e-2   # c,s initial variance [unitless, unit-circle]
    P0_bias_acc::Float64   = 1e-2   # bx,by initial variance [(m/s^2)^2]
    P0_bias_gyro::Float64  = 1e-2   # bg initial variance [(rad/s)^2]
    P0_slip::Float64       = 1e-5   # sx,sy initial variance [(m/s)^2] -- was P0_scale*1e-3
    P0_pos::Float64        = 1.0    # X,Y initial variance [m^2] -- was HARDCODED to 1.0, now a real tunable
    pose_Qn_heading::Float64 = 1e-6 # c,s per-tick process noise
    pose_Qn_pos::Float64     = 1e-6 # X,Y per-tick process noise

    # Kept-structure, no-longer-searched (brief §7.5 "re-examine"; defaults
    # unchanged from v1 so behaviour is comparable, just not free dims)
    slip_R_inflate::Float64  = 10.0
    grip_slip_scale::Float64 = 1e-3
    alpha_acc::Float64 = 1.0
    alpha_yaw::Float64 = 0.5
    r_boost::Float64   = 10.0

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # --- IAE adaptive slip-Q (brief §2) ---
    use_iae::Bool       = true     # ablation: false ⇒ identical to ESKFEstimatorV2
    iae_kind::Symbol    = :nis_ema # :nis_ema | :stf (:sage_husa is flag-only for now)
    tau_iae::Float64    = 0.5      # s — adaptation timescale (EMA), tunable
    kappa_iae::Float64  = 0.5      # adaptation exponent, tunable (mild response)
    gamma_min::Float64  = 1e-2     # PINNED discipline bounds
    gamma_max::Float64  = 1e4

    # --- IAE internal state ---
    gamma_q::Float64  = 1.0        # current slip-Q scale
    nu_bar::Float64   = 3.0        # EMA of wheel-update NIS (init at E[ν]=3: neutral)

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x::MVector{12,Float64} = MVector{12}(0.0,0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0)
    P::MMatrix{12,12,Float64,144} = MMatrix{12,12}(I)
    div_count::Int = 0
    initialized::Bool = false

    # PINNED (derived at init, not tuned)
    slip_threshold::Float64 = NaN     # set in init_eskf_iae_v2! from R_wheel (physical definition)
    nis_thresh::Float64       = 11.34 # chi^2_3,0.99 -- 3-D wheel+gyro measurement
    nis_thresh_flow::Float64  = 13.82 # chi^2_2,0.999 -- 2-D flow measurement (same 99.9% convention as pose-fix gate)
end

_wrap_angle_iae(psi) = atan(sin(psi), cos(psi))
_renorm_heading_iae!(x) = (n = hypot(x[4], x[5]); x[4] /= n; x[5] /= n; x)

function init_eskf_iae_v2!(est::ESKFIAEEstimatorV2, params)
    est.wheel_H = Main.EstimatorMod._wheel_jacobian(params)
    est.x .= 0.0
    est.x[4] = 1.0
    P0 = zeros(ESKF_DIM_IAE, ESKF_DIM_IAE)
    P0[1,1] = est.P0_vel;      P0[2,2] = est.P0_vel
    P0[3,3] = est.P0_yaw
    P0[4,4] = est.P0_heading;  P0[5,5] = est.P0_heading
    P0[6,6] = est.P0_pos;      P0[7,7] = est.P0_pos
    P0[8,8] = est.P0_bias_acc; P0[9,9] = est.P0_bias_acc
    P0[10,10] = est.P0_slip;   P0[11,11] = est.P0_slip
    P0[12,12] = est.P0_bias_gyro
    est.P .= P0
    est.div_count = 0

    # PINNED slip_threshold: 3-sigma of the wheel-pseudo-velocity noise floor
    # (R_wheel's diagonal) -- "the physical slip definition" (brief §7.5): a
    # slip signal is distinguishable from encoder noise once it exceeds this.
    Rw = Main.SensorModV2.R_wheel(est.enc, params)
    est.slip_threshold = 3.0 * sqrt(Rw[1,1])

    dt = 1.0 / est.rate_hz
    @assert est.tau_slip > dt/2 "ESKFIAEEstimatorV2: tau_slip=$(est.tau_slip) too small for dt=$dt -- " *
        "the discrete mean-reversion transition |1-dt/tau_slip| would exceed 1 (unstable)"

    # Reset IAE state every fresh init (e.g. per replay trajectory).
    est.gamma_q = 1.0
    est.nu_bar  = 3.0

    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::ESKFIAEEstimatorV2, params, dt)

Identical tick body to `ESKFEstimatorV2` except:
  1. The slip-block Q entries are scaled by the adaptive factor γ:
        Q[10,10] = Q[11,11] = pn.q_slip · γ · slip_q_scale
  2. After the wheel+gyro update the pre-r_boost NIS ν drives the γ EMA.

When `use_iae=false` (and `iae_kind=:nis_ema`) the method executes the exact
same arithmetic as `ESKFEstimatorV2` because γ is initialised to 1 and never
updated; this is the regression gate checked by `compare_iae_baseline.jl`.
"""
function Main.EstimatorMod.estimator_update!(bus, y, est::ESKFIAEEstimatorV2, params, dt)
    !est.initialized && init_eskf_iae_v2!(est, params)

    ax, ay = y.a_x, y.a_y
    x = est.x

    # --- 1. Nominal propagation (exact nonlinear kinematics, no wrap) -------
    v1, v2, v3 = x[1], x[2], x[3]
    cpsi, spsi = x[4], x[5]
    bx, by = x[8], x[9]
    x[1] = v1 + dt * (ax + v3*v2 - bx)
    x[2] = v2 + dt * (ay - v3*v1 - by)
    x[4] = cpsi + dt * (-v3*spsi)
    x[5] = spsi + dt * ( v3*cpsi)
    x[6] += dt * (v1*cpsi - v2*spsi)
    x[7] += dt * (v1*spsi + v2*cpsi)
    # bx,by,bg constant in prediction; sx,sy mean-revert via A below (not held constant)

    # --- 2. Error-covariance propagation ------------------------------------
    z = Main.EstimatorMod._wheel_body_velocity(y, est.wheel_H)
    v_pred = SVector(x[1], x[2], x[3])
    slip_meas = Main.EstimatorMod.slip_detect(y, v_pred, params)
    gripping = slip_meas < est.slip_threshold

    w1, w2, w3 = x[1], x[2], x[3]
    cp, sp = x[4], x[5]

    spec = (tau_slip=est.tau_slip, sigma_slip=est.sigma_slip, sigma_gyro_bias_rw=est.sigma_gyro_bias_rw)
    pn = Main.EstimatorModV2.derive_process_noise(spec, dt)

    A = zeros(MMatrix{12,12,Float64,144})
    A[1,2] = w3;  A[1,3] = w2;  A[1,8] = -1.0
    A[2,1] = -w3; A[2,3] = -w1; A[2,9] = -1.0
    A[4,3] = -sp; A[4,5] = -w3
    A[5,3] =  cp; A[5,4] =  w3
    A[6,1] = cp;  A[6,2] = -sp; A[6,4] = w1; A[6,5] = -w2
    A[7,1] = sp;  A[7,2] =  cp; A[7,4] = w2; A[7,5] =  w1
    A[10,10] = -pn.decay_slip     # mean-reversion: sx -> 0 as grip returns
    A[11,11] = -pn.decay_slip     # (THE model change -- brief §7.3)
    F = SMatrix{12,12}(I) + dt * A

    # Velocity/yaw-rate process noise DERIVED from accelerometer/gyro white
    # noise (brief: "Qn_diag removed -- derived from accelerometer noise"):
    # one Euler step v_{k+1}=v_k+dt*a_meas propagates a_meas's variance as
    # (dt*sigma)^2; psidot uses the gyro's noise floor as its sensor analog
    # (psidot itself isn't integrated in prediction -- x[3] is corrected only
    # by the measurement -- so this represents unmodelled yaw-accel between
    # ticks, bounded by the gyro channel).
    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(w3)
    q_vx = (dt * est.imu.sigma_acc)^2 * q_scale
    q_vy = q_vx
    q_yr = (dt * est.imu.sigma_gyro)^2 * q_scale
    slip_q_scale = gripping ? est.grip_slip_scale : 1.0

    Q = zeros(MMatrix{12,12,Float64,144})
    Q[1,1] = q_vx; Q[2,2] = q_vy; Q[3,3] = q_yr
    Q[4,4] = est.pose_Qn_heading; Q[5,5] = est.pose_Qn_heading
    Q[6,6] = est.pose_Qn_pos;     Q[7,7] = est.pose_Qn_pos
    Q[8,8] = pn.q_accel_bias
    Q[9,9] = pn.q_accel_bias
    # === IAE change (a): slip-block Q scaled by γ (brief §2) ================
    gamma_eff = est.use_iae && est.iae_kind == :nis_ema ? est.gamma_q : 1.0
    Q[10,10] = pn.q_slip * gamma_eff * slip_q_scale
    Q[11,11] = pn.q_slip * gamma_eff * slip_q_scale
    Q[12,12] = pn.q_gyro_bias
    est.P .= F * est.P * F' + Q

    # === Optional :stf variant: inflate P slip block directly (brief §2) ====
    if est.use_iae && est.iae_kind == :stf
        lambda_stf = clamp((est.nu_bar / 3.0)^est.kappa_iae, 1.0, LAMBDA_MAX_STF)
        est.P[10,10] *= lambda_stf
        est.P[10,11] *= lambda_stf
        est.P[11,10] *= lambda_stf
        est.P[11,11] *= lambda_stf
    end

    # --- 3. Measurement update (error-state), R DERIVED per tick -----------
    H = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
    e = z - H * SVector{12}(x)

    Rw = Main.SensorModV2.R_wheel(est.enc, params; omega_mag=norm(SVector(y.ω[1],y.ω[2],y.ω[3],y.ω[4])))
    r_gyro = Main.SensorModV2.R_gyro(est.imu, w3)
    R = MMatrix{3,3,Float64,9}(Diagonal(SVector(Rw[1,1], Rw[2,2], r_gyro)))
    if !gripping
        R[1,1] *= est.slip_R_inflate
        R[2,2] *= est.slip_R_inflate
    end

    S = H * est.P * H' + R
    Sinv = inv(S)
    nu = dot(e, Sinv * e)
    # === IAE change (b): adapt γ from the HONEST (pre-r_boost) NIS ==========
    nu_pre_boost = nu
    if nu > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = H * est.P * H' + R
        Sinv = inv(S)
        nu = dot(e, Sinv * e)
    end

    K = MMatrix{12,3,Float64,36}(est.P * H' * Sinv)
    if !gripping
        K[8, :] .= 0.0
        K[9, :] .= 0.0
    end

    dx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (R * K')
    est.P .= 0.5 * (Pn + Pn')

    x .+= dx
    _renorm_heading_iae!(x)

    nu > 1e4 ? (est.div_count += 1) : (est.div_count = 0)
    if est.div_count >= 10
        est.P .*= 10.0
        est.div_count = 0
    end

    if est.use_iae
        if est.iae_kind == :nis_ema
            lambda_ema = dt / est.tau_iae
            est.nu_bar = (1.0 - lambda_ema) * est.nu_bar + lambda_ema * nu_pre_boost
            est.gamma_q = clamp(est.gamma_q * (est.nu_bar / 3.0)^est.kappa_iae,
                                est.gamma_min, est.gamma_max)
        elseif est.iae_kind == :stf
            # γ is unused for :stf; λ is applied to P inside propagation above.
            # Keep ν̄ EMA as a diagnostic.
            lambda_ema = dt / est.tau_iae
            est.nu_bar = (1.0 - lambda_ema) * est.nu_bar + lambda_ema * nu_pre_boost
        else
            error("ESKFIAEEstimatorV2: iae_kind=$(est.iae_kind) is not implemented")
        end
    end

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return bus.xhat
end

"""
    apply_flow_iae!(bus, est::ESKFIAEEstimatorV2, m, z_flow, params) -> Bool

Identical body to `EstimatorModV2.apply_flow!` (`estimators_v2.jl:356-381`).
The flow channel is slip-immune (H selects Vx, Vy, yaw-rate lever arm; slip
columns 10-11 are zero), so it needs no Q adaptation. Kept as a dedicated
function so the IAE harness can call it without touching the v2 module.
"""
function apply_flow_iae!(bus, est::ESKFIAEEstimatorV2, m, z_flow::SVector{2}, params)
    !est.initialized && return false
    x = est.x
    r_x, r_y = m.r_offset[1], m.r_offset[2]
    H = @SMatrix [1.0 0.0 -r_y 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0  r_x 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    e = z_flow - H * SVector{12}(x)
    v_mag = norm(SVector(x[1], x[2]))
    Rf = Main.SensorModV2.R_flow(m, v_mag)
    S = H * est.P * H' + Rf
    Sinv = inv(S)
    nu = dot(e, Sinv * e)
    nu > est.nis_thresh_flow && return false

    K = est.P * H' * Sinv
    dx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (Rf * K')   # Joseph form
    est.P .= 0.5 * (Pn + Pn')                    # PSD symmetrization

    x .+= dx
    _renorm_heading_iae!(x)
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

"""
    apply_pose_fix!(bus, est::ESKFIAEEstimatorV2, fix, z_fix) -> Bool

Identical body to `EstimatorModV2.apply_pose_fix!` (`estimators_v2.jl:391-416`).
No reanchor coupling — IAE maintains no open-loop integration state.
"""
function Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFIAEEstimatorV2, fix, z_fix::SVector{3})
    !est.initialized && return false
    H_fix = @SMatrix [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    R_fix = Main.SensorModV2.R_posefix(fix)

    x = est.x
    z4 = SVector(z_fix[1], z_fix[2], cos(z_fix[3]), sin(z_fix[3]))
    e = z4 - H_fix * SVector{12}(x)
    S_fix = H_fix * est.P * H_fix' + R_fix
    nu = dot(e, inv(S_fix) * e)
    nu > 16.27 && return false

    K_fix = est.P * H_fix' * inv(S_fix)
    x .+= K_fix * e
    _renorm_heading_iae!(x)
    I_KH = SMatrix{12,12}(I) - K_fix * H_fix
    Pn = I_KH * est.P * I_KH' + K_fix * (R_fix * K_fix')
    est.P .= 0.5 * (Pn + Pn')

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

end # module
