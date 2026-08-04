# =============================================================================
# hybrid_ctrl_v2/estimators_v2_slipobs.jl — EstimatorModV2SlipObs
# =============================================================================
# Slip-observer channel (:smo super-twisting / :eso linear ESO) feeding the
# ESKF as a 2-D pseudo-measurement of (sx, sy).  The observer itself is
# model-free; the ESKF keeps the Gauss–Markov slip prior and full covariance
# discipline (NIS gate + R_s).
#
# Must be `include`d AFTER hybrid_ctrl_v2/estimators_v2.jl (needs EstimatorModV2)
# and after tune_controller.jl / hybrid_ctrl/estimators.jl (extends
# Main.EstimatorMod.estimator_update! / apply_pose_fix!).
# =============================================================================
module EstimatorModV2SlipObs

using StaticArrays
using LinearAlgebra
using Random

export ESKFSlipObsEstimatorV2, init_eskf_slipobs_v2!, apply_flow_sobs!

const ESKF_DIM = 12

_wrap_angle(psi) = atan(sin(psi), cos(psi))
_renorm_heading!(x) = (n = hypot(x[4], x[5]); x[4] /= n; x[5] /= n; x)

"""
    ESKFSlipObsEstimatorV2

Identical state content, heading-vector formulation, and R/Q-derivation
contract as `EstimatorModV2.ESKFEstimatorV2`, with an additive slip-observer
channel (brief §2).  The observer produces a 2-D slip estimate `ŝ` that is
fed into the ESKF as a direct pseudo-measurement of the slip states
`(sx, sy)`; the Gauss–Markov slip prior is retained unchanged.
"""
Base.@kwdef mutable struct ESKFSlipObsEstimatorV2
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
    # single-trajectory point estimate, likely trajectory-dependent.
    tau_slip::Float64             = 0.06     # s -- measured (lag-1 AR(1), see above)
    sigma_slip::Float64           = 0.014    # m/s -- measured stationary slip std (same measurement)
    sigma_gyro_bias_rw::Float64   = 1e-4    # rad/s/sqrt(s) -- MUST match imu.gyro_bias_rw (shared source)

    # Retained tunable — split by physically distinct quantity/unit.
    P0_vel::Float64        = 1e-2
    P0_yaw::Float64        = 1e-2
    P0_heading::Float64    = 1e-2
    P0_bias_acc::Float64   = 1e-2
    P0_bias_gyro::Float64  = 1e-2
    P0_slip::Float64       = 1e-5
    P0_pos::Float64        = 1.0
    pose_Qn_heading::Float64 = 1e-6
    pose_Qn_pos::Float64     = 1e-6

    # Kept-structure, no-longer-searched (brief §7.5).
    slip_R_inflate::Float64  = 10.0
    grip_slip_scale::Float64 = 1e-3
    alpha_acc::Float64 = 1.0
    alpha_yaw::Float64 = 0.5
    r_boost::Float64   = 10.0

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # --- slip-observer config (brief §2.1) ---
    observer_kind::Symbol = :smo    # :smo | :eso  (channel identical; only the core differs)
    smo_k1::Float64     = 0.1       # super-twist injection gain  (:smo only)
    smo_k2::Float64     = 5.0       # super-twist integral gain, > max|ṡ|  (:smo only)
    smo_delta::Float64  = 1e-2      # boundary layer of Φ — PINNED, not searched
    eso_omega_o::Float64 = 30.0     # ESO bandwidth [rad/s]  (:eso only)
    rho_s::Float64      = 0.042     # slip pseudo-measurement std [m/s] — default 3*sigma_slip
    use_slipobs::Bool   = true      # ablation: false ⇒ identical to ESKFEstimatorV2

    # Internal ESKF state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x::MVector{12,Float64} = MVector{12}(0.0,0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0)
    P::MMatrix{12,12,Float64,144} = MMatrix{12,12}(I)
    div_count::Int = 0
    initialized::Bool = false

    # --- observer internal state ---
    obs_v_imu::MVector{2,Float64} = MVector{2}(0.0, 0.0)  # open-loop IMU velocity
    obs_s::MVector{2,Float64}     = MVector{2}(0.0, 0.0)  # ŝ
    obs_w::MVector{2,Float64}     = MVector{2}(0.0, 0.0)  # :smo integrator / :eso d̂

    # PINNED (derived at init, not tuned)
    slip_threshold::Float64 = NaN     # set in init_eskf_slipobs_v2! from R_wheel
    nis_thresh::Float64       = 11.34 # chi^2_3,0.99 -- 3-D wheel+gyro measurement
    nis_thresh_flow::Float64  = 13.82 # chi^2_2,0.999 -- 2-D flow measurement
    nis_thresh_slipobs::Float64 = 13.82 # chi^2_2,0.999 -- 2-D slip-observer measurement
end

"""
    _smooth_unit(e, δ)

Smooth unit switch used by the :smo super-twisting core (repo rule — no hard
switches in functions seen by stiff implicit solvers; here the observer is
discrete, but the same smooth boundary layer is reused for noise filtering).
"""
_smooth_unit(e::SVector{2,Float64}, δ) = e ./ sqrt.(e.^2 .+ δ^2)

function init_eskf_slipobs_v2!(est::ESKFSlipObsEstimatorV2, params)
    est.wheel_H = Main.EstimatorMod._wheel_jacobian(params)
    est.x .= 0.0
    est.x[4] = 1.0
    P0 = zeros(ESKF_DIM, ESKF_DIM)
    P0[1,1] = est.P0_vel;      P0[2,2] = est.P0_vel
    P0[3,3] = est.P0_yaw
    P0[4,4] = est.P0_heading;  P0[5,5] = est.P0_heading
    P0[6,6] = est.P0_pos;      P0[7,7] = est.P0_pos
    P0[8,8] = est.P0_bias_acc; P0[9,9] = est.P0_bias_acc
    P0[10,10] = est.P0_slip;   P0[11,11] = est.P0_slip
    P0[12,12] = est.P0_bias_gyro
    est.P .= P0
    est.div_count = 0

    # PINNED slip_threshold: 3-sigma of the wheel-pseudo-velocity noise floor.
    Rw = Main.SensorModV2.R_wheel(est.enc, params)
    est.slip_threshold = 3.0 * sqrt(Rw[1,1])

    dt = 1.0 / est.rate_hz
    @assert est.tau_slip > dt/2 "ESKFSlipObsEstimatorV2: tau_slip=$(est.tau_slip) too small for dt=$dt"

    # Zero the observer states at every (re-)initialisation.
    est.obs_v_imu .= 0.0
    est.obs_s     .= 0.0
    est.obs_w     .= 0.0

    est.initialized = true
    return est
end

"""
    slipobs_tick!(est, z_w, ax, ay, v3, dt) -> SVector{2,Float64}

One tick of the model-free slip observer (brief §2.1).  Keeps an open-loop
IMU velocity integration `obs_v_imu`, forms the measured discrepancy
`m = z_w - v̂_imu` (this contains slip plus near-DC bias-drift and noise),
and drives `ŝ` to track `m` with either a 2-D super-twisting (:smo) or a
2nd-order linear ESO (:eso) core.  Returns the updated `ŝ`.
"""
function slipobs_tick!(est::ESKFSlipObsEstimatorV2, z_w::SVector{2,Float64}, ax, ay, v3, dt)
    # b̂ from the ESKF's previous tick
    bx, by = est.x[8], est.x[9]
    a = SVector(ax, ay)
    b = SVector(bx, by)
    v = SVector(est.obs_v_imu[1], est.obs_v_imu[2])

    # Open-loop IMU integration with Coriolis correction
    #   v̂_imu⁺ = v̂_imu + dt·( a − b̂ + [ v3·v̂_imu_y ; −v3·v̂_imu_x ] )
    coriolis = SVector(v3 * v[2], -v3 * v[1])
    est.obs_v_imu .+= dt * (a - b + coriolis)

    m = z_w - SVector(est.obs_v_imu[1], est.obs_v_imu[2])
    e = m - SVector(est.obs_s[1], est.obs_s[2])

    if est.observer_kind == :smo
        Φ = _smooth_unit(e, est.smo_delta)
        # ŝ⁺ = ŝ + dt·( k1·Φ + w )
        est.obs_s .+= dt * (est.smo_k1 * Φ + est.obs_w)
        # w⁺ = w + dt·k2·Φ
        est.obs_w .+= dt * est.smo_k2 * Φ
    elseif est.observer_kind == :eso
        β1 = 2.0 * est.eso_omega_o
        β2 = est.eso_omega_o^2
        # ŝ⁺ = ŝ + dt·( d̂ + β1·e )
        est.obs_s .+= dt * (est.obs_w + β1 * e)
        # d̂⁺ = d̂ + dt·β2·e
        est.obs_w .+= dt * β2 * e
    else
        error("slipobs_tick!: unknown observer_kind=$(est.observer_kind)")
    end

    return SVector(est.obs_s[1], est.obs_s[2])
end

"""
    reanchor_imu_velocity!(est)

Re-anchor the open-loop IMU velocity to the ESKF's velocity estimate at
flow-update or pose-fix acceptance events, when slip-immune velocity
information is available.  This is the only back-coupling from the ESKF
fusion side into the observer.
"""
reanchor_imu_velocity!(est::ESKFSlipObsEstimatorV2) = (est.obs_v_imu .= (est.x[1], est.x[2]))

"""
    estimator_update!(bus, y, est::ESKFSlipObsEstimatorV2, params, dt)

ESKF tick identical to `ESKFEstimatorV2` when `use_slipobs=false`.  When
`use_slipobs=true`, two insertions are made (brief §2):
  1. `slipobs_tick!` after the wheel-pseudo-velocity `z` and `slip_meas`
     are computed;
  2. a 2-D slip pseudo-measurement update (H selects cols 10,11; R=ρ_s²I;
     NIS gate ν < 13.82; Joseph form) immediately before the wheel+gyro
     update, so the latter sees a slip-corrected state and covariance.
"""
function Main.EstimatorMod.estimator_update!(bus, y, est::ESKFSlipObsEstimatorV2, params, dt)
    !est.initialized && init_eskf_slipobs_v2!(est, params)

    ax, ay = y.a_x, y.a_y
    x = est.x

    # --- 1. Nominal propagation (exact nonlinear kinematics) -----------------
    v1, v2, v3 = x[1], x[2], x[3]
    cpsi, spsi = x[4], x[5]
    bx, by = x[8], x[9]
    x[1] = v1 + dt * (ax + v3*v2 - bx)
    x[2] = v2 + dt * (ay - v3*v1 - by)
    x[4] = cpsi + dt * (-v3*spsi)
    x[5] = spsi + dt * ( v3*cpsi)
    x[6] += dt * (v1*cpsi - v2*spsi)
    x[7] += dt * (v1*spsi + v2*cpsi)

    # --- 2. Error-covariance propagation ------------------------------------
    z = Main.EstimatorMod._wheel_body_velocity(y, est.wheel_H)
    v_pred = SVector(x[1], x[2], x[3])
    slip_meas = Main.EstimatorMod.slip_detect(y, v_pred, params)
    gripping = slip_meas < est.slip_threshold

    # --- insertion 1: slip-observer tick (brief §2) -------------------------
    obs_s = est.use_slipobs ? slipobs_tick!(est, SVector(z[1], z[2]), ax, ay, x[3], dt) : nothing

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
    A[10,10] = -pn.decay_slip
    A[11,11] = -pn.decay_slip
    F = SMatrix{12,12}(I) + dt * A

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
    Q[10,10] = pn.q_slip * slip_q_scale
    Q[11,11] = pn.q_slip * slip_q_scale
    Q[12,12] = pn.q_gyro_bias
    est.P .= F * est.P * F' + Q

    # --- insertion 2: slip pseudo-measurement update (brief §2.2) ----------
    # Placed AFTER covariance propagation and immediately BEFORE the wheel+gyro
    # update, so the wheel update sees an already slip-corrected state and P.
    if est.use_slipobs && obs_s !== nothing
        H_s = @SMatrix [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]
        z_s = obs_s
        e_s = z_s - H_s * SVector{12}(x)
        R_s = MMatrix{2,2,Float64,4}(Diagonal(SVector(est.rho_s^2, est.rho_s^2)))
        S_s = H_s * est.P * H_s' + R_s
        S_sinv = inv(S_s)
        ν_s = dot(e_s, S_sinv * e_s)
        if ν_s <= est.nis_thresh_slipobs
            K_s = est.P * H_s' * S_sinv
            dx_s = K_s * e_s
            I_KH = SMatrix{12,12}(I) - K_s * H_s
            Pn = I_KH * est.P * I_KH' + K_s * (R_s * K_s')
            est.P .= 0.5 * (Pn + Pn')
            x .+= dx_s
            _renorm_heading!(x)
        end
    end

    # --- 3. Wheel + gyro measurement update ---------------------------------
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
    ν = dot(e, Sinv * e)
    if ν > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = H * est.P * H' + R
        Sinv = inv(S)
        ν = dot(e, Sinv * e)
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
    _renorm_heading!(x)

    ν > 1e4 ? (est.div_count += 1) : (est.div_count = 0)
    if est.div_count >= 10
        est.P .*= 10.0
        est.div_count = 0
    end

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return bus.xhat
end

"""
    apply_flow_sobs!(bus, est::ESKFSlipObsEstimatorV2, m, z_flow, params) -> Bool

Identical body to `EstimatorModV2.apply_flow!`, plus re-anchoring of the
open-loop IMU velocity on acceptance.
"""
function apply_flow_sobs!(bus, est::ESKFSlipObsEstimatorV2, m, z_flow::SVector{2}, params)
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
    ν = dot(e, Sinv * e)
    ν > est.nis_thresh_flow && return false

    K = est.P * H' * Sinv
    dx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (Rf * K')
    est.P .= 0.5 * (Pn + Pn')

    x .+= dx
    _renorm_heading!(x)
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    reanchor_imu_velocity!(est)
    return true
end

"""
    apply_pose_fix!(bus, est::ESKFSlipObsEstimatorV2, fix, z_fix) -> Bool

Identical body to `EstimatorModV2.apply_pose_fix!`, plus re-anchoring of the
open-loop IMU velocity on acceptance.
"""
function Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFSlipObsEstimatorV2, fix, z_fix::SVector{3})
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
    ν = dot(e, inv(S_fix) * e)
    ν > 16.27 && return false

    K_fix = est.P * H_fix' * inv(S_fix)
    x .+= K_fix * e
    _renorm_heading!(x)
    I_KH = SMatrix{12,12}(I) - K_fix * H_fix
    Pn = I_KH * est.P * I_KH' + K_fix * (R_fix * K_fix')
    est.P .= 0.5 * (Pn + Pn')

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    reanchor_imu_velocity!(est)
    return true
end

end # module
