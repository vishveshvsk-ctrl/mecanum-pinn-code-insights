# =============================================================================
# hybrid_ctrl/estimators.jl — Kalman + SMO velocity/heading/pose estimator
# =============================================================================
module EstimatorMod

using StaticArrays
using LinearAlgebra
using Random

export KalmanEstimator, SMOEstimator, IMMKalmanEstimator, PoseFixModel,
       estimator_update!, slip_detect, sample_pose_fix, apply_pose_fix!

"Estimated state layout: x̂ = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]"
const XHAT_DIM = 6

"Augmented estimator state: x̂_aug = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo, b̂x, b̂y]"
const AUG_DIM = 8

"Build the O-config wheel Jacobian H_ω mapping body twist -> wheel speeds."
function _wheel_jacobian(params)
    l, h, R = params.l, params.h, params.R
    # Mecanum O-config (see run_one.jl allocation):
    #   ω1 = (Vx - Vy - (l+h)ψ̇)/R
    #   ω2 = (Vx + Vy + (l+h)ψ̇)/R
    #   ω3 = (Vx + Vy - (l+h)ψ̇)/R
    #   ω4 = (Vx - Vy + (l+h)ψ̇)/R
    H = @SMatrix [ 1.0/R  -1.0/R  -(l+h)/R;
                   1.0/R   1.0/R   (l+h)/R;
                   1.0/R   1.0/R  -(l+h)/R;
                   1.0/R  -1.0/R   (l+h)/R ]
    return H
end

"Pseudo-measurement of body velocity from wheel speeds."
function _wheel_body_velocity(y, Hω)
    z_vel = Hω \ y.ω
    return SVector(z_vel[1], z_vel[2], y.g_z)
end

"Smooth saturation used for SMO switching boundary layer."
_smoothswitch(s, δ) = s / (sqrt(s^2) + δ)

"Smooth slip gate: 1 when slip is low, 0 when slip is high."
function _slip_gate(slip, thresh)
    thresh <= 0 && return 1.0
    # Smooth step around thresh; steepness ∝ 1/thresh
    return 0.5 - 0.5 * tanh((slip - thresh) / (0.2 * thresh))
end

"Wrap angle to [-π, π]."
_wrap_angle(ψ) = atan(sin(ψ), cos(ψ))

# =============================================================================
# Slip detector (measurement-only)
# =============================================================================

"""
    slip_detect(y, x_pred, params) -> Float64

Runtime slip indicator from MEASURED signals only: norm of the discrepancy
between the wheel-derived body velocity and the accel-predicted body velocity.
Drives KF R-inflation, SMO gate, and ZUPT.  Must NOT use ground truth.
"""
function slip_detect(y, x_pred::SVector{3}, params)
    Hω = _wheel_jacobian(params)
    v_wheel = Hω \ y.ω
    return norm(SVector(v_wheel[1], v_wheel[2], y.g_z) - x_pred)
end

# =============================================================================
# Exteroceptive pose fix (PosRef / docking / transit)
# =============================================================================

"""
    PoseFixModel

Noisy low-rate absolute-pose "sensor" for :pose runs.  Two tiers:
  :transit — intermittent, coarse (≈5–10 Hz, σ_pos≈0.05 m, σ_ψ≈2°)
  :docking — reliable, precise (≈20–30 Hz, σ_pos≈0.01 m, σ_ψ≈0.5°)
Ground truth is used only to synthesise the measurement; the estimator sees
only the noisy fix.
"""
Base.@kwdef mutable struct PoseFixModel
    use_pose_fix::Bool = false
    tier::Symbol       = :transit
    fix_rate_hz::Float64 = 10.0
    sigma_pos::Float64   = 0.05
    sigma_psi::Float64   = deg2rad(2.0)
    dropout_frac::Float64 = 0.2
    outlier_frac::Float64 = 0.01
    latency_ms::Float64   = 0.0
    gate_thresh::Float64  = 1.0
    seed::Int             = 42
    rng::MersenneTwister  = MersenneTwister(seed)
    last_fix_t::Float64   = -Inf
    # Controlled dropout window (seconds).  drop_start ≤ t ≤ drop_start+duration
    # returns no fix regardless of dropout_frac.  Defaults to an empty window.
    dropout_start::Float64    = Inf
    dropout_duration::Float64 = 0.0
end

function PoseFixModel(tier::Symbol; seed::Int=42)
    if tier == :transit
        return PoseFixModel(
            use_pose_fix=true, tier=tier, fix_rate_hz=10.0,
            sigma_pos=0.05, sigma_psi=deg2rad(2.0),
            dropout_frac=0.2, outlier_frac=0.01,
            latency_ms=0.0, gate_thresh=1.0, seed=seed,
            rng=MersenneTwister(seed), last_fix_t=-Inf)
    elseif tier == :docking
        return PoseFixModel(
            use_pose_fix=true, tier=tier, fix_rate_hz=30.0,
            sigma_pos=0.01, sigma_psi=deg2rad(0.5),
            dropout_frac=0.0, outlier_frac=0.005,
            latency_ms=0.0, gate_thresh=0.5, seed=seed,
            rng=MersenneTwister(seed), last_fix_t=-Inf)
    else
        error("PoseFixModel: unknown tier '$tier'; use :transit or :docking")
    end
end

"""
    sample_pose_fix(u, fix::PoseFixModel, t) -> Union{SVector{3},Nothing}

Exteroceptive absolute-pose "measurement" (x, y, ψ) for :pose runs: true pose +
Gaussian noise at fix_rate_hz, with dropout (→ Nothing this tick) and rare
outlier jumps.  Ground truth used only to GENERATE the fix.
"""
function sample_pose_fix(u, fix::PoseFixModel, t)
    fix.use_pose_fix || return nothing
    dt_fix = 1.0 / fix.fix_rate_hz
    # Sample on a regular grid anchored at t=0
    k = round(Int, t / dt_fix)
    t_fix = k * dt_fix
    # Avoid duplicate samples within the same tick
    if t_fix <= fix.last_fix_t
        return nothing
    end
    if abs(t - t_fix) > 0.5 * dt_fix
        return nothing
    end
    fix.last_fix_t = t_fix

    # Controlled dropout window: no fix for a defined interval.
    if t >= fix.dropout_start && t <= fix.dropout_start + fix.dropout_duration
        return nothing
    end

    rand(fix.rng) < fix.dropout_frac && return nothing

    x_true, y_true, psi_true = u[17], u[18], u[4]
    nx = fix.sigma_pos * randn(fix.rng)
    ny = fix.sigma_pos * randn(fix.rng)
    npsi = fix.sigma_psi * randn(fix.rng)

    # Rare outlier: large jump
    if rand(fix.rng) < fix.outlier_frac
        nx += 10.0 * nx
        ny += 10.0 * ny
        npsi += 10.0 * npsi
    end

    return SVector(x_true + nx, y_true + ny, _wrap_angle(psi_true + npsi))
end

function reset_pose_fix!(fix::PoseFixModel)
    fix.last_fix_t = -Inf
    Random.seed!(fix.rng, fix.seed)
    return fix
end

# =============================================================================
# KalmanEstimator (accel-fused EKF with slip-adaptive R and bias states)
# =============================================================================

Base.@kwdef mutable struct KalmanEstimator
    # Process / measurement noise
    Qn::SMatrix{3,3,Float64,9} = Diagonal(SVector(1e-2, 1e-2, 1e-2))
    Rn_base::SMatrix{3,3,Float64,9} = Diagonal(SVector(1e-2, 1e-2, 1e-3))
    bias_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-4, 1e-4))
    P0_scale::Float64 = 1e-2

    # Slip-adaptive weighting
    slip_threshold::Float64 = 0.1      # [m/s] slip inflation threshold
    slip_R_inflate::Float64 = 10.0     # wheel-R multiplier during slip
    zupt_threshold::Float64 = 0.02     # [m/s] strong-grip / zero-velocity threshold

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x_aug::MVector{8,Float64} = MVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    P::MMatrix{8,8,Float64,64} = MMatrix{8,8}(I)
    initialized::Bool = false
end

function init_kalman!(est::KalmanEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.x_aug .= 0.0
    est.P .= Matrix(I(8)) .* est.P0_scale
    # Give pose states a modest prior so the first fix is not over-weighted.
    est.P[4,4] = 1.0
    est.P[5,5] = 1.0
    est.P[6,6] = 1.0
    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::KalmanEstimator, params, dt)

Accel-fused EKF tick.  Predicts translational velocity from measured proper
acceleration minus the estimated accel-bias, updates with the 3-D measurement
(wheel pseudo-velocity + gyro), inflates wheel-R during slip, and only corrects
bias while gripping.  Dead-reckons pose using the updated velocity.
"""
function estimator_update!(bus, y, est::KalmanEstimator, params, dt)
    !est.initialized && init_kalman!(est, params)

    x = SVector{8}(est.x_aug)
    v = SVector(x[1], x[2], x[3])
    psi = x[4]
    bx, by = x[7], x[8]

    # --- Prediction (accel-driven) --------------------------------------------
    ax, ay, gz = y.a_x, y.a_y, y.g_z
    # Body-frame kinematic reconstruction: V̇x = ax + ψ̇·Vy − bx, etc.
    v_pred = SVector(
        v[1] + dt * (ax + v[3] * v[2] - bx),
        v[2] + dt * (ay - v[3] * v[1] - by),
        v[3])                       # yaw rate is corrected by gyro measurement

    psi_pred = psi + dt * v[3]
    cψ, sψ = cos(psi), sin(psi)
    Xo_pred = x[5] + dt * (v[1] * cψ - v[2] * sψ)
    Yo_pred = x[6] + dt * (v[1] * sψ + v[2] * cψ)

    x_pred = SVector(v_pred[1], v_pred[2], v_pred[3], psi_pred, Xo_pred, Yo_pred, bx, by)

    # Process noise for augmented state
    Q_aug = zeros(MMatrix{8,8,Float64,64})
    Q_aug[1:3,1:3] .= est.Qn
    Q_aug[4:6,4:6] .= Diagonal(SVector(1e-6, 1e-6, 1e-6))
    Q_aug[7:8,7:8] .= est.bias_Qn

    Pp = est.P + Q_aug

    # --- Measurement and slip-adaptive R --------------------------------------
    z = _wheel_body_velocity(y, est.wheel_H)
    slip = slip_detect(y, v_pred, params)
    gripping = slip < est.slip_threshold

    Rn = MMatrix{3,3,Float64,9}(est.Rn_base)
    if !gripping
        Rn[1,1] *= est.slip_R_inflate
        Rn[2,2] *= est.slip_R_inflate
    end

    # Measurement matrix: z = [Vx, Vy, psi_dot]
    C = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]

    S = C * Pp * C' + Rn
    K = Pp * C' * inv(S)

    # Bias correction only while gripping; also boost correction in strong grip (ZUPT)
    if slip >= est.zupt_threshold
        # Not a strong-grip instant: still allow bias correction if below slip threshold
        if !gripping
            K[7:8, :] .= 0.0
        end
    end

    innov = z - C * x_pred
    x_new = x_pred + K * innov
    P_new = (I - K * C) * Pp

    # Keep yaw angle wrapped
    x_new = SVector(x_new[1], x_new[2], x_new[3], _wrap_angle(x_new[4]),
                    x_new[5], x_new[6], x_new[7], x_new[8])

    est.x_aug .= x_new
    est.P .= P_new

    bus.xhat = SVector(x_new[1], x_new[2], x_new[3], x_new[4], x_new[5], x_new[6])
    bus.d_hat = SVector(0.0, 0.0, 0.0)
    return bus.xhat
end

function apply_pose_fix!(bus, est::KalmanEstimator, fix::PoseFixModel, z_fix::SVector{3})
    # Innovation gate on pose block
    e_fix = z_fix - SVector(est.x_aug[5], est.x_aug[6], est.x_aug[4])
    # Heading wrap for innovation
    e_fix = SVector(e_fix[1], e_fix[2], _wrap_angle(e_fix[3]))
    norm(e_fix) > 3.0 * fix.gate_thresh && return false

    # Standard KF update on pose block (states 4:6).  Zero the Kalman-gain rows
    # for velocity/bias so the exteroceptive fix corrects pose only, matching the
    # brief's "velocity states untouched" contract.
    H_fix = @SMatrix [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]
    R_fix = Diagonal(SVector(
        fix.sigma_pos^2, fix.sigma_pos^2, fix.sigma_psi^2))
    S_fix = H_fix * est.P * H_fix' + R_fix
    K_fix = est.P * H_fix' * inv(S_fix)
    K_fix[1:3, :] .= 0.0   # velocity rows
    K_fix[7:8, :] .= 0.0   # bias rows
    x_new = SVector{8}(est.x_aug) + K_fix * (z_fix - H_fix * SVector{8}(est.x_aug))
    x_new = SVector(x_new[1], x_new[2], x_new[3], _wrap_angle(x_new[4]),
                    x_new[5], x_new[6], x_new[7], x_new[8])
    est.x_aug .= x_new
    est.P .= (I - K_fix * H_fix) * est.P
    bus.xhat = SVector(x_new[1], x_new[2], x_new[3], x_new[4], x_new[5], x_new[6])
    return true
end

# =============================================================================
# SMOEstimator (accel-driven gated sliding-mode observer + ZUPT)
# =============================================================================

Base.@kwdef mutable struct SMOEstimator
    L::SVector{3,Float64} = SVector(15.0, 15.0, 15.0)   # sliding gain
    K::SVector{3,Float64} = SVector(5.0, 5.0, 5.0)      # integral (d-hat) gain
    δ::Float64            = 1e-2                         # boundary layer

    slip_gate_thresh::Float64 = 0.1   # slip magnitude where correction is gated
    zupt_threshold::Float64   = 0.02  # strong-grip re-anchor threshold
    bias_gain::SVector{2,Float64} = SVector(0.5, 0.5)

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = true

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x_aug::MVector{8,Float64} = MVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zeta::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)   # integral → bus.d_hat
    initialized::Bool = false
end

function init_smo!(est::SMOEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.x_aug .= 0.0
    est.zeta .= 0.0
    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::SMOEstimator, params, dt)

Accel-driven SMO tick.  Predicts from accel (minus bias); corrects toward the
wheel-velocity pseudo-measurement through a slip-gated smooth switch; the
integral term ζ accumulates the wheel↔accel disagreement and is exposed as
bus.d_hat.  ZUPT re-anchors velocity and bias when grip is detected.
"""
function estimator_update!(bus, y, est::SMOEstimator, params, dt)
    !est.initialized && init_smo!(est, params)

    x = SVector{8}(est.x_aug)
    v = SVector(x[1], x[2], x[3])
    psi = x[4]
    bx, by = x[7], x[8]

    ax, ay, gz = y.a_x, y.a_y, y.g_z

    # --- Prediction (accel-driven) --------------------------------------------
    v_pred = SVector(
        v[1] + dt * (ax + v[3] * v[2] - bx),
        v[2] + dt * (ay - v[3] * v[1] - by),
        v[3])

    psi_pred = psi + dt * v[3]
    cψ, sψ = cos(psi), sin(psi)
    Xo_pred = x[5] + dt * (v[1] * cψ - v[2] * sψ)
    Yo_pred = x[6] + dt * (v[1] * sψ + v[2] * cψ)

    # --- Slip-gated correction ------------------------------------------------
    z = _wheel_body_velocity(y, est.wheel_H)
    slip = slip_detect(y, v_pred, params)
    gate = _slip_gate(slip, est.slip_gate_thresh)
    # ZUPT: force full correction when strongly gripping
    if slip < est.zupt_threshold
        gate = 1.0
    end

    e = z - v_pred
    switch = _smoothswitch.(e, est.δ)

    v_new = v_pred + dt * est.L .* switch .* gate
    # Yaw rate is also pulled toward gyro via the same switch (e[3])
    # (v_new already includes this through v_pred[3] + correction)

    # Integral / disturbance estimate
    dzeta = est.K .* switch .* gate
    est.zeta .+= dt .* dzeta

    # Bias estimate: only integrates while gripping
    bx_new = bx + dt * est.bias_gain[1] * switch[1] * gate
    by_new = by + dt * est.bias_gain[2] * switch[2] * gate

    x_new = SVector(v_new[1], v_new[2], v_new[3], _wrap_angle(psi_pred),
                    Xo_pred, Yo_pred, bx_new, by_new)

    est.x_aug .= x_new

    bus.xhat = SVector(x_new[1], x_new[2], x_new[3], x_new[4], x_new[5], x_new[6])
    bus.d_hat = SVector(est.zeta[1], est.zeta[2], est.zeta[3])
    return bus.xhat
end

function apply_pose_fix!(bus, est::SMOEstimator, fix::PoseFixModel, z_fix::SVector{3})
    e_fix = z_fix - SVector(est.x_aug[5], est.x_aug[6], est.x_aug[4])
    e_fix = SVector(e_fix[1], e_fix[2], _wrap_angle(e_fix[3]))
    # Reject outliers using the configured gate threshold
    if norm(e_fix) > 3.0 * fix.gate_thresh
        return false
    end
    # Simple fixed-gain correction on pose block (no covariance in SMO)
    gain = 0.5
    x = est.x_aug
    x[4] = _wrap_angle(x[4] + gain * e_fix[3])
    x[5] += gain * e_fix[1]
    x[6] += gain * e_fix[2]
    bus.xhat = SVector(x[1], x[2], x[3], x[4], x[5], x[6])
    return true
end

# =============================================================================
# IMMKalmanEstimator (2-model IMM EKF: grip / slip, 10-dim augmented state)
# =============================================================================

"IMM estimator state: x = [Vx, Vy, ψ̇, ψ, Xo, Yo, bx, by, sx, sy] (sx, sy = wheel-slip velocity, random walk)."
const IMM_DIM = 10

Base.@kwdef mutable struct IMMKalmanEstimator
    # Process / measurement noise
    Qn::SMatrix{3,3,Float64,9}      = Diagonal(SVector(1e-2, 1e-2, 1e-2))
    Rn_base::SMatrix{3,3,Float64,9} = Diagonal(SVector(1e-2, 1e-2, 1e-3))
    bias_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-4, 1e-4))
    slip_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-2, 1e-2))
    P0_scale::Float64 = 1e-2

    # Slip-adaptive weighting (kept for config compatibility; in the IMM the
    # slip model itself replaces the legacy binary R-inflation path)
    slip_threshold::Float64 = 0.1
    slip_R_inflate::Float64 = 10.0
    zupt_threshold::Float64 = 0.02

    # Adaptive process noise (A4) and NIS-triggered R boost (A5)
    alpha_acc::Float64 = 1.0
    alpha_yaw::Float64 = 0.5
    r_boost::Float64   = 10.0

    # IMM mode-transition stay probabilities
    p_stay_grip::Float64 = 0.95
    p_stay_slip::Float64 = 0.9

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x1::MVector{10,Float64} = zeros(MVector{10,Float64})       # model 1: grip
    x2::MVector{10,Float64} = zeros(MVector{10,Float64})       # model 2: slip
    P1::MMatrix{10,10,Float64,100} = MMatrix{10,10}(I)
    P2::MMatrix{10,10,Float64,100} = MMatrix{10,10}(I)
    # Mode probabilities [grip, slip], initialised grip-favoured (0.9/0.1):
    # the nominal no-slip regime should dominate from rest.
    mu::MVector{2,Float64} = MVector(0.9, 0.1)
    div_count1::Int = 0   # consecutive-divergence counters (NIS > 1e4)
    div_count2::Int = 0
    initialized::Bool = false
end

function init_imm_kalman!(est::IMMKalmanEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.x1 .= 0.0
    est.x2 .= 0.0
    P0 = Matrix(I(IMM_DIM)) .* est.P0_scale
    # Modest pose prior so the first fix is not over-weighted (legacy contract).
    P0[4,4] = 1.0
    P0[5,5] = 1.0
    P0[6,6] = 1.0
    # Near-frozen slip prior: with z_wheel = v + s the v/s decomposition is only
    # weakly observable, so an equal-variance init would permanently leak a
    # fraction of body velocity into the slip states.  Start slip variance at
    # P0_scale × 1e-3 (mirrors the grip model's slip_Qn scaling); the slip
    # model's random walk re-inflates it within a tick when slip is real.
    P0[9,9] = est.P0_scale * 1e-3
    P0[10,10] = est.P0_scale * 1e-3
    est.P1 .= P0
    est.P2 .= P0
    est.mu .= (0.9, 0.1)
    est.div_count1 = 0
    est.div_count2 = 0
    est.initialized = true
    return est
end

"""
    _imm_predict(x, P, est, ax, ay, dt, slip_q_scale) -> (x_pred, Pp)

One-model prediction step of the IMM EKF.  Kinematics identical to the legacy
KalmanEstimator, extended to 10 states (biases and slip states constant).
Covariance propagated through the linearized transition F = I + dt·A(x) with
adaptive process noise (A4); `slip_q_scale` sets the slip-state random-walk
intensity (1e-3 for the grip model, 1.0 for the slip model).
"""
function _imm_predict(x::SVector{10,Float64}, P, est::IMMKalmanEstimator,
                      ax, ay, dt, slip_q_scale)
    v1, v2, v3 = x[1], x[2], x[3]
    psi = x[4]
    bx, by = x[7], x[8]
    cψ, sψ = cos(psi), sin(psi)

    x_pred = SVector{10}(
        v1 + dt * (ax + v3 * v2 - bx),
        v2 + dt * (ay - v3 * v1 - by),
        v3,                                  # yaw rate corrected by gyro measurement
        _wrap_angle(psi + dt * v3),
        x[5] + dt * (v1 * cψ - v2 * sψ),
        x[6] + dt * (v1 * sψ + v2 * cψ),
        bx, by, x[9], x[10])

    # Linearized transition F = I + dt·A(x)
    A = zeros(MMatrix{10,10,Float64,100})
    A[1,2] = v3;  A[1,3] = v2;  A[1,7] = -1.0
    A[2,1] = -v3; A[2,3] = -v1; A[2,8] = -1.0
    A[4,3] = 1.0
    A[5,1] = cψ;  A[5,2] = -sψ; A[5,4] = -v1 * sψ - v2 * cψ
    A[6,1] = sψ;  A[6,2] = cψ;  A[6,4] = v1 * cψ - v2 * sψ
    F = SMatrix{10,10}(I) + dt * A

    # Adaptive Q (A4): translational block scaled by accel/yaw activity
    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(v3)
    Q = zeros(MMatrix{10,10,Float64,100})
    Q[1,1] = est.Qn[1,1] * q_scale
    Q[2,2] = est.Qn[2,2] * q_scale
    Q[3,3] = est.Qn[3,3] * q_scale
    Q[4,4] = 1e-6; Q[5,5] = 1e-6; Q[6,6] = 1e-6
    Q[7,7] = est.bias_Qn[1,1]
    Q[8,8] = est.bias_Qn[2,2]
    Q[9,9]  = est.slip_Qn[1,1] * slip_q_scale
    Q[10,10] = est.slip_Qn[2,2] * slip_q_scale

    Pp = F * P * F' + Q
    return x_pred, Pp
end

"""
    _imm_update(x_pred, Pp, est, z, slip_model) -> (x_new, P_new, ν, logdetS)

One-model measurement update of the IMM EKF.  z = [wheel pseudo-Vx, wheel
pseudo-Vy, gyro ψ̇]; wheels see v + s, gyro is clean.  The slip model inflates
wheel-R by slip_R_inflate and disables bias correction (legacy contract).
Includes the NIS-triggered adaptive-R boost (A5) and a Joseph-form covariance
update.  Returns the NIS ν and logdet(S) for the IMM likelihood update.
"""
function _imm_update(x_pred::SVector{10,Float64}, Pp, est::IMMKalmanEstimator,
                     z::SVector{3,Float64}, slip_model::Bool)
    C = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

    R = MMatrix{3,3,Float64,9}(est.Rn_base)
    if slip_model
        R[1,1] *= est.slip_R_inflate
        R[2,2] *= est.slip_R_inflate
    end

    S = C * Pp * C' + R
    innov = z - C * x_pred
    Sinv = inv(S)
    ν = dot(innov, Sinv * innov)

    # Adaptive R (A5): one NIS-triggered inflation + recompute (χ²₃,₀.₉₉ ≈ 9.21)
    if ν > 9.21
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = C * Pp * C' + R
        Sinv = inv(S)
        ν = dot(innov, Sinv * innov)
    end

    K = MMatrix{10,3,Float64,30}(Pp * C' * Sinv)   # mutable: slip model zeroes bias rows
    if slip_model
        K[7, :] .= 0.0     # no bias correction while slipping (legacy contract)
        K[8, :] .= 0.0
    end

    x_new = x_pred + K * innov
    I_KC = SMatrix{10,10}(I) - K * C
    P_new = I_KC * Pp * I_KC' + K * (R * K')   # Joseph form
    P_new = 0.5 * (P_new + P_new')             # PSD symmetrisation

    x_out = MVector{10,Float64}(x_new)
    x_out[4] = _wrap_angle(x_out[4])
    P_out = MMatrix{10,10,Float64,100}(P_new)
    return x_out, P_out, ν, logdet(S)
end

"IMM fusion: x̂ = Σ μ_j x_j; P = Σ μ_j (P_j + (x_j−x̂)(x_j−x̂)')."
function _imm_fuse(x1, P1, x2, P2, μ::SVector{2,Float64})
    xf = μ[1] * x1 + μ[2] * x2
    d1 = x1 - xf
    d2 = x2 - xf
    Pf = μ[1] .* (P1 .+ d1 * d1') .+ μ[2] .* (P2 .+ d2 * d2')
    return MVector{10,Float64}(xf), MMatrix{10,10,Float64,100}(Pf)
end

"""
    estimator_update!(bus, y, est::IMMKalmanEstimator, params, dt)

2-model IMM EKF tick over the shared 10-dim state
x = [Vx, Vy, ψ̇, ψ, Xo, Yo, bx, by, sx, sy].  Model 1 "grip": nominal R,
near-frozen slip states (slip_Qn × 1e-3), bias correction on.  Model 2 "slip":
wheel-R × slip_R_inflate, free slip random walk, bias correction off.
Standard IMM cycle: mixing → per-model predict/update → mode-probability
update (log-domain likelihoods) → divergence guard → fusion.
bus.xhat ← fused x[1:6]; bus.d_hat ← fused [sx, sy, 0].
"""
function estimator_update!(bus, y, est::IMMKalmanEstimator, params, dt)
    !est.initialized && init_imm_kalman!(est, params)

    ax, ay = y.a_x, y.a_y
    z = _wheel_body_velocity(y, est.wheel_H)

    # --- 1. Mixing --------------------------------------------------------------
    # Π_ij = P(mode j | mode i); row i = from-model i
    Π = @SMatrix [est.p_stay_grip       1.0 - est.p_stay_grip;
                  1.0 - est.p_stay_slip est.p_stay_slip]
    μ = SVector(est.mu[1], est.mu[2])
    c̄ = Π' * μ
    c̄ = c̄ / sum(c̄)
    W = MMatrix{2,2,Float64,4}(undef)
    for i in 1:2, j in 1:2
        W[i, j] = Π[i, j] * μ[i] / c̄[j]
    end

    xs = (SVector{10}(est.x1), SVector{10}(est.x2))
    Ps = (est.P1, est.P2)
    x1m = W[1,1] * xs[1] + W[2,1] * xs[2]
    x2m = W[1,2] * xs[1] + W[2,2] * xs[2]
    P1m = zeros(MMatrix{10,10,Float64,100})
    P2m = zeros(MMatrix{10,10,Float64,100})
    for i in 1:2
        d1 = xs[i] - x1m
        d2 = xs[i] - x2m
        P1m .+= W[i,1] .* (Ps[i] .+ d1 * d1')
        P2m .+= W[i,2] .* (Ps[i] .+ d2 * d2')
    end

    # --- 2+3. Per-model predict + measurement update ------------------------------
    x1p, P1p = _imm_predict(x1m, P1m, est, ax, ay, dt, 1e-3)  # grip: near-frozen slip
    x2p, P2p = _imm_predict(x2m, P2m, est, ax, ay, dt, 1.0)   # slip: free slip walk

    x1n, P1n, ν1, ldS1 = _imm_update(x1p, P1p, est, z, false)
    x2n, P2n, ν2, ldS2 = _imm_update(x2p, P2p, est, z, true)

    # --- 4. Mode-probability update (log-domain, max-subtracted softmax) ----------
    logw1 = -0.5 * (ν1 + ldS1) + log(c̄[1])
    logw2 = -0.5 * (ν2 + ldS2) + log(c̄[2])
    lmax = max(logw1, logw2)
    e1, e2 = exp(logw1 - lmax), exp(logw2 - lmax)
    esum = e1 + e2
    est.mu[1] = e1 / esum
    est.mu[2] = e2 / esum
    μn = SVector(est.mu[1], est.mu[2])

    # --- 5. Divergence guard -------------------------------------------------------
    # Tentative fusion is the reset target for a diverged model.
    xf0, Pf0 = _imm_fuse(x1n, P1n, x2n, P2n, μn)
    ν1 > 1e4 ? (est.div_count1 += 1) : (est.div_count1 = 0)
    ν2 > 1e4 ? (est.div_count2 += 1) : (est.div_count2 = 0)
    if est.div_count1 >= 10
        x1n .= xf0
        P1n .= 10.0 .* Pf0
        est.div_count1 = 0
    end
    if est.div_count2 >= 10
        x2n .= xf0
        P2n .= 10.0 .* Pf0
        est.div_count2 = 0
    end

    # --- 6. Fusion → bus -----------------------------------------------------------
    xf, _Pf = _imm_fuse(x1n, P1n, x2n, P2n, μn)
    est.x1 .= x1n
    est.x2 .= x2n
    est.P1 .= P1n
    est.P2 .= P2n

    bus.xhat = SVector(xf[1], xf[2], xf[3], _wrap_angle(xf[4]), xf[5], xf[6])
    bus.d_hat = SVector(xf[9], xf[10], 0.0)
    return bus.xhat
end

"""
    apply_pose_fix!(bus, est::IMMKalmanEstimator, fix::PoseFixModel, z_fix) -> Bool

Coupled exteroceptive pose fix: standard KF update applied to EACH IMM model
on the pose block.  z_fix = (x, y, ψ); H_fix rows select states (Xo, Yo, ψ) =
(5, 6, 4).  Unlike the legacy KalmanEstimator, NO Kalman-gain rows are zeroed:
velocity/bias/slip are corrected through the cross-covariance built by true
F-propagation.  Legacy 3σ innovation gate retained; rejection returns false.
Mode probabilities are unchanged; models are re-fused into bus.xhat.
"""
function apply_pose_fix!(bus, est::IMMKalmanEstimator, fix::PoseFixModel, z_fix::SVector{3})
    !est.initialized && return false

    μ = SVector(est.mu[1], est.mu[2])
    # Innovation gate on the fused pose (legacy 3σ contract)
    xf0, _Pf0 = _imm_fuse(SVector{10}(est.x1), est.P1, SVector{10}(est.x2), est.P2, μ)
    e_fix = z_fix - SVector(xf0[5], xf0[6], xf0[4])
    e_fix = SVector(e_fix[1], e_fix[2], _wrap_angle(e_fix[3]))
    norm(e_fix) > 3.0 * fix.gate_thresh && return false

    H_fix = @SMatrix [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;   # → Xo (state 5)
                      0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;   # → Yo (state 6)
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0]   # → ψ  (state 4)
    R_fix = Diagonal(SVector(fix.sigma_pos^2, fix.sigma_pos^2, fix.sigma_psi^2))

    for (xj, Pj) in ((est.x1, est.P1), (est.x2, est.P2))
        S_fix = H_fix * Pj * H_fix' + R_fix
        K_fix = Pj * H_fix' * inv(S_fix)
        innov = z_fix - H_fix * SVector{10}(xj)
        innov = SVector(innov[1], innov[2], _wrap_angle(innov[3]))
        x_new = SVector{10}(xj) + K_fix * innov
        I_KH = SMatrix{10,10}(I) - K_fix * H_fix
        P_new = I_KH * Pj * I_KH' + K_fix * (R_fix * K_fix')   # Joseph form
        P_new = 0.5 * (P_new + P_new')
        xj .= x_new
        xj[4] = _wrap_angle(xj[4])
        Pj .= P_new
    end

    # Re-fuse and write bus (mode probabilities unchanged)
    xf, _Pf = _imm_fuse(SVector{10}(est.x1), est.P1, SVector{10}(est.x2), est.P2, μ)
    bus.xhat = SVector(xf[1], xf[2], xf[3], _wrap_angle(xf[4]), xf[5], xf[6])
    bus.d_hat = SVector(xf[9], xf[10], 0.0)
    return true
end

# Default dispatch guard
estimator_update!(bus, y, est, params, dt) =
    error("estimator_update!: unknown estimator type $(typeof(est))")

end # module
