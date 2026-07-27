# =============================================================================
# hybrid_ctrl/estimators.jl — Kalman + SMO velocity/heading/pose estimator
# =============================================================================
module EstimatorMod

using StaticArrays
using LinearAlgebra
using Random

export KalmanEstimator, SMOEstimator, IMMKalmanEstimator, ESKFEstimator, PoseFixModel,
       OracleEstimator, estimator_update!, slip_detect, sample_pose_fix,
       apply_pose_fix!, oracle_feed!

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

Noisy absolute-pose "sensor" for :pose runs (and velref runs with an explicit
fix tier).  Two tiers:
  :transit — intermittent, coarse (≈5–10 Hz, σ_pos≈0.05 m, σ_ψ≈2°)
  :docking — reliable, precise (100 Hz, σ_pos≈0.01 m, σ_ψ≈0.5°)
Ground truth is used only to synthesise the measurement; the estimator sees
only the noisy fix.
"""
Base.@kwdef mutable struct PoseFixModel
    use_pose_fix::Bool = false
    tier::Symbol       = :transit
    fix_rate_hz::Float64 = 10.0
    sigma_pos::Float64   = 0.05
    sigma_psi::Float64   = deg2rad(2.0)
    # Optional constant biases for robustness ablations (drawn once at reset).
    bias_px::Float64      = 0.0
    bias_py::Float64      = 0.0
    bias_psi::Float64     = 0.0
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
    rng = MersenneTwister(seed)
    if tier == :transit
        return PoseFixModel(
            use_pose_fix=true, tier=tier, fix_rate_hz=10.0,
            sigma_pos=0.05, sigma_psi=deg2rad(2.0),
            dropout_frac=0.2, outlier_frac=0.01,
            latency_ms=0.0, gate_thresh=1.0, seed=seed,
            rng=rng, last_fix_t=-Inf)
    elseif tier == :docking
        return PoseFixModel(
            use_pose_fix=true, tier=tier, fix_rate_hz=100.0,
            sigma_pos=0.01, sigma_psi=deg2rad(0.5),
            dropout_frac=0.0, outlier_frac=0.005,
            latency_ms=0.0, gate_thresh=0.5, seed=seed,
            rng=rng, last_fix_t=-Inf)
    elseif tier == :realistic
        # Indoor SLAM/AMCL-grade noise + constant biases for ablations.
        return PoseFixModel(
            use_pose_fix=true, tier=tier, fix_rate_hz=100.0,
            sigma_pos=0.02, sigma_psi=deg2rad(0.6),
            bias_px=0.01 * randn(rng), bias_py=0.01 * randn(rng),
            bias_psi=deg2rad(0.3) * randn(rng),
            dropout_frac=0.0, outlier_frac=0.005,
            latency_ms=0.0, gate_thresh=0.5, seed=seed,
            rng=rng, last_fix_t=-Inf)
    else
        error("PoseFixModel: unknown tier '$tier'; use :transit, :docking, or :realistic")
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
    nx = fix.sigma_pos * randn(fix.rng) + fix.bias_px
    ny = fix.sigma_pos * randn(fix.rng) + fix.bias_py
    npsi = fix.sigma_psi * randn(fix.rng) + fix.bias_psi

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
# IMMKalmanEstimator (2-model IMM EKF: grip / slip, 11-dim augmented state)
# =============================================================================

"IMM estimator state: x = [Vx, Vy, ψ̇, ψ, Xo, Yo, bx, by, sx, sy, bg] (sx, sy = wheel-slip velocity, bg = gyro bias; random walks)."
const IMM_DIM = 11

Base.@kwdef mutable struct IMMKalmanEstimator
    # Process / measurement noise
    Qn::SMatrix{3,3,Float64,9}      = Diagonal(SVector(1e-2, 1e-2, 1e-2))
    Rn_base::SMatrix{3,3,Float64,9} = Diagonal(SVector(1e-2, 1e-2, 1e-3))
    bias_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-4, 1e-4))
    slip_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-2, 1e-2))
    gyro_bias_Qn::Float64 = 1e-6    # gyro-bias random-walk intensity Q[11,11]
    pose_Qn::Float64      = 1e-6    # pose-state process noise Q[4:6,4:6]
    P0_scale::Float64 = 1e-2

    # Slip-adaptive weighting (kept for config compatibility; in the IMM the
    # slip model itself replaces the legacy binary R-inflation path)
    slip_threshold::Float64 = 0.1
    slip_R_inflate::Float64 = 10.0
    zupt_threshold::Float64 = 0.02

    # Grip-model slip-state random-walk scale (near-frozen slip in model 1)
    grip_slip_scale::Float64 = 1e-3

    # Adaptive process noise (A4) and NIS-triggered R boost (A5)
    alpha_acc::Float64  = 1.0
    alpha_yaw::Float64  = 0.5
    r_boost::Float64    = 10.0
    nis_thresh::Float64 = 9.21      # NIS gate for the R boost (χ²₃,₀.₉₉)

    # IMM mode-transition stay probabilities
    p_stay_grip::Float64 = 0.95
    p_stay_slip::Float64 = 0.9

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x1::MVector{11,Float64} = zeros(MVector{11,Float64})       # model 1: grip
    x2::MVector{11,Float64} = zeros(MVector{11,Float64})       # model 2: slip
    P1::MMatrix{11,11,Float64,121} = MMatrix{11,11}(I)
    P2::MMatrix{11,11,Float64,121} = MMatrix{11,11}(I)
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
    # Gyro-bias prior stays at the plain P0_scale (not the scaled slip prior).
    P0[11,11] = est.P0_scale
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
KalmanEstimator, extended to 11 states (biases, slip states and gyro bias
constant — random walks via Q).  Covariance propagated through the linearized
transition F = I + dt·A(x) with adaptive process noise (A4); `slip_q_scale`
sets the slip-state random-walk intensity (grip_slip_scale for the grip
model, 1.0 for the slip model).
"""
function _imm_predict(x::SVector{11,Float64}, P, est::IMMKalmanEstimator,
                      ax, ay, dt, slip_q_scale)
    v1, v2, v3 = x[1], x[2], x[3]
    psi = x[4]
    bx, by = x[7], x[8]
    cψ, sψ = cos(psi), sin(psi)

    x_pred = SVector{11}(
        v1 + dt * (ax + v3 * v2 - bx),
        v2 + dt * (ay - v3 * v1 - by),
        v3,                                  # yaw rate corrected by gyro measurement
        _wrap_angle(psi + dt * v3),
        x[5] + dt * (v1 * cψ - v2 * sψ),
        x[6] + dt * (v1 * sψ + v2 * cψ),
        bx, by, x[9], x[10], x[11])

    # Linearized transition F = I + dt·A(x)  (no plant-side coupling into bg)
    A = zeros(MMatrix{11,11,Float64,121})
    A[1,2] = v3;  A[1,3] = v2;  A[1,7] = -1.0
    A[2,1] = -v3; A[2,3] = -v1; A[2,8] = -1.0
    A[4,3] = 1.0
    A[5,1] = cψ;  A[5,2] = -sψ; A[5,4] = -v1 * sψ - v2 * cψ
    A[6,1] = sψ;  A[6,2] = cψ;  A[6,4] = v1 * cψ - v2 * sψ
    F = SMatrix{11,11}(I) + dt * A

    # Adaptive Q (A4): translational block scaled by accel/yaw activity
    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(v3)
    Q = zeros(MMatrix{11,11,Float64,121})
    Q[1,1] = est.Qn[1,1] * q_scale
    Q[2,2] = est.Qn[2,2] * q_scale
    Q[3,3] = est.Qn[3,3] * q_scale
    Q[4,4] = est.pose_Qn; Q[5,5] = est.pose_Qn; Q[6,6] = est.pose_Qn
    Q[7,7] = est.bias_Qn[1,1]
    Q[8,8] = est.bias_Qn[2,2]
    Q[9,9]  = est.slip_Qn[1,1] * slip_q_scale
    Q[10,10] = est.slip_Qn[2,2] * slip_q_scale
    Q[11,11] = est.gyro_bias_Qn

    Pp = F * P * F' + Q
    return x_pred, Pp
end

"""
    _imm_update(x_pred, Pp, est, z, slip_model) -> (x_new, P_new, ν, logdetS)

One-model measurement update of the IMM EKF.  z = [wheel pseudo-Vx, wheel
pseudo-Vy, gyro ψ̇]; wheels see v + s, gyro sees ψ̇ + bg.  The slip model
inflates wheel-R by slip_R_inflate and disables ACCEL-bias correction (rows
7:8, legacy contract); gyro-bias correction (row 11) stays ON in BOTH models —
in grip bg is observable from the wheel/gyro discrepancy, and in slip the gyro
is the trusted channel so its bias estimate must keep tracking.  Includes the
NIS-triggered adaptive-R boost (A5) and a Joseph-form covariance update.
Returns the NIS ν and logdet(S) for the IMM likelihood update.
"""
function _imm_update(x_pred::SVector{11,Float64}, Pp, est::IMMKalmanEstimator,
                     z::SVector{3,Float64}, slip_model::Bool)
    C = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

    R = MMatrix{3,3,Float64,9}(est.Rn_base)
    if slip_model
        R[1,1] *= est.slip_R_inflate
        R[2,2] *= est.slip_R_inflate
    end

    S = C * Pp * C' + R
    innov = z - C * x_pred
    Sinv = inv(S)
    ν = dot(innov, Sinv * innov)

    # Adaptive R (A5): one NIS-triggered inflation + recompute
    if ν > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = C * Pp * C' + R
        Sinv = inv(S)
        ν = dot(innov, Sinv * innov)
    end

    K = MMatrix{11,3,Float64,33}(Pp * C' * Sinv)   # mutable: slip model zeroes accel-bias rows
    if slip_model
        K[7, :] .= 0.0     # no accel-bias correction while slipping (legacy contract)
        K[8, :] .= 0.0
    end

    x_new = x_pred + K * innov
    I_KC = SMatrix{11,11}(I) - K * C
    P_new = I_KC * Pp * I_KC' + K * (R * K')   # Joseph form
    P_new = 0.5 * (P_new + P_new')             # PSD symmetrisation

    x_out = MVector{11,Float64}(x_new)
    x_out[4] = _wrap_angle(x_out[4])
    P_out = MMatrix{11,11,Float64,121}(P_new)
    return x_out, P_out, ν, logdet(S)
end

"IMM fusion: x̂ = Σ μ_j x_j; P = Σ μ_j (P_j + (x_j−x̂)(x_j−x̂)')."
function _imm_fuse(x1, P1, x2, P2, μ::SVector{2,Float64})
    xf = μ[1] * x1 + μ[2] * x2
    d1 = x1 - xf
    d2 = x2 - xf
    Pf = μ[1] .* (P1 .+ d1 * d1') .+ μ[2] .* (P2 .+ d2 * d2')
    return MVector{11,Float64}(xf), MMatrix{11,11,Float64,121}(Pf)
end

"""
    estimator_update!(bus, y, est::IMMKalmanEstimator, params, dt)

2-model IMM EKF tick over the shared 11-dim state
x = [Vx, Vy, ψ̇, ψ, Xo, Yo, bx, by, sx, sy, bg].  Model 1 "grip": nominal R,
near-frozen slip states (slip_Qn × grip_slip_scale), bias correction on.
Model 2 "slip": wheel-R × slip_R_inflate, free slip random walk, accel-bias
correction off.  Standard IMM cycle: mixing → per-model predict/update →
mode-probability update (log-domain likelihoods) → divergence guard → fusion.
bus.xhat ← fused x[1:6]; bus.d_hat ← fused [sx, sy, 0] (bg stays internal).
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

    xs = (SVector{11}(est.x1), SVector{11}(est.x2))
    Ps = (est.P1, est.P2)
    x1m = W[1,1] * xs[1] + W[2,1] * xs[2]
    x2m = W[1,2] * xs[1] + W[2,2] * xs[2]
    P1m = zeros(MMatrix{11,11,Float64,121})
    P2m = zeros(MMatrix{11,11,Float64,121})
    for i in 1:2
        d1 = xs[i] - x1m
        d2 = xs[i] - x2m
        P1m .+= W[i,1] .* (Ps[i] .+ d1 * d1')
        P2m .+= W[i,2] .* (Ps[i] .+ d2 * d2')
    end

    # --- 2+3. Per-model predict + measurement update ------------------------------
    x1p, P1p = _imm_predict(x1m, P1m, est, ax, ay, dt, est.grip_slip_scale)  # grip: near-frozen slip
    x2p, P2p = _imm_predict(x2m, P2m, est, ax, ay, dt, 1.0)                  # slip: free slip walk

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
F-propagation.  Gate is a covariance-aware NIS test on the FUSED state
(replaces the legacy fixed 3σ norm gate, which locked out all fixes once pose
drift exceeded the threshold in heavy slip): a filter with legitimately-large
covariance in slip keeps accepting fixes; a confident-but-wrong filter rejects
outliers.  Mode probabilities are unchanged; models are re-fused into bus.xhat.
"""
function apply_pose_fix!(bus, est::IMMKalmanEstimator, fix::PoseFixModel, z_fix::SVector{3})
    !est.initialized && return false

    μ = SVector(est.mu[1], est.mu[2])
    H_fix = @SMatrix [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;   # → Xo (state 5)
                      0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;   # → Yo (state 6)
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]   # → ψ  (state 4)
    R_fix = Diagonal(SVector(fix.sigma_pos^2, fix.sigma_pos^2, fix.sigma_psi^2))

    # Scale-aware NIS gate on the fused state: ν = e' S⁻¹ e, reject at χ²₃,₉₉.₉
    xf0, Pf0 = _imm_fuse(SVector{11}(est.x1), est.P1, SVector{11}(est.x2), est.P2, μ)
    e_fix = z_fix - SVector(xf0[5], xf0[6], xf0[4])
    e_fix = SVector(e_fix[1], e_fix[2], _wrap_angle(e_fix[3]))
    S_gate = H_fix * Pf0 * H_fix' + R_fix
    ν_fix = dot(e_fix, inv(S_gate) * e_fix)
    ν_fix > 14.16 && return false   # χ²₃ at 99.9%

    for (xj, Pj) in ((est.x1, est.P1), (est.x2, est.P2))
        S_fix = H_fix * Pj * H_fix' + R_fix
        K_fix = Pj * H_fix' * inv(S_fix)
        innov = z_fix - H_fix * SVector{11}(xj)
        innov = SVector(innov[1], innov[2], _wrap_angle(innov[3]))
        x_new = SVector{11}(xj) + K_fix * innov
        I_KH = SMatrix{11,11}(I) - K_fix * H_fix
        P_new = I_KH * Pj * I_KH' + K_fix * (R_fix * K_fix')   # Joseph form
        P_new = 0.5 * (P_new + P_new')
        xj .= x_new
        xj[4] = _wrap_angle(xj[4])
        Pj .= P_new
    end

    # Re-fuse and write bus (mode probabilities unchanged)
    xf, _Pf = _imm_fuse(SVector{11}(est.x1), est.P1, SVector{11}(est.x2), est.P2, μ)
    bus.xhat = SVector(xf[1], xf[2], xf[3], _wrap_angle(xf[4]), xf[5], xf[6])
    bus.d_hat = SVector(xf[9], xf[10], 0.0)
    return true
end

# =============================================================================
# OracleEstimator — inject the TRUE plant state (+ optional relative noise) as
# x̂, so controllers can be tuned on a fixed, well-characterised state feed
# while the real KF/SMO is still being fixed/tuned.  NOT a deployable estimator.
# =============================================================================

"""
    OracleEstimator

Synthetic "estimator" that feeds controllers the true plant state plus
configurable relative noise (a clean controlled variable).  Presets:
  :clean — zero noise (true labels)
  :noisy — 10% relative noise on position (Xo, Yo), 5% on IMU-derived states
           (Vx, Vy, ψ̇, ψ)
Per-channel σ = frac · max(|state|, floor), so noise never vanishes at zero.
Works for both :velocity (velocity + heading used) and :pose (position used).
"""
Base.@kwdef mutable struct OracleEstimator
    # Physically-consistent sensor error model, per channel:
    #   measured = true + bias + (σ0 + SF·|true|)·randn()
    #     σ0   = additive white noise (thermal / ARW / quantization) — constant
    #     SF   = scale-factor error (proportional) — small (gyro ~0.5%, odo ~2%)
    #     bias = constant per-run turn-on offset (drawn once at construction)
    sig_vel::Float64  = 0.0   # [m/s]   velocity white noise
    sf_vel::Float64   = 0.0   # [-]     velocity scale-factor
    bias_vx::Float64  = 0.0   # [m/s]   constant velocity bias (x)
    bias_vy::Float64  = 0.0   # [m/s]   constant velocity bias (y)
    sig_rate::Float64 = 0.0   # [rad/s] gyro white noise
    sf_rate::Float64  = 0.0   # [-]     gyro scale-factor
    bias_rate::Float64 = 0.0  # [rad/s] constant gyro bias
    sig_psi::Float64  = 0.0   # [rad]   heading white noise
    bias_psi::Float64 = 0.0   # [rad]   constant heading bias
    sig_pos::Float64  = 0.0   # [m]     position-fix white noise
    sf_pos::Float64   = 0.0   # [-]     position scale-factor (≈0 for absolute fix)
    bias_px::Float64  = 0.0   # [m]     constant position bias (x)
    bias_py::Float64  = 0.0   # [m]     constant position bias (y)
    seed::Int = 42
    rng::MersenneTwister = MersenneTwister(seed)
    rate_hz::Float64 = 1000.0
    use_dhat::Bool = false
end

function OracleEstimator(kind::Symbol; seed::Int=42, scale::Float64=1.0)
    rng = MersenneTwister(seed)
    if kind == :clean
        return OracleEstimator(seed=seed, rng=rng)   # all-zero noise (scale ignored)
    elseif kind == :noisy
        s = scale   # multiplies EVERY noise coefficient (white, scale-factor, bias)
        # Constant per-run turn-on biases drawn ONCE (held over the trajectory).
        bvx, bvy   = s*0.005*randn(rng), s*0.005*randn(rng)
        brate      = s*0.003*randn(rng)
        bpsi       = s*0.005*randn(rng)
        bpx, bpy   = s*0.01*randn(rng),  s*0.01*randn(rng)
        return OracleEstimator(
            sig_vel=s*0.010, sf_vel=s*0.02,  bias_vx=bvx, bias_vy=bvy,
            sig_rate=s*0.003, sf_rate=s*0.005, bias_rate=brate,
            sig_psi=s*0.010, bias_psi=bpsi,
            sig_pos=s*0.020, sf_pos=0.0,   bias_px=bpx, bias_py=bpy,
            seed=seed, rng=rng)
    else
        error("OracleEstimator: kind must be :clean or :noisy (got $kind)")
    end
end

"""
    oracle_feed!(bus, u, est::OracleEstimator, dt) -> SVector{6}

Write bus.xhat = true state (+ relative noise) from plant state `u`.
x̂ = [Vx, Vy, ψ̇, ψ, Xo, Yo] ← u-indices [1, 2, 3, 4, 17, 18].
"""
function oracle_feed!(bus, u, est::OracleEstimator, dt)
    Vx, Vy, psidot, psi = u[1], u[2], u[3], u[4]
    Xo, Yo = u[17], u[18]
    # measured = true + bias + (σ0 + SF·|true|)·randn()   (white ⊕ scale-factor ⊕ bias)
    Vx_h  = Vx     + est.bias_vx   + (est.sig_vel  + est.sf_vel  * abs(Vx))     * randn(est.rng)
    Vy_h  = Vy     + est.bias_vy   + (est.sig_vel  + est.sf_vel  * abs(Vy))     * randn(est.rng)
    pd_h  = psidot + est.bias_rate + (est.sig_rate + est.sf_rate * abs(psidot)) * randn(est.rng)
    psi_h = _wrap_angle(psi + est.bias_psi + est.sig_psi * randn(est.rng))
    Xo_h  = Xo + est.bias_px + (est.sig_pos + est.sf_pos * abs(Xo)) * randn(est.rng)
    Yo_h  = Yo + est.bias_py + (est.sig_pos + est.sf_pos * abs(Yo)) * randn(est.rng)
    bus.xhat  = SVector(Vx_h, Vy_h, pd_h, psi_h, Xo_h, Yo_h)
    bus.d_hat = SVector(0.0, 0.0, 0.0)
    return bus.xhat
end

# =============================================================================
# ESKFEstimator (single-model error-state Kalman filter, 12-dim nominal state,
# heading-vector / SO(2) direction-cosine formulation)
# =============================================================================
# Successor to IMMKalmanEstimator per chat-handoff/
# estimator_architecture_pivot_eskf_ukf_handoff.md (Step 1): same state
# content, but a SINGLE adaptive filter — no IMM wrapper.  The medicine is the
# slip-aware Q structure: the pose block of Q is inflated by measured slip
# activity so the covariance stays honest in slip and the NIS pose-fix gate
# keeps accepting fixes (fixes the IMM lockout-divergence bistability).
# Heading is carried as (c, s) = (cosψ, sinψ): no angle wrapping anywhere in
# the filter, and the heading/pose entries of F are EXACT (not linearized).
# atan2 appears only at the bus boundary.

"ESKF nominal/error state dimension (heading-vector formulation)."
const ESKF_DIM = 12

Base.@kwdef mutable struct ESKFEstimator
    # Process / measurement noise
    Qn::SMatrix{3,3,Float64,9}      = Diagonal(SVector(1e-2, 1e-2, 1e-2))
    Rn_base::SMatrix{3,3,Float64,9} = Diagonal(SVector(1e-2, 1e-2, 1e-3))
    bias_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-4, 1e-4))
    slip_Qn::SMatrix{2,2,Float64,4} = Diagonal(SVector(1e-2, 1e-2))
    gyro_bias_Qn::Float64 = 1e-6    # gyro-bias random-walk intensity Q[12,12]
    pose_Qn::Float64      = 1e-6    # base heading+pose process noise (slip-scaled)
    P0_scale::Float64 = 1e-2

    # Slip-adaptive weighting
    slip_threshold::Float64 = 0.1   # [m/s] grip/slip boundary
    slip_R_inflate::Float64 = 10.0  # wheel-R multiplier during slip
    zupt_threshold::Float64 = 0.02  # kept for config compatibility

    # Slip-aware Q structure
    grip_slip_scale::Float64 = 1e-3  # slip-state random-walk scale in grip
    pose_slip_gain::Float64  = 10.0  # pose-Q inflation per unit slip activity

    # Adaptive process noise (A4) and NIS-triggered R boost (A5)
    alpha_acc::Float64  = 1.0
    alpha_yaw::Float64  = 0.5
    r_boost::Float64    = 10.0
    nis_thresh::Float64 = 9.21      # NIS gate for the R boost (χ²₃,₀.₉₉)

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x::MVector{12,Float64} = MVector{12}(0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # nominal, (c,s)=(1,0)
    P::MMatrix{12,12,Float64,144} = MMatrix{12,12}(I)          # error-state covariance
    div_count::Int = 0   # consecutive-divergence counter (NIS > 1e4)
    initialized::Bool = false
end

"Renormalize the heading vector (c, s) ← (c, s)/hypot(c, s) in place."
function _renorm_heading!(x)
    n = hypot(x[4], x[5])
    x[4] /= n
    x[5] /= n
    return x
end

function init_eskf!(est::ESKFEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.x .= 0.0
    est.x[4] = 1.0        # (c, s) = (1, 0) ⇔ ψ = 0
    P0 = Matrix(I(ESKF_DIM)) .* est.P0_scale
    # Modest position prior so the first fix is not over-weighted (legacy
    # contract); heading-vector prior stays at the plain P0_scale.
    P0[6,6] = 1.0
    P0[7,7] = 1.0
    # Near-frozen slip prior (weak-observability lesson from the IMM): with
    # z_wheel = v + s an equal-variance init permanently leaks body velocity
    # into the slip states.
    P0[10,10] = est.P0_scale * 1e-3
    P0[11,11] = est.P0_scale * 1e-3
    # Gyro-bias prior stays at the plain P0_scale.
    P0[12,12] = est.P0_scale
    est.P .= P0
    est.div_count = 0
    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::ESKFEstimator, params, dt)

Single-model error-state KF tick over the nominal state
x = [Vx, Vy, ψ̇, c, s, X, Y, bx, by, sx, sy, bg] with (c, s) = (cosψ, sinψ).
The nominal state is propagated with the exact nonlinear kinematics (no angle
wrapping anywhere); the error covariance is propagated with F = I + dt·A
evaluated at the propagated nominal (the heading/pose entries of A are EXACT
in this formulation); the error state δx = K·e is injected immediately after
the update and the heading vector renormalized.
bus.xhat ← [Vx, Vy, ψ̇, atan2(s, c), X, Y]; bus.d_hat ← [sx, sy, 0].
"""
function estimator_update!(bus, y, est::ESKFEstimator, params, dt)
    !est.initialized && init_eskf!(est, params)

    ax, ay = y.a_x, y.a_y
    x = est.x

    # --- 1. Nominal propagation (exact nonlinear kinematics, no wrap) --------------
    v1, v2, v3 = x[1], x[2], x[3]
    cψ, sψ = x[4], x[5]
    bx, by = x[8], x[9]
    x[1] = v1 + dt * (ax + v3 * v2 - bx)
    x[2] = v2 + dt * (ay - v3 * v1 - by)
    # x[3] unchanged: yaw rate is corrected by the gyro measurement
    x[4] = cψ + dt * (-v3 * sψ)
    x[5] = sψ + dt * ( v3 * cψ)
    x[6] += dt * (v1 * cψ - v2 * sψ)
    x[7] += dt * (v1 * sψ + v2 * cψ)
    # bx, by, sx, sy, bg constant in prediction (random walks via Q)

    # --- 2. Error-covariance propagation ------------------------------------------
    z = _wheel_body_velocity(y, est.wheel_H)
    v_pred = SVector(x[1], x[2], x[3])
    slip_meas = slip_detect(y, v_pred, params)     # measurement-only indicator
    gripping = slip_meas < est.slip_threshold
    slip_activity = max(0.0, slip_meas - est.slip_threshold) / est.slip_threshold

    # F = I + dt·A, evaluated at the current (propagated) nominal.  The
    # heading/pose rows are EXACT in the heading-vector formulation (the (c,s)
    # and (X,Y) dynamics are linear in the state given v, ψ̇), not linearized
    # trigonometry as in the IMM's ψ formulation.
    w1, w2, w3 = x[1], x[2], x[3]
    cp, sp = x[4], x[5]
    A = zeros(MMatrix{12,12,Float64,144})
    A[1,2] = w3;  A[1,3] = w2;  A[1,8] = -1.0
    A[2,1] = -w3; A[2,3] = -w1; A[2,9] = -1.0
    A[4,3] = -sp; A[4,5] = -w3           # δċ = −ψ̇·δs − s·δψ̇
    A[5,3] =  cp; A[5,4] =  w3           # δṡ =  ψ̇·δc + c·δψ̇
    A[6,1] = cp;  A[6,2] = -sp; A[6,4] = w1; A[6,5] = -w2
    A[7,1] = sp;  A[7,2] =  cp; A[7,4] = w2; A[7,5] =  w1
    F = SMatrix{12,12}(I) + dt * A

    # Slip-aware Q structure (THE point of this estimator): slip is unobserved
    # pose error — the wheel/gyro measurements cannot see the pose drift that
    # slip induces — so P_pose must grow with slip activity to keep the
    # covariance honest and the NIS pose-fix gate open (fixes the IMM
    # lockout-divergence bistability documented in the pivot handoff).
    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(w3)
    pose_q = est.pose_Qn * (1.0 + est.pose_slip_gain * slip_activity)
    slip_q_scale = gripping ? est.grip_slip_scale : 1.0
    Q = zeros(MMatrix{12,12,Float64,144})
    Q[1,1] = est.Qn[1,1] * q_scale
    Q[2,2] = est.Qn[2,2] * q_scale
    Q[3,3] = est.Qn[3,3] * q_scale
    Q[4,4] = pose_q; Q[5,5] = pose_q; Q[6,6] = pose_q; Q[7,7] = pose_q
    Q[8,8] = est.bias_Qn[1,1]
    Q[9,9] = est.bias_Qn[2,2]
    Q[10,10] = est.slip_Qn[1,1] * slip_q_scale
    Q[11,11] = est.slip_Qn[2,2] * slip_q_scale
    Q[12,12] = est.gyro_bias_Qn
    est.P .= F * est.P * F' + Q

    # --- 3. Measurement update (error-state) ---------------------------------------
    # z = [wheel pseudo-Vx, wheel pseudo-Vy, gyro ψ̇]; wheels see v + s, gyro ψ̇ + bg
    H = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
    e = z - H * SVector{12}(x)

    R = MMatrix{3,3,Float64,9}(est.Rn_base)
    if !gripping
        R[1,1] *= est.slip_R_inflate
        R[2,2] *= est.slip_R_inflate
    end

    S = H * est.P * H' + R
    Sinv = inv(S)
    ν = dot(e, Sinv * e)

    # Adaptive R (A5): one NIS-triggered inflation + recompute
    if ν > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = H * est.P * H' + R
        Sinv = inv(S)
        ν = dot(e, Sinv * e)
    end

    K = MMatrix{12,3,Float64,36}(est.P * H' * Sinv)   # mutable: slip zeroes accel-bias rows
    if !gripping
        K[8, :] .= 0.0     # no accel-bias learning in slip (legacy contract)
        K[9, :] .= 0.0
    end
    # Row 12 (gyro bias) stays ON in both regimes: the gyro is the clean channel.

    δx = K * e
    I_KH = SMatrix{12,12}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (R * K')   # Joseph form
    est.P .= 0.5 * (Pn + Pn')                  # PSD symmetrisation

    # --- 4. Inject + reset -----------------------------------------------------------
    x .+= δx
    _renorm_heading!(x)
    # First-order error-state reset and renormalisation pass-through: P
    # unchanged.  The exact reset Jacobian G = I − ∂g/∂δx ≈ I for the additive
    # reset, and the (c,s) renormalisation Jacobian is likewise ≈ I to first
    # order — standard ESKF / heading-vector practice.

    # --- 5. Divergence guard ----------------------------------------------------------
    ν > 1e4 ? (est.div_count += 1) : (est.div_count = 0)
    if est.div_count >= 10
        # Single model: no fused partner to reset toward — inflate P and let
        # the measurements re-converge the filter.
        est.P .*= 10.0
        est.div_count = 0
    end

    # --- 6. Bus -----------------------------------------------------------------------
    # atan2 only at the bus boundary; the filter itself never wraps.
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return bus.xhat
end

"""
    apply_pose_fix!(bus, est::ESKFEstimator, fix::PoseFixModel, z_fix) -> Bool

Coupled exteroceptive pose fix on the nominal state, heading-vector form.
z_fix = (x, y, ψ) is converted to the 4-D LINEAR measurement
z4 = [x, y, cosψ, sinψ] with H_fix (4×12) selecting states (X, Y, c, s) =
(6, 7, 4, 5); R_fix = diag(σ_pos², σ_pos², σ_ψ², σ_ψ²) — the σ_ψ² on the
(c, s) channels is the small-angle tangent approximation (comment: heading
noise mapped onto the unit-circle tangent).  NO angle wrapping anywhere —
that is the point of the formulation.  NO Kalman-gain rows are zeroed:
velocity/bias/slip are corrected through the cross-covariance.  Gate is the
covariance-aware NIS test ν = e' S⁻¹ e, rejected at χ²₄,₉₉.₉ ≈ 16.27 (4-D
measurement now): a large-but-honest P (grown by the slip-aware Q) keeps
accepting fixes; a small-but-wrong P rejects outliers.  Heading vector is
renormalized after injection.
"""
function apply_pose_fix!(bus, est::ESKFEstimator, fix::PoseFixModel, z_fix::SVector{3})
    !est.initialized && return false

    H_fix = @SMatrix [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;   # → X (state 6)
                      0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;   # → Y (state 7)
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;   # → c (state 4)
                      0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]   # → s (state 5)
    # Small-angle tangent approximation: heading noise σ_ψ mapped onto the
    # unit-circle tangent, so the (c, s) channels carry the same variance.
    R_fix = Diagonal(SVector(fix.sigma_pos^2, fix.sigma_pos^2,
                             fix.sigma_psi^2, fix.sigma_psi^2))

    x = est.x
    z4 = SVector(z_fix[1], z_fix[2], cos(z_fix[3]), sin(z_fix[3]))
    e = z4 - H_fix * SVector{12}(x)      # no wrapping anywhere
    S_fix = H_fix * est.P * H_fix' + R_fix
    ν = dot(e, inv(S_fix) * e)
    ν > 16.27 && return false   # χ²₄ at 99.9%

    K_fix = est.P * H_fix' * inv(S_fix)
    x .+= K_fix * e
    _renorm_heading!(x)
    I_KH = SMatrix{12,12}(I) - K_fix * H_fix
    Pn = I_KH * est.P * I_KH' + K_fix * (R_fix * K_fix')   # Joseph form
    est.P .= 0.5 * (Pn + Pn')

    # atan2 only at the bus boundary
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

# Default dispatch guard
estimator_update!(bus, y, est, params, dt) =
    error("estimator_update!: unknown estimator type $(typeof(est))")

end # module
