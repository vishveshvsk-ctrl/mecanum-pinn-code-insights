# =============================================================================
# hybrid_ctrl_v2/estimators_v3.jl — EstimatorModV3
# (yaw-acceleration state augmentation — structural fix for the yaw-rate
#  tracking lag; see chat-handoff/eskf_v2_mu0p5_train12_retune_handoff.md)
# =============================================================================
# `estimators_v2.jl` is NEVER edited — ESKFEstimatorV3 is a NEW struct with
# its own `estimator_update!`/`apply_flow!`/`apply_pose_fix!` methods added
# via multiple-dispatch extension of the ORIGINAL generic functions (same
# pattern as estimators_v2.jl). ESKFEstimatorV2 and every existing caller are
# completely untouched.
#
# Must be `include`d after `sensors_v2.jl`, `estimators_v2.jl` (reuses
# `EstimatorModV2.derive_process_noise`/`_renorm_heading!`), and
# `tune_controller.jl`/`hybrid_ctrl/estimators.jl` (extends
# `Main.EstimatorMod.estimator_update!`/`apply_pose_fix!`, reuses
# `Main.EstimatorMod._wheel_jacobian`/`slip_detect`).
#
# RATIONALE (k3 analysis, verified by the v3 tuning runs):
# The v2 yaw channel is a constant-rate model: x[3] (psidot) is never
# propagated, only corrected by measurement. A type-1 tracker on a ramp has
# steady-state lag e* = -alpha*dt/K = -alpha*tau (tau = 1/sqrt(q_scale) ~ 1 s
# at the derived q_yr), and the bias-variance product e_lag*sigma^2 = alpha*dt*R
# is INVARIANT to q -- the v3-wide run pinned alpha_acc at its 1e2 bound and
# still only reached score 6.63 with vel_rmse regressing: the q_scale trade is
# exhausted. The structural fix is a second integrator in the model:
#
#     psidot_{k+1} = psidot_k + dt*alpha_k
#     alpha_{k+1}  = alpha_k + w_alpha          (Singer-style random walk)
#
# A type-2 tracker follows a constant-acceleration ramp with ZERO steady-state
# error (fixed-point of the error recursion: e_p* = e_alpha* = 0 for any
# stabilizing gains); the residual drops from O(alpha) to O(jerk). alpha is
# observable through the psidot time series (observability matrix [H; HF] full
# rank) -- no new sensor, same pattern as the bias states.
#
# q_alpha is TUNABLE (per user direction; supersedes the derive-from-dpsidot
# option) via param_space_v4.jl. P0_alpha is PINNED at the physical prior
# 0.25 (+-0.5 rad/s^2 initial yaw-accel uncertainty, matching profile
# accelerations), per the "P0 = statement of knowledge" convention of
# param_space_v3.jl.
# =============================================================================
module EstimatorModV3

using StaticArrays
using LinearAlgebra
using Random

export ESKFEstimatorV3

"ESKF V3 nominal/error state dimension: v2's 12 + yaw acceleration (state 13)."
const ESKF_DIM_V3 = 13

"""
    ESKFEstimatorV3

`ESKFEstimatorV2` + a 13th state `alpha` (yaw acceleration, rad/s^2).
State: x = [Vx,Vy,psidot,c,s,X,Y,bx,by,sx,sy,bg, alpha].
Identical sensor/R/Q derivation to V2 except:
  - nominal propagation gains `x[3] += dt*x[13]` (psidot IS now integrated);
  - `A` gains `A[3,13] = 1.0` (alpha couples into psidot error dynamics);
  - `Q` gains `Q[13,13] = q_alpha` (per-tick random-walk variance, TUNABLE);
  - `P0` gains `P0[13,13] = P0_alpha` (PINNED physical prior, 0.25).
`q_yr` (the old gyro-noise-floor process noise on psidot) is RETAINED as a
small robustness margin for sub-tick jitter, but the maneuver-scale yaw
uncertainty now lives in the alpha state, not in q_scale*alpha_yaw -- so
alpha_yaw is expected to retune much lower (interior) here.
"""
Base.@kwdef mutable struct ESKFEstimatorV3
    imu::Any    # SensorModV2.ImuModel
    enc::Any    # SensorModV2.EncoderModel
    flow::Any = nothing   # Union{SensorModV2.FlowModel,Nothing}

    # Physical specification for derive_process_noise (measured constants --
    # see ESKFEstimatorV2's struct docstring for the measurement trail).
    tau_slip::Float64             = 0.06
    sigma_slip::Float64           = 0.014
    sigma_gyro_bias_rw::Float64   = 1e-4

    # Pinned physical priors (param_space_v3.jl convention) — defaults match
    # PINNED_V3; the v4 decode passes them explicitly.
    P0_vel::Float64        = 1e-4
    P0_yaw::Float64        = 1e-3
    P0_heading::Float64    = 0.5
    P0_bias_acc::Float64   = 4e-4
    P0_bias_gyro::Float64  = 9e-6
    P0_slip::Float64       = 2e-4
    P0_pos::Float64        = 0.25
    P0_alpha::Float64      = 0.25   # [(rad/s^2)^2] -- +-0.5 rad/s^2 initial yaw-accel uncertainty (profile-scale)
    pose_Qn_heading::Float64 = 1e-6
    pose_Qn_pos::Float64     = 1e-7

    # Update-rate / policy knobs (tunable via param_space_v4.jl)
    alpha_acc::Float64 = 1.0
    alpha_yaw::Float64 = 0.5
    grip_slip_scale::Float64 = 1e-3
    r_boost::Float64   = 10.0
    slip_R_inflate::Float64  = 10.0
    q_alpha::Float64   = 1e-4   # [(rad/s^2)^2 per tick] -- TUNABLE (1e-8..1e-1)

    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false

    # Internal state
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    x::MVector{13,Float64} = MVector{13}(0.0,0.0,0.0, 1.0,0.0, 0.0,0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.0)
    P::MMatrix{13,13,Float64,169} = MMatrix{13,13}(I)
    div_count::Int = 0
    initialized::Bool = false

    # PINNED (derived at init, not tuned)
    slip_threshold::Float64 = NaN     # set in init_eskf_v3! from R_wheel (physical definition)
    nis_thresh::Float64       = 11.34 # chi^2_3,0.99 -- 3-D wheel+gyro measurement
    nis_thresh_flow::Float64  = 13.82 # chi^2_2,0.999 -- 2-D flow measurement
end

function init_eskf_v3!(est::ESKFEstimatorV3, params)
    est.wheel_H = Main.EstimatorMod._wheel_jacobian(params)
    est.x .= 0.0
    est.x[4] = 1.0
    P0 = zeros(ESKF_DIM_V3, ESKF_DIM_V3)
    P0[1,1] = est.P0_vel;      P0[2,2] = est.P0_vel
    P0[3,3] = est.P0_yaw
    P0[4,4] = est.P0_heading;  P0[5,5] = est.P0_heading
    P0[6,6] = est.P0_pos;      P0[7,7] = est.P0_pos
    P0[8,8] = est.P0_bias_acc; P0[9,9] = est.P0_bias_acc
    P0[10,10] = est.P0_slip;   P0[11,11] = est.P0_slip
    P0[12,12] = est.P0_bias_gyro
    P0[13,13] = est.P0_alpha
    est.P .= P0
    est.div_count = 0

    Rw = Main.SensorModV2.R_wheel(est.enc, params)
    est.slip_threshold = 3.0 * sqrt(Rw[1,1])

    dt = 1.0 / est.rate_hz
    @assert est.tau_slip > dt/2 "ESKFEstimatorV3: tau_slip=$(est.tau_slip) too small for dt=$dt -- " *
        "the discrete mean-reversion transition |1-dt/tau_slip| would exceed 1 (unstable)"

    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::ESKFEstimatorV3, params, dt)

Single-model error-state KF tick — identical to `ESKFEstimatorV2`'s update
(R/Q derived per tick, slip mean-reversion via the corrected `gauss_markov_q`
continuous-time rate) except the yaw channel is now a double integrator:
`x[3] += dt*x[13]` in prediction and `A[3,13] = 1.0` in the error Jacobian.
The gyro/wheel/flow measurement rows are UNCHANGED (they observe psidot;
alpha is informed through the P[3,13] covariance built by F*P*F').
"""
function Main.EstimatorMod.estimator_update!(bus, y, est::ESKFEstimatorV3, params, dt)
    !est.initialized && init_eskf_v3!(est, params)

    ax, ay = y.a_x, y.a_y
    x = est.x

    # --- 1. Nominal propagation (exact nonlinear kinematics, no wrap) -------
    v1, v2, v3 = x[1], x[2], x[3]
    cpsi, spsi = x[4], x[5]
    bx, by = x[8], x[9]
    x[1] = v1 + dt * (ax + v3*v2 - bx)
    x[2] = v2 + dt * (ay - v3*v1 - by)
    x[3] = v3 + dt * x[13]          # THE model change: psidot integrated via alpha
    x[4] = cpsi + dt * (-v3*spsi)
    x[5] = spsi + dt * ( v3*cpsi)
    x[6] += dt * (v1*cpsi - v2*spsi)
    x[7] += dt * (v1*spsi + v2*cpsi)
    # OU mean-reversion of the slip STATE itself (matches the covariance-side
    # decay in A below; previously covariance-only -- state/covariance ran
    # different process models, see chat-handoff/eskf_v4_yawaccel_tuning_handoff.md)
    x[10] *= (1 - dt/est.tau_slip)
    x[11] *= (1 - dt/est.tau_slip)
    # bx,by,bg,alpha constant in prediction

    # --- 2. Error-covariance propagation ------------------------------------
    z = Main.EstimatorMod._wheel_body_velocity(y, est.wheel_H)
    v_pred = SVector(x[1], x[2], x[3])
    slip_meas = Main.EstimatorMod.slip_detect(y, v_pred, params)
    gripping = slip_meas < est.slip_threshold

    w1, w2, w3 = x[1], x[2], x[3]
    cp, sp = x[4], x[5]

    spec = (tau_slip=est.tau_slip, sigma_slip=est.sigma_slip, sigma_gyro_bias_rw=est.sigma_gyro_bias_rw)
    pn = Main.EstimatorModV2.derive_process_noise(spec, dt)

    A = zeros(MMatrix{13,13,Float64,169})
    A[1,2] = w3;  A[1,3] = w2;  A[1,8] = -1.0
    A[2,1] = -w3; A[2,3] = -w1; A[2,9] = -1.0
    A[3,13] = 1.0                 # alpha -> psidot coupling (THE model change)
    A[4,3] = -sp; A[4,5] = -w3
    A[5,3] =  cp; A[5,4] =  w3
    A[6,1] = cp;  A[6,2] = -sp; A[6,4] = w1; A[6,5] = -w2
    A[7,1] = sp;  A[7,2] =  cp; A[7,4] = w2; A[7,5] =  w1
    A[10,10] = -pn.decay_slip     # continuous-time rate 1/tau_slip (fixed)
    A[11,11] = -pn.decay_slip
    F = SMatrix{13,13}(I) + dt * A

    q_scale = 1.0 + est.alpha_acc * norm(SVector(ax, ay)) + est.alpha_yaw * abs(w3)
    q_vx = (dt * est.imu.sigma_acc)^2 * q_scale
    q_vy = q_vx
    q_yr = (dt * est.imu.sigma_gyro)^2 * q_scale   # sub-tick jitter margin only;
                                                   # maneuver-scale yaw uncertainty
                                                   # now lives in the alpha state
    slip_q_scale = gripping ? est.grip_slip_scale : 1.0

    Q = zeros(MMatrix{13,13,Float64,169})
    Q[1,1] = q_vx; Q[2,2] = q_vy; Q[3,3] = q_yr
    Q[4,4] = est.pose_Qn_heading; Q[5,5] = est.pose_Qn_heading
    Q[6,6] = est.pose_Qn_pos;     Q[7,7] = est.pose_Qn_pos
    Q[8,8] = pn.q_accel_bias
    Q[9,9] = pn.q_accel_bias
    Q[10,10] = pn.q_slip * slip_q_scale
    Q[11,11] = pn.q_slip * slip_q_scale
    Q[12,12] = pn.q_gyro_bias
    Q[13,13] = est.q_alpha        # TUNABLE (param_space_v4.jl)
    est.P .= F * est.P * F' + Q

    # --- 3. Measurement update (error-state), R DERIVED per tick -----------
    H = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]
    e = z - H * SVector{13}(x)

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
    if nu > est.nis_thresh
        R[1,1] *= est.r_boost
        R[2,2] *= est.r_boost
        S = H * est.P * H' + R
        Sinv = inv(S)
        nu = dot(e, Sinv * e)
    end

    K = MMatrix{13,3,Float64,39}(est.P * H' * Sinv)
    if !gripping
        K[8, :] .= 0.0
        K[9, :] .= 0.0
    end

    dx = K * e
    I_KH = SMatrix{13,13}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (R * K')
    est.P .= 0.5 * (Pn + Pn')

    x .+= dx
    Main.EstimatorModV2._renorm_heading!(x)

    nu > 1e4 ? (est.div_count += 1) : (est.div_count = 0)
    if est.div_count >= 10
        est.P .*= 10.0
        est.div_count = 0
    end

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return bus.xhat
end

"""
    apply_flow!(bus, est::ESKFEstimatorV3, m::FlowModel, z_flow, params) -> Bool

Optical-flow update — same as V2 but with a 2x13 Jacobian (alpha column ZERO:
flow observes body velocity and the lever-arm psidot coupling, never alpha
directly; alpha is informed through covariance). Same NIS gate (chi^2_2,0.999).
"""
function Main.EstimatorModV2.apply_flow!(bus, est::ESKFEstimatorV3, m, z_flow::SVector{2}, params)
    !est.initialized && return false
    x = est.x
    r_x, r_y = m.r_offset[1], m.r_offset[2]
    H = @SMatrix [1.0 0.0 -r_y 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0  r_x 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    e = z_flow - H * SVector{13}(x)
    v_mag = norm(SVector(x[1], x[2]))
    Rf = Main.SensorModV2.R_flow(m, v_mag)
    S = H * est.P * H' + Rf
    Sinv = inv(S)
    nu = dot(e, Sinv * e)
    nu > est.nis_thresh_flow && return false

    K = est.P * H' * Sinv
    dx = K * e
    I_KH = SMatrix{13,13}(I) - K * H
    Pn = I_KH * est.P * I_KH' + K * (Rf * K')   # Joseph form
    est.P .= 0.5 * (Pn + Pn')                    # PSD symmetrization

    x .+= dx
    Main.EstimatorModV2._renorm_heading!(x)
    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

"""
    apply_pose_fix!(bus, est::ESKFEstimatorV3, fix::PoseFixModel, z_fix) -> Bool

4-D (x,y,cospsi,sinpsi) pose-fix update with a 4x13 Jacobian (alpha column
zero). NIS gate at chi^2_4,0.999 = 16.27, Joseph form — same as V2.
"""
function Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFEstimatorV3, fix, z_fix::SVector{3})
    !est.initialized && return false
    H_fix = @SMatrix [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
                      0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    R_fix = Main.SensorModV2.R_posefix(fix)

    x = est.x
    z4 = SVector(z_fix[1], z_fix[2], cos(z_fix[3]), sin(z_fix[3]))
    e = z4 - H_fix * SVector{13}(x)
    S_fix = H_fix * est.P * H_fix' + R_fix
    nu = dot(e, inv(S_fix) * e)
    nu > 16.27 && return false

    K_fix = est.P * H_fix' * inv(S_fix)
    x .+= K_fix * e
    Main.EstimatorModV2._renorm_heading!(x)
    I_KH = SMatrix{13,13}(I) - K_fix * H_fix
    Pn = I_KH * est.P * I_KH' + K_fix * (R_fix * K_fix')
    est.P .= 0.5 * (Pn + Pn')

    bus.xhat = SVector(x[1], x[2], x[3], atan(x[5], x[4]), x[6], x[7])
    bus.d_hat = SVector(x[10], x[11], 0.0)
    return true
end

end # module
