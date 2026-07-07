# =============================================================================
# hybrid_ctrl/estimators.jl — Kalman + optional SMO velocity/heading estimator
# =============================================================================
module EstimatorMod

using StaticArrays
using LinearAlgebra

export KalmanEstimator, SMOEstimator, estimator_update!

"Estimated state layout: x̂ = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]"
const XHAT_DIM = 6

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

Base.@kwdef mutable struct KalmanEstimator
    Qn::SMatrix{3,3,Float64,9} = Diagonal(SVector(0.1, 0.1, 0.01))
    Rn::SMatrix{4,4,Float64,16} = Diagonal(SVector(0.01, 0.01, 0.01, 0.01))
    P::MMatrix{6,6,Float64,36} = MMatrix{6,6}(I)
    rate_hz::Float64 = 1000.0
    use_dhat::Bool   = false
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    initialized::Bool = false
end

function init_kalman!(est::KalmanEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.P .= Matrix(I(6)) .* 1e-3
    est.P[5,5] = 1.0
    est.P[6,6] = 1.0
    est.initialized = true
    return est
end

"""
    estimator_update!(bus, y, est::KalmanEstimator, params, dt)

Discrete EKF: measurement = encoder-derived body velocity pseudo-measurement
(least-squares inverse of wheel map) + gyro yaw rate.  Pose is dead-reckoned
(unobservable -> covariance grows).
"""
function estimator_update!(bus, y, est::KalmanEstimator, params, dt)
    !est.initialized && init_kalman!(est, params)
    Hω = est.wheel_H
    # Measurement: z = [V̂x_body, V̂y_body, ψ̂̇]
    z_vel = Hω \ y.ω          # pseudo-measurement from wheel speeds
    z     = SVector(z_vel[1], z_vel[2], y.g_z)

    # Observation matrix C (3 measured -> 6 states)
    C = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0]

    # Prediction (constant-velocity + yaw integrate)
    x = bus.xhat
    psi = x[4]
    cψ, sψ = cos(psi), sin(psi)
    xpred = SVector(x[1], x[2], x[3],
                    x[4] + dt * x[3],
                    x[5] + dt * (x[1]*cψ - x[2]*sψ),
                    x[6] + dt * (x[1]*sψ + x[2]*cψ))

    F = @SMatrix [1.0 0.0 0.0 0.0 0.0 0.0;
                  0.0 1.0 0.0 0.0 0.0 0.0;
                  0.0 0.0 1.0 0.0 0.0 0.0;
                  0.0 0.0 dt  1.0 0.0 0.0;
                  dt*cψ -dt*sψ 0.0 0.0 1.0 0.0;
                  dt*sψ  dt*cψ 0.0 0.0 0.0 1.0]

    Q = zeros(MMatrix{6,6,Float64,36})
    Q[1:3,1:3] .= est.Qn
    Q[4:6,4:6] .= Diagonal(SVector(1e-6, 1e-4, 1e-4))  # pose process noise

    Pp = F * est.P * F' + Q
    S  = C * Pp * C' + est.Rn
    K  = Pp * C' * inv(S)
    xnew = xpred + K * (z - C * xpred)
    est.P .= (I - K*C) * Pp
    bus.xhat = xnew
    bus.d_hat = SVector(0.0, 0.0, 0.0)
    return bus.xhat
end

# -----------------------------------------------------------------------------
# Sliding-mode observer estimator (optional)
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct SMOEstimator
    L::SVector{3,Float64} = SVector(20.0, 20.0, 15.0)
    K::SVector{3,Float64} = SVector(80.0, 80.0, 60.0)
    δ::Float64 = 0.05
    rate_hz::Float64 = 1000.0
    use_dhat::Bool = true
    xhat::MVector{6,Float64} = MVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d_hat::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)
    wheel_H::SMatrix{4,3,Float64,12} = zeros(SMatrix{4,3,Float64,12})
    initialized::Bool = false
end

function init_smo!(est::SMOEstimator, params)
    est.wheel_H = _wheel_jacobian(params)
    est.initialized = true
    return est
end

_smoothswitch(s, δ) = s / (abs(s) + δ)

function estimator_update!(bus, y, est::SMOEstimator, params, dt)
    !est.initialized && init_smo!(est, params)
    Hω = est.wheel_H
    z_vel = Hω \ y.ω
    z = SVector(z_vel[1], z_vel[2], y.g_z)

    x = SVector{6}(est.xhat)
    psi = x[4]
    cψ, sψ = cos(psi), sin(psi)

    # Velocity prediction with nominal model M_aug_inv * F ≈ 0 placeholder;
    # here use simple integrator model plus disturbance estimate.
    f_x = SVector(0.0, 0.0, 0.0)
    e = z - x[1:3]
    s_switch = _smoothswitch.(e, est.δ)

    dx_body = f_x .+ est.d_hat .+ est.L .* s_switch
    xnew = SVector(
        x[1] + dt * dx_body[1],
        x[2] + dt * dx_body[2],
        x[3] + dt * dx_body[3],
        x[4] + dt * x[3],
        x[5] + dt * (x[1]*cψ - x[2]*sψ),
        x[6] + dt * (x[1]*sψ + x[2]*cψ)
    )

    # Disturbance estimate (equivalent injection / integral term)
    dd = est.K .* s_switch
    est.d_hat .+= dt .* dd
    est.xhat .= xnew
    bus.xhat = xnew
    bus.d_hat = SVector(est.d_hat[1], est.d_hat[2], est.d_hat[3])
    return xnew
end

# Default dispatch guard (should never be called for unknown estimator types)
estimator_update!(bus, y, est, params, dt) =
    error("estimator_update!: unknown estimator type $(typeof(est))")

end # module
