# =============================================================================
# hybrid_ctrl/controllers.jl — ASMC / MPC / PID task-space wrench laws
# =============================================================================
module ControllerMod

using StaticArrays
using LinearAlgebra

export ASMCController, MPCController, PIDController,
       asmc_wrench!, mpc_wrench!, pid_wrench!

# -----------------------------------------------------------------------------
# ASMC controller (carried over; DOB removed, d_hat feedforward from SMO)
# -----------------------------------------------------------------------------
Base.@kwdef struct ASMCController
    gamma_x::Float64   = 8.0
    gamma_y::Float64   = 15.0
    gamma_psi::Float64 = 25.0
    eps::Float64       = 0.0175
    eps_psi::Float64   = 0.08
    K_max_x::Float64   = 60.0
    K_max_y::Float64   = 80.0
    K_max_psi::Float64 = 100.0
    lam_x_min::Float64 = 0.1
    lam_x_max::Float64 = 1.5
    lam_y_min::Float64 = 0.2
    lam_y_max::Float64 = 2.5
    lam_psi_min::Float64 = 0.5
    lam_psi_max::Float64 = 5.0
    mu_xy::Float64     = 25.0
    mu_psi::Float64    = 100.0
    sigma_x::Float64   = 0.5
    sigma_y::Float64   = 0.25
    sigma_psi::Float64 = 0.25
    decay_k::Float64   = 0.25
    K_x0::Float64      = 5.0
    K_y0::Float64      = 5.0
    K_psi0::Float64    = 20.0
    rate_hz::Float64   = 1000.0
end

_smooth_bound(K, K_max) = 0.5 - 0.5 * tanh(1.0 * (K - (K_max - 2.0)))

function _get_dynamic_lambda(e, edot, lam_min, lam_max, mu)
    exp_term = exp(-mu * e^2)
    lam = lam_min + (lam_max - lam_min) * exp_term
    lam_dot = -2 * mu * e * edot * (lam_max - lam_min) * exp_term
    return lam, lam_dot
end

"Yaw error with smooth wrap."
_e_psi(dψ) = 2 * tan(dψ/2) * (1 + 2 * (1 - cos(dψ)))
_edash_psi(dψ) = (sec(dψ/2))^2 * (3 - 2 * cos(dψ)) + 4 * tan(dψ/2) * sin(dψ)

"""
    asmc_wrench!(bus, xhat, ref, params, asmc, dt; mode=:velocity)

Adaptive sliding-mode task-space wrench [Wx, Wy, Wψ].  Keeps cubic + linear-
leakage gain update from run_one.jl:521–523.  `mode` is :pose or :velocity.
`bus.K` is integrated by forward-Euler at the ASMC rate.
"""
function asmc_wrench!(bus, xhat, ref, params, asmc::ASMCController, dt; mode::Symbol=:velocity)
    m, ms, Is, J1 = params.m, params.ms, params.Is, params.J_wheel
    aX, aY = params.aX, params.aY
    p1, p2 = params.p1_case1, params.p2_case1
    l, h, R = params.l, params.h, params.R

    Vx, Vy, psi_dot, psi = xhat[1], xhat[2], xhat[3], xhat[4]
    K_x, K_y, K_psi = bus.K[1], bus.K[2], bus.K[3]

    d_hat = bus.d_hat
    I_psi = Is + 4*(l + h)^2 / R^2 * J1

    if mode == :velocity
        Vx_d, Vy_d = ref.Vx(bus.t_now[]), ref.Vy(bus.t_now[])
        omega_d    = ref.Wz(bus.t_now[])
        Ax_d, Ay_d = ref.Ax(bus.t_now[]), ref.Ay(bus.t_now[])
        alpha_d    = ref.al(bus.t_now[])
        dψ = psi - ref.psi(bus.t_now[])

        s_x = Vx - Vx_d
        s_y = Vy - Vy_d
        eψ  = _e_psi(dψ)
        edψ = _edash_psi(dψ) * (psi_dot - omega_d)
        lam_psi, lam_dot_psi = _get_dynamic_lambda(eψ, edψ, asmc.lam_psi_min, asmc.lam_psi_max, asmc.mu_psi)
        s_psi = edψ + lam_psi * eψ

        Ax_eq, Ay_eq = Ax_d, Ay_d
        alpha_eq = alpha_d - lam_dot_psi * eψ - lam_psi * edψ * _edash_psi(dψ)
    else
        error("asmc_wrench! :pose mode not implemented in this notebook")
    end

    ss_x   = tanh(s_x   / asmc.eps)
    ss_y   = tanh(s_y   / asmc.eps)
    ss_psi = tanh(s_psi / asmc.eps_psi)

    Mx_sw   = -K_x   * ss_x
    My_sw   = -K_y   * ss_y
    Mpsi_sw = -K_psi * ss_psi

    Mx_eq = R * ((ms + 4*J1/R^2) * Ax_eq
                 - ms * psi_dot * Vy
                 - m  * aY * alpha_eq
                 - m  * aX * psi_dot^2
                 + 4  * p1 * Vx / R^2)
    My_eq = R * ((ms + 4*J1/R^2) * Ay_eq
                 + ms * psi_dot * Vx
                 + m  * aX * alpha_eq
                 - m  * aY * psi_dot^2
                 + (4 * p1 / R^2 + 8 * p2 / (R - params.Ra)^2) * Vy)
    M_psi_eq = (I_psi * alpha_eq
                - m * aY * (Ax_eq - psi_dot * Vy)
                + m * aX * (Ay_eq + psi_dot * Vx)
                + (4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-params.Ra)^2) * psi_dot)

    # SMO disturbance feedforward (was DOB compensation)
    if bus.use_dhat
        Mdh = params.M_aug * d_hat
        Mx_eq    -= R * Mdh[1]
        My_eq    -= R * Mdh[2]
        M_psi_eq -=     Mdh[3]
    end

    # Adaptive gain dynamics (forward-Euler)
    base_dK_x   = asmc.gamma_x   * (s_x   * ss_x)
    base_dK_y   = asmc.gamma_y   * (s_y   * ss_y)
    base_dK_psi = asmc.gamma_psi * (s_psi * ss_psi)
    dK_x   = base_dK_x   * _smooth_bound(K_x,   asmc.K_max_x)   - 0.1*(K_x/asmc.K_max_x)^3   - asmc.sigma_x  *(K_x  - asmc.K_x0  *0.95)*exp(asmc.decay_k*(1 - s_x^2  /(9*asmc.eps_psi^2)))
    dK_y   = base_dK_y   * _smooth_bound(K_y,   asmc.K_max_y)   - 0.3*(K_y/asmc.K_max_y)^3   - asmc.sigma_y  *(K_y  - asmc.K_y0  *0.95)*exp(asmc.decay_k*(1 - s_y^2  /(9*asmc.eps_psi^2)))
    dK_psi = base_dK_psi * _smooth_bound(K_psi, asmc.K_max_psi) - 0.5*(K_psi/asmc.K_max_psi)^3 - asmc.sigma_psi*(K_psi- asmc.K_psi0*0.95)*exp(asmc.decay_k*(1 - s_psi^2/(9*asmc.eps_psi^2)))

    bus.K = SVector(K_x + dt*dK_x, K_y + dt*dK_y, K_psi + dt*dK_psi)
    return SVector(Mx_sw + Mx_eq, My_sw + My_eq, Mpsi_sw + M_psi_eq)
end

# -----------------------------------------------------------------------------
# PID controller (Kalman-PID)
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct PIDController
    Kp::SVector{3,Float64} = SVector(20.0, 25.0, 15.0)
    Ki::SVector{3,Float64} = SVector(1.0, 1.0, 0.5)
    Kd::SVector{3,Float64} = SVector(2.0, 2.0, 1.0)
    I_max::SVector{3,Float64} = SVector(50.0, 50.0, 30.0)
    rate_hz::Float64 = 100.0
    prev_e::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)
    initialized::Bool = false
end

function pid_wrench!(bus, xhat, ref, pid::PIDController, dt)
    t = bus.t_now[]
    Vx_d, Vy_d = ref.Vx(t), ref.Vy(t)
    omega_d    = ref.Wz(t)

    e = SVector(xhat[1] - Vx_d, xhat[2] - Vy_d, xhat[3] - omega_d)
    de = pid.initialized ? (e - SVector(pid.prev_e...)) / dt : SVector(0.0, 0.0, 0.0)
    pid.prev_e .= e
    pid.initialized = true

    bus.I_pid = clamp.(bus.I_pid .+ dt .* e, .-pid.I_max, pid.I_max)
    W = .-pid.Kp .* e .- pid.Ki .* bus.I_pid .- pid.Kd .* de
    return W
end

# -----------------------------------------------------------------------------
# MPC controller — simplified QP over per-wheel voltages
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct MPCController
    Np::Int = 10
    Q::SVector{3,Float64} = SVector(10.0, 10.0, 5.0)
    R::SVector{4,Float64} = SVector(0.01, 0.01, 0.01, 0.01)
    S::SVector{4,Float64} = SVector(0.001, 0.001, 0.001, 0.001)
    V_max::Float64 = 24.0
    dV_max::Float64 = 10.0
    i_max::Float64 = 3.5
    rate_hz::Float64 = 100.0
    decision::Symbol = :wrench  # :wrench or :voltage
    last_W::SVector{3,Float64} = zeros(SVector{3})
    warm::Vector{Float64} = Float64[]
end

"Allocate a continuous voltage command to a task-space wrench prediction."
function _voltage_to_wrench_pred(V, ω, motor, params)
    # Predicted torque at the future wheel speed (assume ω unchanged over horizon)
    τ = Main.PlantMod.motor_torque(SVector{4}(V), ω, motor)
    lever = params.R / (params.l + params.h)
    Wx = τ[1] + τ[2] + τ[3] + τ[4]
    Wy = -τ[1] + τ[2] + τ[3] - τ[4]
    Wψ = (-τ[1] + τ[2] - τ[3] + τ[4]) / lever
    return SVector(Wx, Wy, Wψ)
end

function mpc_wrench!(bus, xhat, ref, params, mpc::MPCController)
    t = bus.t_now[]
    # Reference velocity target
    r = SVector(ref.Vx(t), ref.Vy(t), ref.Wz(t))
    e = SVector(xhat[1], xhat[2], xhat[3]) - r

    if mpc.decision == :wrench
        # Analytical LQR-like gain (no QP solve) for robustness & brevity
        K = SVector(mpc.Q[1]/100, mpc.Q[2]/100, mpc.Q[3]/100)
        W = .-K .* e
        # Saturate to plausible wrench envelope
        Wmax = params.Max_torque * SVector(4.0, 4.0, 4.0*(params.l+params.h)/params.R)
        W = clamp.(W, .-Wmax, Wmax)
        mpc.last_W = W
        return W
    else
        error("MPC voltage-decision mode not yet implemented; use :wrench")
    end
end

end # module
