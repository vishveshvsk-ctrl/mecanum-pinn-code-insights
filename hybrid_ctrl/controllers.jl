# =============================================================================
# hybrid_ctrl/controllers.jl — ASMC / MPC / PID task-space wrench laws
# =============================================================================
module ControllerMod

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP

export ASMCController, MPCController, PIDController,
       asmc_wrench!, mpc_wrench!, pid_wrench!, pose_outer_loop

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

    t = bus.t_now[]
    if mode == :velocity
        Vx_d, Vy_d = ref.Vx(t), ref.Vy(t)
        omega_d    = ref.Wz(t)
        Ax_d, Ay_d = ref.Ax(t), ref.Ay(t)
        alpha_d    = ref.al(t)
        dψ = psi - ref.psi(t)

        s_x = Vx - Vx_d
        s_y = Vy - Vy_d
        eψ  = _e_psi(dψ)
        edψ = _edash_psi(dψ) * (psi_dot - omega_d)
        lam_psi, lam_dot_psi = _get_dynamic_lambda(eψ, edψ, asmc.lam_psi_min, asmc.lam_psi_max, asmc.mu_psi)
        s_psi = edψ + lam_psi * eψ

        Ax_eq, Ay_eq = Ax_d, Ay_d
        alpha_eq = alpha_d - lam_dot_psi * eψ - lam_psi * edψ * _edash_psi(dψ)

    elseif mode == :pose
        # Position-tracking degree-2 sliding surface, ported from run_one.jl:asmc_torques.
        # Uses the ESTIMATED pose (X̂o,Ŷo,ψ̂); truth is not available to the controller.
        ref::Main.Profiles.PosRef
        e_xo = xhat[5] - ref.xo(t)
        e_yo = xhat[6] - ref.yo(t)
        cψ, sψ = cos(psi), sin(psi)
        e_x  =  e_xo * cψ + e_yo * sψ
        e_y  = -e_xo * sψ + e_yo * cψ
        dψ   = psi - ref.psi(t)
        eψ   = _e_psi(dψ)

        Vx_d, Vy_d, omega_d = Main.Profiles.global_to_local_frame(t, psi, ref.Vxo, ref.Vyo, ref.om)

        edot_x   = Vx - Vx_d + psi_dot * e_y
        edot_y   = Vy - Vy_d - psi_dot * e_x
        edot_psi = psi_dot - omega_d
        edashψ   = _edash_psi(dψ)

        lam_x,   lam_dot_x   = _get_dynamic_lambda(e_x,   edot_x,   asmc.lam_x_min,   asmc.lam_x_max,   asmc.mu_xy)
        lam_y,   lam_dot_y   = _get_dynamic_lambda(e_y,   edot_y,   asmc.lam_y_min,   asmc.lam_y_max,   asmc.mu_xy)
        lam_psi, lam_dot_psi = _get_dynamic_lambda(eψ,    edashψ * edot_psi,
                                                    asmc.lam_psi_min, asmc.lam_psi_max, asmc.mu_psi)

        s_x   = edot_x   + lam_x   * e_x
        s_y   = edot_y   + lam_y   * e_y
        s_psi = edot_psi + lam_psi * eψ

        Ax_des, Ay_des, alpha_des = Main.Profiles.global_to_local_frame(t, psi, ref.Axo, ref.Ayo, ref.al)
        alpha_eq = alpha_des - lam_dot_psi * eψ - lam_psi * edot_psi * edashψ
        Ax_eq    = Ax_des - lam_dot_x * e_x - lam_x * edot_x + psi_dot * (Vy_d - edot_y) - alpha_eq * e_y
        Ay_eq    = Ay_des - lam_dot_y * e_y - lam_y * edot_y - psi_dot * (Vx_d - edot_x) + alpha_eq * e_x

    else
        error("asmc_wrench! mode must be :velocity or :pose; got $mode")
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
    Kp_pos::SVector{3,Float64} = SVector(1.0, 1.0, 2.0)
    Kd_pos::SVector{3,Float64} = SVector(0.5, 0.5, 1.0)
    rate_hz::Float64 = 100.0
    prev_e::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)
    initialized::Bool = false
    prev_e_pos::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)  # outer-loop body pos error (for Kd_pos)
    pos_initialized::Bool = false
end

"""
    pose_outer_loop(xhat, ref, pid, t, dt) -> SVector{3}

Cascade outer loop: world position error (X̂o,Ŷo,ψ̂)−(ref.xo,ref.yo,ref.psi)
rotated to body frame; returns a body-velocity setpoint
V_cmd = ff(Vxo,Vyo,om)_body − Kp_pos·e_body (− Kd_pos·ė_body).
No integral in the outer loop to avoid stacked windup.
"""
function pose_outer_loop(xhat, ref::Main.Profiles.PosRef, pid::PIDController, t::Real, dt::Real)
    psi = xhat[4]
    cψ, sψ = cos(psi), sin(psi)
    e_xo = xhat[5] - ref.xo(t)
    e_yo = xhat[6] - ref.yo(t)
    e_x  =  e_xo * cψ + e_yo * sψ
    e_y  = -e_xo * sψ + e_yo * cψ
    e_psi = Main.EstimatorMod._wrap_angle(xhat[4] - ref.psi(t))
    e_body = SVector(e_x, e_y, e_psi)

    # Body-frame position-error derivative (PD outer loop). First tick has no
    # history → zero to avoid a startup derivative kick.
    de_body = pid.pos_initialized ?
        (e_body - SVector(pid.prev_e_pos...)) / dt : SVector(0.0, 0.0, 0.0)
    pid.prev_e_pos .= e_body
    pid.pos_initialized = true

    # Body-frame velocity feedforward
    Vx_ff, Vy_ff, om_ff = Main.Profiles.global_to_local_frame(t, psi, ref.Vxo, ref.Vyo, ref.om)
    V_cmd = SVector(Vx_ff, Vy_ff, om_ff) .- pid.Kp_pos .* e_body .- pid.Kd_pos .* de_body
    return V_cmd
end

function pid_wrench!(bus, xhat, ref, pid::PIDController, dt; mode::Symbol=:velocity)
    t = bus.t_now[]
    if mode == :velocity
        Vx_d, Vy_d = ref.Vx(t), ref.Vy(t)
        omega_d    = ref.Wz(t)
        e = SVector(xhat[1] - Vx_d, xhat[2] - Vy_d, xhat[3] - omega_d)
    elseif mode == :pose
        ref::Main.Profiles.PosRef
        V_cmd = pose_outer_loop(xhat, ref, pid, t, dt)
        e = SVector(xhat[1] - V_cmd[1], xhat[2] - V_cmd[2], xhat[3] - V_cmd[3])
    else
        error("pid_wrench! mode must be :velocity or :pose; got $mode")
    end

    de = pid.initialized ? (e - SVector(pid.prev_e...)) / dt : SVector(0.0, 0.0, 0.0)
    pid.prev_e .= e
    pid.initialized = true

    bus.I_pid = clamp.(bus.I_pid .+ dt .* e, .-pid.I_max, pid.I_max)
    W = .-pid.Kp .* e .- pid.Ki .* bus.I_pid .- pid.Kd .* de
    return W
end

# -----------------------------------------------------------------------------
# MPC controller — receding-horizon QP over per-wheel motor voltages (OSQP)
# -----------------------------------------------------------------------------
# Decision variable  U = [V₁…V₄]_{k=1..Np}  (per-wheel motor voltage over the
# horizon).  Linear discrete body model x=[Vx,Vy,ψ̇]; only the first-step voltage
# is applied (receding horizon) and mapped back to a task-space wrench so the
# mixer blends MPC uniformly with ASMC/PID.
#
# Cost:  Σ‖x_k − x_ref,k‖²_Q + ‖U_k‖²_R + ‖ΔU_k‖²_S
# HARD constraints (enforced inside the QP, not just penalised):
#   • voltage box      −V_max ≤ V_{k,j} ≤ V_max
#   • current limit    |i_{k,j}| ≤ i_max  ⇔  Kb·G·ω_j − i_max·Ra ≤ V ≤ Kb·G·ω_j + i_max·Ra
#                       (folded into the per-wheel voltage bounds at the frozen
#                        measured ω; at Ra=2.0/V_max=24 the voltage box binds first,
#                        so the effective wheel-torque cap is ≈9.35 N·m)
#   • slew-rate        |V_{k,j} − V_{k−1,j}| ≤ dV_max·dt   (first step vs last applied V)
# Physical limits (V_max, i_max, dV_max) come from `motor` — single source of
# truth shared with the plant motor map and the mixer.  Q/R/S/Np are the MPC
# tuning knobs.  Falls back to the last applied voltage on QP infeasibility.
Base.@kwdef mutable struct MPCController
    Np::Int = 10
    Q::SVector{3,Float64} = SVector(50.0, 50.0, 80.0)   # velocity-mode state weights
    R::SVector{4,Float64} = SVector(0.01, 0.01, 0.01, 0.01)  # voltage-effort weights
    S::SVector{4,Float64} = SVector(0.05, 0.05, 0.05, 0.05)  # voltage-rate weights
    Np_pose::Int = 15
    Q_pose::SVector{6,Float64} = SVector(10.0, 10.0, 5.0, 1.0, 1.0, 0.5)
    rate_hz::Float64 = 100.0
    use_ltv::Bool = true    # true: re-linearize along the reference each step (LTV);
                            # false: freeze linearization at the current state (LTI)
end

"""
    _mpc_body_matrices(params, motor, dt) -> (A, B)

Linear discrete body model  x_{k+1} = A·x_k + B·U_k , state x=[Vx,Vy,ψ̇],
input U=[V₁…V₄].  B folds the quasi-static voltage→wheel-torque sensitivity
dτ/dV = G·η·Kt/Ra, the O-config allocation, and M_aug⁻¹; A carries the linear
viscous (p1) damping.  (Nominal model for prediction only; the applied command
still round-trips through the exact nonlinear motor map.)
"""
function _mpc_body_matrices(params, motor, dt)
    R, l, h = params.R, params.l, params.h
    lever = R / (l + h)
    # O-config allocation Amix (4×3): task wrench [Wx,Wy,Wψ] → 4 wheel torques
    # (identical 0.25 mix to mixer.jl / run_one.jl:504–512).
    Amix = SMatrix{4,3,Float64,12}(
         0.25, 0.25, 0.25, 0.25,          # Wx column
        -0.25, 0.25, 0.25,-0.25,          # Wy column
        -0.25*lever, 0.25*lever, -0.25*lever, 0.25*lever)  # Wψ column
    dTaudV = motor.G * motor.eta * motor.Kt / motor.Ra
    # Wheel torques → body-accel-generating wrench.  Use pinv(Amix) (CONSISTENT
    # with the output conversion W = pinv(Amix)·τ) and the physical torque→force
    # factor 1/R on the translational channels (yaw uses the I_psi path, no R).
    # The old `transpose(Amix)·diag(R,R,1)` was ~R² too small AND inconsistent
    # with the output map, so the QP saw voltage as near-powerless → under-actuated.
    pinvA = inv(transpose(Amix) * Amix) * transpose(Amix)          # 3×4 = pinv(Amix)
    Bwrench = SMatrix{3,4,Float64,12}(Diagonal(SVector(1.0/R, 1.0/R, 1.0)) * pinvA)
    A = SMatrix{3,3,Float64,9}(I) -
        dt .* SMatrix{3,3,Float64,9}(Diagonal(SVector(4*params.p1_case1/R^2,
                                                      4*params.p1_case1/R^2, 0.0)) * params.M_aug_inv)
    B = dt .* params.M_aug_inv * Bwrench .* dTaudV
    return A, B, Amix
end

# --- LTV linearization helpers -------------------------------------------------
# Coriolis/centrifugal Jacobian ∂C/∂v at operating velocity v=[Vx,Vy,ψ̇], where
# C(v) = [ms·ψ̇·Vy + m·aX·ψ̇²;  −ms·ψ̇·Vx + m·aY·ψ̇²;  −m·ψ̇·(aX·Vx + aY·Vy)]
# (matches the nonlinear RHS0/1/2 terms in plant_rhs!).
function _coriolis_jac(params, v)
    ms, m, aX, aY = params.ms, params.m, params.aX, params.aY
    Vx, Vy, w = v[1], v[2], v[3]
    return @SMatrix [ 0.0      ms*w     ms*Vy + 2*m*aX*w;
                     -ms*w     0.0     -ms*Vx + 2*m*aY*w;
                     -m*w*aX  -m*w*aY  -m*(aX*Vx + aY*Vy) ]
end

# Discrete 3-state velocity transition A = I + dt·M_aug⁻¹·(J_cor(v_op) − viscous),
# linearized at the operating velocity v_op — LTV: v_op varies along the horizon,
# so Coriolis coupling is captured per step instead of dropped.
function _mpc_vel_A(params, dt, v_op)
    R = params.R
    Jc   = _coriolis_jac(params, v_op)
    visc = Diagonal(SVector(4*params.p1_case1/R^2, 4*params.p1_case1/R^2, 0.0))
    Ac   = params.M_aug_inv * (Jc - visc)
    return SMatrix{3,3,Float64,9}(I) + dt .* Ac
end

# 6-state pose transition at operating (v_op, ψ_op): position integrates body
# velocity through R(ψ_op); the velocity block is the LTV _mpc_vel_A.
function _mpc_pose_A(params, dt, v_op, psi)
    Av = _mpc_vel_A(params, dt, v_op)
    c, s = cos(psi), sin(psi)
    A = zeros(MMatrix{6,6,Float64,36})
    A[1,1]=1.0; A[1,4]= dt*c; A[1,5]=-dt*s
    A[2,2]=1.0; A[2,4]= dt*s; A[2,5]= dt*c
    A[3,3]=1.0; A[3,6]= dt
    A[4,4]=Av[1,1]; A[4,5]=Av[1,2]; A[4,6]=Av[1,3]
    A[5,4]=Av[2,1]; A[5,5]=Av[2,2]; A[5,6]=Av[2,3]
    A[6,4]=Av[3,1]; A[6,5]=Av[3,2]; A[6,6]=Av[3,3]
    return SMatrix{6,6,Float64,36}(A)
end

"""
    mpc_wrench!(bus, xhat, ref, params, motor, mpc; mode=:velocity) -> SVector{3}

Receding-horizon QP-MPC task-space wrench.  LTV: BOTH modes re-linearize the
no-slip model at the REFERENCE operating point (velocity, heading, Coriolis) at
every horizon step, so the prediction follows the trajectory as the regime
changes — instead of a single frozen linearization.  `:velocity` tracks a VelRef
(3-state), `:pose` a PosRef (6-state position-augmented).  Falls back to the last
applied voltage on solver failure/infeasibility.
"""
function mpc_wrench!(bus, xhat, ref, params, motor, mpc::MPCController; mode::Symbol=:velocity)
    t  = bus.t_now[]
    dt = 1.0 / mpc.rate_hz

    # Frozen measured wheel speed (for the current-limit box and the wrench map).
    ω = SVector{4,Float64}(bus.y_last.ω[1], bus.y_last.ω[2], bus.y_last.ω[3], bus.y_last.ω[4])

    # Voltage→Δv input matrix (B) and O-config allocation (Amix); both ψ-independent.
    _, Bvel, Amix = _mpc_body_matrices(params, motor, dt)

    if mode == :velocity
        n, m, Np = 3, 4, mpc.Np
        Bm = Matrix(Bvel)
        x0 = SVector(xhat[1], xhat[2], xhat[3])
        Qtrack = mpc.Q
        # LTV: linearize per step at the reference operating velocity; LTI: freeze
        # at the current estimated velocity for all steps.
        As = Matrix{Float64}[]
        for k in 1:Np
            tk = t + (k-1)*dt
            v_op = mpc.use_ltv ? SVector(ref.Vx(tk), ref.Vy(tk), ref.Wz(tk)) :
                                 SVector(xhat[1], xhat[2], xhat[3])
            push!(As, Matrix(_mpc_vel_A(params, dt, v_op)))
        end
        xref_fn = tk -> SVector(ref.Vx(tk), ref.Vy(tk), ref.Wz(tk))

    elseif mode == :pose
        ref::Main.Profiles.PosRef
        n, m, Np = 6, 4, mpc.Np_pose
        Bm = zeros(6, 4);  Bm[4:6, :] .= Matrix(Bvel)
        x0 = SVector(xhat[5], xhat[6], xhat[4], xhat[1], xhat[2], xhat[3])
        Qtrack = mpc.Q_pose
        # LTV: linearize per step at reference heading+velocity; LTI: freeze at
        # the current estimated heading+velocity for all steps.
        As = Matrix{Float64}[]
        for k in 1:Np
            tk = t + (k-1)*dt
            if mpc.use_ltv
                psi_k = ref.psi(tk)
                ck, sk = cos(psi_k), sin(psi_k)
                Vxw, Vyw = ref.Vxo(tk), ref.Vyo(tk)
                v_op = SVector(Vxw*ck + Vyw*sk, -Vxw*sk + Vyw*ck, ref.om(tk))
            else
                psi_k = xhat[4]
                v_op = SVector(xhat[1], xhat[2], xhat[3])
            end
            push!(As, Matrix(_mpc_pose_A(params, dt, v_op, psi_k)))
        end
        xref_fn = function (tk)
            ck, sk = cos(ref.psi(tk)), sin(ref.psi(tk))
            Vxw, Vyw = ref.Vxo(tk), ref.Vyo(tk)
            return SVector(ref.xo(tk), ref.yo(tk), ref.psi(tk),
                           Vxw*ck + Vyw*sk, -Vxw*sk + Vyw*ck, ref.om(tk))
        end
    else
        error("mpc_wrench! mode must be :velocity or :pose; got $mode")
    end

    # --- Condensed LTV prediction  x_k = Φ(k,0)·x0 + Σ_j Φ(k,j)·B·U_j -------
    # Φ(k,j) = As[k]·As[k-1]·…·As[j+1]  (product of the intervening transitions).
    Sx = zeros(n*Np, n)
    Su = zeros(n*Np, m*Np)
    for k in 1:Np
        Phi = Matrix{Float64}(I, n, n)
        for i in k:-1:1;  Phi = Phi * As[i];  end
        Sx[(k-1)*n+1:k*n, :] = Phi
        for j in 1:k
            P = Matrix{Float64}(I, n, n)
            for i in k:-1:(j+1);  P = P * As[i];  end
            Su[(k-1)*n+1:k*n, (j-1)*m+1:j*m] = P * Bm
        end
    end

    # Future reference preview.
    xref = zeros(n*Np)
    for k in 1:Np
        tk = t + k*dt
        xref[(k-1)*n+1:k*n] = collect(xref_fn(tk))
    end

    # --- Rate-difference operator D (D·U = [U₁; U₂−U₁; …]) -------------------
    D = zeros(m*Np, m*Np)
    for k in 1:Np
        D[(k-1)*m+1:k*m, (k-1)*m+1:k*m] = Matrix{Float64}(I, m, m)
        if k > 1
            D[(k-1)*m+1:k*m, (k-2)*m+1:(k-1)*m] = -Matrix{Float64}(I, m, m)
        end
    end
    # Previous applied voltage padded to full horizon; ΔU_1 is measured vs it.
    p_prev = vcat(collect(bus.mpc_last_u), zeros(m*(Np-1)))

    # --- Cost  0.5 Uᵀ H U + fᵀ U -------------------------------------------
    Qd = Diagonal(repeat(collect(Qtrack), Np))
    Rd = Diagonal(repeat(collect(mpc.R), Np))
    Sd = Diagonal(repeat(collect(mpc.S), Np))
    H = Su' * Qd * Su + Rd + D' * Sd * D
    H = (H + H') / 2 + 1e-6 * I
    f = Su' * Qd * (Sx * x0 .- xref) .- D' * Sd * p_prev

    # --- Hard constraints  l ≤ [I; D]·U ≤ u --------------------------------
    vlo = zeros(m); vhi = zeros(m)
    for j in 1:m
        centre = motor.Kb * motor.G * ω[j]
        vlo[j] = clamp(centre - motor.i_max * motor.Ra, -motor.V_max, motor.V_max)
        vhi[j] = clamp(centre + motor.i_max * motor.Ra, -motor.V_max, motor.V_max)
    end
    lb_box = repeat(vlo, Np);  ub_box = repeat(vhi, Np)
    dstep  = motor.dV_max * dt
    lb_rate = p_prev .- dstep
    ub_rate = p_prev .+ dstep

    Acon = vcat(sparse(1.0I, m*Np, m*Np), sparse(D))
    lcon = vcat(lb_box, lb_rate)
    ucon = vcat(ub_box, ub_rate)

    # --- Solve (warm-started, polished) ------------------------------------
    model = OSQP.Model()
    OSQP.setup!(model; P = sparse(H), q = Vector(f), A = Acon, l = lcon, u = ucon,
                verbose = false, warm_start = true, polish = true)
    if length(bus.mpc_warm) == m*Np
        OSQP.warm_start!(model; x = bus.mpc_warm)
    end
    res = OSQP.solve!(model)

    if res.info.status_val in (1, 2)          # SOLVED, SOLVED_INACCURATE
        u0 = SVector{4,Float64}(res.x[1], res.x[2], res.x[3], res.x[4])
        bus.mpc_last_u = u0
        bus.mpc_warm   = copy(res.x)
    else
        u0 = bus.mpc_last_u                    # feasible fallback
    end

    # Map the applied voltage back to an equivalent task-space wrench.
    τ_u0 = Main.PlantMod.motor_torque(u0, ω, motor)
    W = SVector{3,Float64}(pinv(Matrix(Amix)) * Vector(τ_u0))
    return W
end

end # module
