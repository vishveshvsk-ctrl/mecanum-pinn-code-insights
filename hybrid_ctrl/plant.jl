# =============================================================================
# hybrid_ctrl/plant.jl — voltage-input plant + youBot DC-motor actuator model
# =============================================================================
module PlantMod

using StaticArrays
using LinearAlgebra

export MotorParams, PlantODEParams, motor_torque, motor_torque_inverse,
       plant_rhs!, NSTATE_QUASI, NSTATE_FULL

const NSTATE_QUASI = 30
const NSTATE_FULL  = 34

"""
    MotorParams

youBot base DC-motor + gearbox parameters (from youbot_driver/config/youbot-base.cfg).
`dynamic_electrical=true` appends four motor-current states to the plant state.
"""
Base.@kwdef struct MotorParams
    Kt::Float64    = 0.0335           # N·m/A  (torque constant)
    Kb::Float64    = 0.0335           # V·s/rad (back-EMF, ideal-motor equality)
    G::Float64     = 9405.0 / 364.0   # ≈ 25.84  gear reduction motor:wheel
    Ra::Float64    = 2.0              # Ω   winding resistance (calibrate; 2.0 lets ~12 A flow at 24 V)
    La::Float64    = 0.0              # H   (quasi-static default)
    i_max::Float64 = 12.8             # A   current limit (G·η·Kt·12.8A ≈ 9.97 N·m at η=0.9).
                                      #     NOTE: at Ra=2.0, V_max=24 the voltage limit binds first
                                      #     (i≈V_max/Ra=12 A), so the effective torque cap is ≈9.35 N·m
                                      #     and this current limit stays near-slack.
    V_max::Float64 = 24.0             # V   (battery bus voltage)
    dV_max::Float64 = 200.0           # V/s per-second voltage slew limit (hard in MPC + mixer)
    eta::Float64   = 0.9              # gear efficiency (0.9 makes i_max=12.8 A ≈ 10 N·m cap)
    tau_f::Float64 = 0.01             # N·m motor-shaft friction
    cpr::Int       = 4000             # encoder counts/rev on MOTOR shaft
    dynamic_electrical::Bool = false
end

function nstate(m::MotorParams)
    return m.dynamic_electrical ? NSTATE_FULL : NSTATE_QUASI
end

"Smooth sign for C∞ RHS (avoid raw sign near zero)."  
_smoothsign(x, k=200.0) = tanh(k * x)

"""
    motor_torque(v_cmd, ω, motor) -> SVector{4}

Quasi-static DC-motor map: τ_wheel = G·η·(Kt·i − τ_f·sign(ω_m)) where
i = clamp((V − Kb·ω_m)/Ra, ±i_max) and ω_m = G·ω.  Full-electrical variant
reads currents from an external vector (not handled here; see plant_rhs!).
"""
function motor_torque(v_cmd::SVector{4}, ω::SVector{4}, motor::MotorParams)
    ωm = motor.G .* ω
    i  = clamp.((v_cmd .- motor.Kb .* ωm) ./ motor.Ra, -motor.i_max, motor.i_max)
    τm = motor.Kt .* i .- motor.tau_f .* _smoothsign.(ωm)
    return motor.G .* motor.eta .* τm
end

"""
    motor_torque_inverse(τ_wheel, ω, motor) -> SVector{4}

Invert the quasi-static motor map for a desired wheel torque: solve for V such
that motor_torque(V,ω) ≈ τ_wheel.  Clamped to ±V_max and to |i|≤i_max.
"""
function motor_torque_inverse(τ_wheel::SVector{4}, ω::SVector{4}, motor::MotorParams)
    ωm = motor.G .* ω
    τm_des = τ_wheel ./ (motor.G .* motor.eta)
    i_des  = (τm_des .+ motor.tau_f .* _smoothsign.(ωm)) ./ motor.Kt
    i_cl   = clamp.(i_des, -motor.i_max, motor.i_max)
    V      = motor.Ra .* i_cl .+ motor.Kb .* ωm
    return clamp.(V, -motor.V_max, motor.V_max)
end

"""
    PlantODEParams

Immutable param pack passed to the ODE RHS.  `bus` is the shared mutable
ControllerBus; the RHS only reads `bus.v_cmd`.
"""
struct PlantODEParams{P,L,M,B}
    params::P        # PlatformParams
    chi::Float64
    p1::Float64
    p2::Float64
    coupling::Symbol
    lugre::L         # LuGreParams
    motor::M         # MotorParams
    bus::B           # ControllerBus
end

"Layout accessors (quasi-static default)."
@inline idx_Vx()   = 1; @inline idx_Vy() = 2; @inline idx_psidot() = 3; @inline idx_psi() = 4
@inline idx_theta() = 5:8; @inline idx_omega() = 9:12; @inline idx_gamma() = 13:16
@inline idx_Xo()    = 17; @inline idx_Yo() = 18
@inline idx_zx()    = 19:22; @inline idx_zy() = 23:26; @inline idx_zs() = 27:30
@inline idx_current(m::MotorParams) = m.dynamic_electrical ? (31:34) : (0:-1)

"""
    plant_rhs!(du, u, p, t)

Voltage-input plant RHS.  Reads `p.bus.v_cmd` (ZOH motor voltage), forms wheel
torque via `motor_torque`, then integrates the same roller+LuGre physics as
`run_one.jl:dynamics_full_mf_asmc!` (lines 675–742).  No controller logic.

State layout (quasi-static): see idx_* accessors above.
"""
function plant_rhs!(du, u, p::PlantODEParams, t)
    params, chi, p1, p2, coupling, lugre, motor, bus =
        p.params, p.chi, p.p1, p.p2, p.coupling, p.lugre, p.motor, p.bus

    Vx, Vy, psi_dot, psi = u[1], u[2], u[3], u[4]
    ti = SVector(u[5],  u[6],  u[7],  u[8])
    wi = SVector(u[9],  u[10], u[11], u[12])
    gi = SVector(u[13], u[14], u[15], u[16])
    zx = SVector(u[19], u[20], u[21], u[22])
    zy = SVector(u[23], u[24], u[25], u[26])
    zs = SVector(u[27], u[28], u[29], u[30])

    px, py = params.wc_x, params.wc_y
    R, Rd  = params.R, params.Ra
    delta  = params.delta
    aX, aY = params.aX, params.aY
    ms, m  = params.ms, params.m

    sdi = sin.(delta); cdi = cos.(delta); tdi = tan.(delta)
    ti_t  = Main.sawtooth_approx.(ti)
    sti_t = sin.(ti_t);  cti_t = cos.(ti_t);  tti_t = tan.(ti_t)
    DYi   = Rd .* tdi .* tti_t

    Vpi_x = @. Vx - psi_dot * (py + DYi) - wi * R +
               gi * sdi * (Rd * cti_t - R) + DYi * gi * cdi * sti_t
    Vpi_y = @. Vy + psi_dot * px +
               gi * cdi * (R * cti_t - Rd)
    wzi   = @. psi_dot - gi * (-sti_t * cdi)

    Ni = params.N_per_roller
    fr = ntuple(4) do i
        Main.lugre_dyn_rates(lugre, coupling, params.f_coulomb, Ni[i], chi,
                             wzi[i], Vpi_x[i], Vpi_y[i], zx[i], zy[i], zs[i])
    end
    Fx_i = SVector(fr[1][1], fr[2][1], fr[3][1], fr[4][1])
    Fy_i = SVector(fr[1][2], fr[2][2], fr[3][2], fr[4][2])
    Mz_i = SVector(fr[1][3], fr[2][3], fr[3][3], fr[4][3])
    dzx  = SVector(fr[1][4], fr[2][4], fr[3][4], fr[4][4])
    dzy  = SVector(fr[1][5], fr[2][5], fr[3][5], fr[4][5])
    dzs  = SVector(fr[1][6], fr[2][6], fr[3][6], fr[4][6])

    # Actuator: commanded voltage -> wheel torque (back-EMF acts every RHS eval)
    if motor.dynamic_electrical
        i_cur = SVector(u[31], u[32], u[33], u[34])
        ωm    = motor.G .* wi
        τm    = motor.Kt .* i_cur .- motor.tau_f .* _smoothsign.(ωm)
        Mi_sat = motor.G .* motor.eta .* τm
    else
        Mi_sat = motor_torque(bus.v_cmd, wi, motor)
    end

    Mz_rolleri = @. px * Fy_i - (py  + DYi) * Fx_i + Mz_i
    RHS0 = sum(Fx_i)          + ms * psi_dot * Vy  + m * aX * psi_dot^2
    RHS1 = sum(Fy_i)          - ms * psi_dot * Vx  + m * aY * psi_dot^2
    RHS2 = sum(Mz_rolleri)    - m * psi_dot * (aX * Vx + aY * Vy)
    dv   = params.M_inv * SVector(RHS0, RHS1, RHS2)

    dwi = @. (Mi_sat - Fx_i * R - p1 * wi) / params.J_wheel
    dgi = @. (- p2 * gi
              - Fx_i * (sdi * (R - Rd * cti_t) - DYi * sti_t * cdi)
              - Fy_i * cdi * (Rd - R * cti_t)
              - Mz_i * sti_t * cdi) / params.J_roller / params.rollers_per_wheel

    du[1] = dv[1];  du[2] = dv[2];  du[3] = dv[3]
    du[4] = psi_dot
    @inbounds for i in 1:4
        du[4+i]  = wi[i]
        du[8+i]  = dwi[i]
        du[12+i] = dgi[i]
    end
    du[17] = Vx * cos(psi) - Vy * sin(psi)
    du[18] = Vx * sin(psi) + Vy * cos(psi)
    @inbounds for i in 1:4
        du[18+i] = dzx[i]
        du[22+i] = dzy[i]
        du[26+i] = dzs[i]
    end

    if motor.dynamic_electrical
        ωm = motor.G .* wi
        i_cur = SVector(u[31], u[32], u[33], u[34])
        di = @. (bus.v_cmd[i] - motor.Ra * i_cur[i] - motor.Kb * ωm[i]) / max(motor.La, 1e-9)
        du[31] = di[1]; du[32] = di[2]; du[33] = di[3]; du[34] = di[4]
    end
    return nothing
end

end # module
