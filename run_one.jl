# ============================================================================
# run_one.jl  —  AUTO-GENERATED from the Mecanum ASMC+DOB Julia notebook.
#
# Regenerate with:
#   python extract_run_one.py --notebook <notebook>.ipynb --out run_one.jl
#
# Provides (consumed by Data_Generation_Julia.jl after `include`):
#   - Profiles, DataStore            (via include("profiles.jl"/"datastore.jl"))
#   - PlatformParams(base; mu_friction), ASMCParams, ESOParams, LuGreParams
#   - dynamics_full_mf_asmc!, build_initial_state, run_one_chi
#   - asmc_torques, asmc_torques_vel, lugre / lugre_dyn_rates / coupling_of,
#     sawtooth_approx
# DO NOT hand-edit — changes are overwritten on the next extraction. Edit the
# notebook instead.
# ============================================================================

# --- Stand-in defaults for the stripped `parameters` cell. -------------------
# The sweep driver overrides the physics point per job; these exist only so the
# definition cells compile, and so the notebook-only sawtooth diagnostic plot
# (gated on `!write_data`) is SKIPPED at include time.
write_data     = true            # true ⇒ skip the notebook-only diagnostic plot
mu_friction    = 0.5
friction_case  = 1
friction_model = :lugre_adamov
use_dob        = false           # ASMCParams' gamma_y/gamma_psi defaults read this;
                                 # the sweep driver overrides it from base.toml [dob].enable.

# --- [skipped: parameters cell 2 — replaced by BANNER stand-ins] ---

# [extract] dropped `using Revise` (batch script, no live reload)
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqCallbacks
using StaticArrays
using DataFrames
using Arrow
using JLD2
using Plots
using Printf
using ProgressMeter
using Base.Threads: @spawn

# Reproducibility
import Random
Random.seed!(0)

# ---------------------------
# Pipeline modules — keep profiles.jl / datastore.jl / configs/ beside the notebook
# ---------------------------
include("profiles.jl");  using .Profiles
include("datastore.jl"); using .DataStore
println("Profiles: ", join(sort(collect(keys(Profiles.BUILDERS))), ", "))

# ---------------------------
# Platform geometry and parameters
# ---------------------------
struct PlatformParams
    # Geometry
    h::Float64          # half-length (m)
    l::Float64          # half-width (m)
    R::Float64          # wheel outer radius (m)
    Ra::Float64         # roller axle distance (m)
    # Masses/inertias
    m::Float64
    m_wheel::Float64
    J_wheel::Float64
    J_roller::Float64
    ms::Float64         # total mass
    Is::Float64         # platform moment of inertia
    # Viscous friction cases
    p1_case1::Float64
    p2_case1::Float64
    p1_case2::Float64
    p2_case2::Float64
    # Friction
    f_coulomb::Float64
    N_total::Float64
    rollers_per_wheel::Int
    # Per-wheel static fields (SVector for stack allocation)
    delta::SVector{4,Float64}       # roller axis angle
    wc_x::SVector{4,Float64}        # wheel center X
    wc_y::SVector{4,Float64}        # wheel center Y
    aX::Float64                     # COM offset X
    aY::Float64                     # COM offset Y
    N_per_roller::SVector{4,Float64}
    M_inv::SMatrix{3,3,Float64,9}        # plant body mass matrix (no wheel reflection)
    M_aug::SMatrix{3,3,Float64,9}        # augmented mass matrix (with wheel reflection)
    M_aug_inv::SMatrix{3,3,Float64,9}    # inverse of the augmented mass matrix
    Max_torque::Float64
end

function PlatformParams(base::AbstractDict; mu_friction::Float64 = mu_friction)
    geo = base["platform"]["geometry"];  mas = base["platform"]["mass"]
    com = base["platform"]["com_offset"]; vis = base["platform"]["viscous"]
    con = base["platform"]["contact"]
    h  = geo["h"];  l = geo["l"];  R = geo["R"];  Ra = geo["Ra"]
    m  = mas["m"];  m_wheel = mas["m_wheel"]
    J_wheel = mas["J_wheel"];  J_roller = mas["J_roller"]
    ms = m + 4.0 * m_wheel                       # derived — stays in code
    Is = mas["Is"]
    p1_case1 = vis["p1_case1"]; p2_case1 = vis["p2_case1"]
    p1_case2 = vis["p1_case2"]; p2_case2 = vis["p2_case2"]
    f_coulomb = mu_friction                      # sweep variable: kwarg wins
    N_total = m * 9.81
    rollers_per_wheel = con["rollers_per_wheel"]
    delta = SVector(-π/4, π/4, π/4, -π/4)        # structural, not config
    wc_x  = SVector(h, h, -h, -h);  wc_y = SVector(l, -l, l, -l)
    aX = com["aX"];  aY = com["aY"]

    N_per_roller = SVector(
        N_total/4 * (1 + aX/h + aY/l) + m_wheel * 9.81,
        N_total/4 * (1 + aX/h - aY/l) + m_wheel * 9.81,
        N_total/4 * (1 - aX/h + aY/l) + m_wheel * 9.81,
        N_total/4 * (1 - aX/h - aY/l) + m_wheel * 9.81
    )
    M_mat = @SMatrix [ ms       0.0     -m*aY;
                       0.0      ms       m*aX;
                      -m*aY     m*aX     Is   ]
    M_inv = inv(M_mat)
    # MIMO DOB uses the augmented mass matrix that the equivalent control already inverts
    # (m_tilde and I_psi_aug pick up the wheel rotational inertia reflected to the body).
    m_tilde   = ms + 4 * J_wheel / R^2
    I_psi_aug = Is + 4 * (l + h)^2 / R^2 * J_wheel
    M_aug = @SMatrix [ m_tilde   0.0       -m*aY ;
                       0.0       m_tilde    m*aX ;
                      -m*aY      m*aX       I_psi_aug ]
    M_aug_inv = inv(M_aug)
    Max_torque = 10.0
    PlatformParams(h, l, R, Ra, m, m_wheel, J_wheel, J_roller, ms, Is,
                   p1_case1, p2_case1, p1_case2, p2_case2, f_coulomb,
                   N_total, rollers_per_wheel, delta, wc_x, wc_y,
                   aX, aY, N_per_roller, M_inv, M_aug, M_aug_inv, Max_torque)
end

# ---------------------------
# ASMC Controller Configuration  (per-axis gains)
#   λ ordering  ψ > y > x   — closed-loop bandwidth / authority.
#   γ, K_max    the MIMO DOB carries the lumped disturbance on all three axes, so γ_y
#               and γ_psi are reduced when DOB is active to avoid the adaptive gain and
#               the DOB double-counting the same disturbance. Y > X because My_eq carries
#               the extra 8·p2/(R-Rd)² roller-viscous drag term that Mx_eq does not.
# ---------------------------
Base.@kwdef struct ASMCParams
    gamma_x::Float64     = 8.0      # adaptation rate (X)
    gamma_y::Float64     = use_dob ? 12.0 : 15.0      # adaptation rate (Y)
    gamma_psi::Float64   = use_dob ? 12.0 : 25.0     # adaptation rate (heading) — reduced when DOB active
    eps::Float64         = 0.0175    # boundary-layer thickness (X/Y)
    eps_psi::Float64     = 0.08     # boundary-layer thickness (yaw residual) — slightly thicker
    K_max_x::Float64     = 60.0     # gain saturation (X)
    K_max_y::Float64     = 80.0     # gain saturation (Y)
    K_max_psi::Float64   = 100.0    # gain saturation (heading)
    lam_x_min::Float64   = 0.1
    lam_x_max::Float64   = 1.5
    lam_y_min::Float64   = 0.2
    lam_y_max::Float64   = 2.5
    lam_psi_min::Float64 = 0.5
    lam_psi_max::Float64 = 5.0
    mu_xy::Float64       = 25.0     # sharpness of λ(e) curve for X/Y
    mu_psi::Float64      = 100.0    # sharpness of λ(e) curve for heading
    sigma_psi::Float64   = 0.25
    sigma_x::Float64   = 0.5
    sigma_y::Float64   = 0.25
    decay_k::Float64   = 0.25
    K_x0::Float64   = 5.0
    K_y0::Float64   = 5.0
    K_psi0::Float64   = 20.0
end

# ---------------------------
# MIMO disturbance-observer (reduced-order DOB) configuration.
#   Nominal:   v̇ = M_aug⁻¹·F_p + δ       (v=[Vx,Vy,ψ̇]ᵀ; δ=[δ_x,δ_y,δ_ψ]ᵀ)
#   Estimate:  δ̂ = ζ + ω_o·v             (ζ ∈ R³  at state[34], state[35], state[36])
#   ODE:       ζ̇ = -ω_o·ζ - ω_o²·v - ω_o·M_aug⁻¹·F_p_applied
#   Compens:   ΔM_c = -diag(R,R,1)·M_aug·δ̂   added to the equivalent-control wrench
#   F_p_applied recovered from saturated wheel torques via the un-map (see asmc_torques).
#   One scalar bandwidth ω_o serves all three channels (Ω = ω_o·I₃).
# ---------------------------
Base.@kwdef struct ESOParams
       kind::Symbol         = :super_twisting   # :super_twisting | :linear
       # --- reduced-order linear DOB bandwidths (used when kind == :linear) ---
       omega_o_x::Float64   = 20.0
       omega_o_y::Float64   = 20.0
       omega_o_psi::Float64 = 20.0
       # --- super-twisting observer gains (used when kind == :super_twisting) ---
       #   v̂̇ = a_nom + δ̂ + k₁·φ₁(v−v̂);  δ̂̇ = k₂·φ₂(v−v̂)
       k1_x::Float64        = 0.0
       k2_x::Float64        = 0.0
       k1_y::Float64        = 0.0
       k2_y::Float64        = 0.0
       k1_psi::Float64      = 15.0
       k2_psi::Float64      = 80.0
       eps_obs::Float64     = 1e-2   # C∞ boundary layer for φ₁, φ₂
       enable::Bool         = true
   end

# ---------------------------
# Smoothed sawtooth: two implementations
#
# SAWTOOTH = :tanh     → tanh-smoothed peak: fast (~5× over Fourier), localized error at handoff.
# SAWTOOTH = :fourier  → 14-term Lanczos-smoothed Fourier sum (matches Python v4 notebook).
#
# Period = 30° = π/6 rad, output in radians, range ±π/12 (±15°).
# ---------------------------
const SAWTOOTH = :tanh            # switch to :fourier to reproduce Python v4 exactly
const TANH_K   = 60.0             # tanh steepness at the peaks; 20–60 is a reasonable range

# --- tanh-peak version (preferred for speed) ---
# Idea: take the *ideal* sawtooth, then soften only the peaks.
# s_ideal(θ) = ((θ + π/12) mod π/6) - π/12,  which equals  atan(tan(6θ))/6  (single branch).
# We write it as (1/6)·atan(tan(6θ)) and replace the unbounded `tan` with a bounded
# tanh-shaped peak so the function stays C∞:  sin(12θ) / (β + cos(12θ)) form.
@inline function sawtooth_tanh(θ)
    # Bounded, C∞ approximation with a single `atan` + `sin` + `cos`.
    # On the linear ramp this agrees with the ideal sawtooth to ~1e-4 rad for TANH_K ≥ 40.
    s = sin(12θ)
    c = cos(12θ)
    return atan(TANH_K * s, TANH_K * c + 1) / 12  # 2-arg atan -> smooth branch switch
end

# --- 14-term Lanczos-smoothed Fourier sum (reference / backward compatibility) ---
@inline function sawtooth_fourier(θ)
     (0.165450866601201 * sin( 12*θ)
    - 0.080917684102647 * sin( 24*θ)
    + 0.051971626877147 * sin( 36*θ)
    - 0.036960991377466 * sin( 48*θ)
    + 0.027566444771090 * sin( 60*θ)
    - 0.021022964684463 * sin( 72*θ)
    + 0.016151334251121 * sin( 84*θ)
    - 0.012365865286014 * sin( 96*θ)
    + 0.009343539859761 * sin(108*θ)
    - 0.006891611192772 * sin(120*θ)
    + 0.004887403818508 * sin(132*θ)
    - 0.003248226679822 * sin(144*θ)
    + 0.001915211458051 * sin(156*θ)
    - 0.000844137074496 * sin(168*θ))
end

# Dispatch once at include time — no per-call branch cost.
const sawtooth_approx = SAWTOOTH === :tanh ? sawtooth_tanh : sawtooth_fourier
if !write_data
    # Quick comparison plot
    let θ = range(0, 2π; length=2000)
        y_tanh    = sawtooth_tanh.(θ)
        #y_fourier = sawtooth_fourier.(θ)
        y_ideal   = @. mod(rad2deg(θ) + 15, 30) - 15
        plt = plot(rad2deg.(θ), rad2deg.(y_tanh);    label="tanh-peak", lw=2)
        #plot!(plt, rad2deg.(θ), rad2deg.(y_fourier); label="Lanczos-Fourier (N=14)", lw=2, ls=:dash)
        plot!(plt, rad2deg.(θ), y_ideal;             label="Ideal sawtooth", ls=:dot,
            xlabel="θ (deg)", ylabel="φ̃ (deg)", title="Smoothed sawtooth — peak vs Fourier")
        display(plt)
        #Plot error as well
        plt2 = plot(rad2deg.(θ), deg2rad.(y_ideal)-y_tanh; label="tanh", lw=2, xlabel="θ (deg)", ylabel="Error (rad)", xlims=(0,60))
        display(plt2)
    end
end

# ---------------------------
# Contact friction. The ODE uses the DYNAMIC LuGre (bristle states integrated by the solver).
# `multicomponent_friction` (Adamov static) and `lugre_ss_friction` (steady-state) are kept as
# references only; they are NOT used in the RHS.
# ---------------------------

# (reference) original Adamov static law
# @inline function multicomponent_friction(f::Real, N::Real, chi::Real,
#                                          w_z::Real, Vpx::Real, Vpy::Real; eps::Real = 1e-6)
#     Vp = sqrt(Vpx^2 + Vpy^2 + eps^2); awz = sqrt(w_z^2 + eps^2)
#     fs = f * tanh(200 * Vp)
#     Fx = -fs * N * Vpx / ((8/(3π))*awz*chi + Vp)
#     Fy = -fs * N * Vpy / ((8/(3π))*awz*chi + Vp)
#     Mz = -fs * N * (chi^2 * w_z) / ((16/(3π))*awz*chi + 5*Vp)
#     return Fx, Fy, Mz
# end

# --- LuGre / Stribeck parameters (CALIBRATE to your polyurethane rollers) ---
Base.@kwdef struct LuGreParams
    sigma0::Float64   = 1.64e3    # translational bristle stiffness [1/m]
    sigma1::Float64   = 1.6    # translational micro-damping     [s/m]
    sigma2::Float64   = 0.0      # translational viscous           [s/m]  (0: handled by p1/p2)
    sigma0_s::Float64 = 1.09e3    # spin bristle stiffness          [1/m]
    sigma1_s::Float64 = 1.1   # spin micro-damping              [s/m]
    sigma2_s::Float64 = 0.0      # spin viscous
    stiction_ratio::Float64 = 1.1  # μs/μc  (1.0 disables Stribeck dip)
    v_str::Float64    = 0.01     # translational Stribeck velocity [m/s]
    w_str::Float64    = 0.01     # spin Stribeck scale (mean local slip) [m/s]
    use_mindlin::Bool = true     # state-based Mindlin ramp on spin→translation coupling
    mindlin_iters::Int = 2       # (only used by the steady-state reference fn below)
    eps_reg::Float64  = 1e-4     # regularisation [m/s]
end

@inline stribeck_g(s, mu_c, ratio, vs) = mu_c * (1 + (ratio - 1) * exp(-(s / vs)^2))
coupling_of(fm::Symbol) = fm === :lugre_adamov ? :adamov : :uncoupled

# DYNAMIC LuGre: returns forces/torque AND the bristle derivatives (żx, ży, żs).
@inline function lugre_dyn_rates(lg::LuGreParams, coupling::Symbol,
                                 f::Real, N::Real, chi::Real,
                                 w_z::Real, Vpx::Real, Vpy::Real,
                                 zx::Real, zy::Real, zs::Real)
    er  = lg.eps_reg
    Vp  = sqrt(Vpx^2 + Vpy^2 + er^2)
    awz = sqrt(w_z^2 + er^2)
    c_t = (8/(3π)) * awz * chi             # gross-sliding spin → translation coupling

    if coupling === :adamov
        if lg.use_mindlin
            znorm = sqrt(zx^2 + zy^2 + 1e-18)
            dstar = (lg.stiction_ratio * f) / lg.sigma0     # δ* = μs/σ0 (breakaway deflection)
            sfrac = clamp(znorm / dstar, 0.0, 1.0)
            b     = max(1 - sfrac, 1e-9)
            fsl   = 1 - b^(2/3)
        else
            fsl = 1.0
        end
        s_t = fsl * c_t + Vp
        s_s = (16/(3π)) * awz * chi + 5 * Vp
    else  # :uncoupled — no Adamov coupling in any regime
        s_t = Vp
        s_s = (16/(3π)) * awz * chi + er
    end

    g_t = stribeck_g(s_t, f, lg.stiction_ratio, lg.v_str)
    g_s = stribeck_g(s_s, f, lg.stiction_ratio, lg.w_str)

    # bristle derivatives (integrated by the solver)
    dzx = Vpx - lg.sigma0   * s_t / g_t * zx
    dzy = Vpy - lg.sigma0   * s_t / g_t * zy
    dzs = w_z - lg.sigma0_s * s_s / g_s * zs

    # forces / spin torque (include the σ1·ż micro-damping term)
    Fx = -N * (lg.sigma0   * zx + lg.sigma1   * dzx + lg.sigma2   * Vpx)
    Fy = -N * (lg.sigma0   * zy + lg.sigma1   * dzy + lg.sigma2   * Vpy)
    Mz = -N * chi^2 * (lg.sigma0_s * zs + lg.sigma1_s * dzs + lg.sigma2_s * w_z)
    return Fx, Fy, Mz, dzx, dzy, dzs
end

# (reference) steady-state algebraic LuGre — bristle eliminated, Mindlin via Picard.
@inline function mindlin_fslip(V, c, iters)
    x = V / (c + V); fs = 1.0
    @inbounds for _ in 1:iters
        b = max(1 - x, 1e-9); fs = 1 - b^(2/3); x = V / (fs * c + V)
    end
    b = max(1 - x, 1e-9); return 1 - b^(2/3)
end
@inline function lugre_ss_friction(lg::LuGreParams, coupling::Symbol,
                                   f::Real, N::Real, chi::Real, w_z::Real, Vpx::Real, Vpy::Real)
    er = lg.eps_reg; Vp = sqrt(Vpx^2+Vpy^2+er^2); awz = sqrt(w_z^2+er^2)
    c_t = (8/(3π))*awz*chi
    if coupling === :adamov
        fsl = lg.use_mindlin ? mindlin_fslip(Vp, c_t, max(lg.mindlin_iters,1)) : 1.0
        s_t = fsl*c_t + Vp; s_s = (16/(3π))*awz*chi + 5*Vp
    else
        s_t = Vp; s_s = (16/(3π))*awz*chi + er
    end
    g_t = stribeck_g(s_t,f,lg.stiction_ratio,lg.v_str)
    g_s = stribeck_g(s_s,f,lg.stiction_ratio,lg.w_str)
    return (-N*g_t*Vpx/s_t, -N*g_t*Vpy/s_t, -N*chi^2*g_s*w_z/s_s)
end

# Default LuGre parameter set.
lugre = LuGreParams()

# --- [skipped: trajectory-build cell 14 — driver builds refs per job] ---

@inline function smooth_bound(K, K_max)
    # same 1-0 smooth gate as Python v4
    return 0.5 - 0.5 * tanh(1.0 * (K - (K_max - 2.0)))
end

@inline function get_dynamic_lambda(e, edot, lam_min, lam_max, mu)
    exp_term = exp(-mu * e^2)
    lam = lam_min + (lam_max - lam_min) * exp_term
    lam_dot = -2 * mu * e * edot * (lam_max - lam_min) * exp_term
    return lam, lam_dot
end

@inline smooth_sat(M, L, n::Int = 3) = M / (1 + (M / L)^(2n))^(1 / (2n))
# Smoother but sharper saturation compared to tanh. 

# ---- C∞ regularizations for the super-twisting observer ---------------------
# Smooth everywhere (denominator never hits 0), so RadauIIA5's Newton/error
# estimator and ForwardDiff stay well-behaved as the observer error crosses 0.
@inline reg_sign(x, eps) = x / sqrt(x^2 + eps^2)        # ≈ sign(x)
@inline reg_sqrt(x, eps) = x / (x^2 + eps^2)^0.25       # ≈ |x|^½ · sign(x)

# global_to_local_frame now comes from Profiles (exported) — defining it here
# over the `using .Profiles` binding would throw a method-definition error.

# ---------------------------
# ASMC Control Law (full, smoothed) + MIMO disturbance-observer compensation.
# Returns: Mi_sw, Mi_eq (each SVector{4}), dK (SVector{3}), δ̂ (SVector{3} = [δ̂_x, δ̂_y, δ̂_ψ]).
# The equivalent-control wrench is corrected by ΔM_c = -diag(R,R,1)·M_aug·δ̂.
#   δ̂ = ζ + ω_o·v        with v = [Vx, Vy, ψ̇]ᵀ  and  ζ = [state[34], state[35], state[36]]ᵀ.
# Per-axis compensation (derivation §7, eqs 18-20):
#   ΔM_x   = -R·m̃·δ̂_x + R·m·aY·δ̂_ψ
#   ΔM_y   = -R·m̃·δ̂_y - R·m·aX·δ̂_ψ
#   ΔM_ψ   =    m·aY·δ̂_x - m·aX·δ̂_y - I_ψ·δ̂_ψ 
# ---------------------------
function asmc_torques(state::AbstractVector, t::Real,
                     params::PlatformParams, asmc::ASMCParams, eso::ESOParams)
    # Unpack params
    m, ms, Is, J1 = params.m, params.ms, params.Is, params.J_wheel
    aX, aY = params.aX, params.aY
    p1, p2 = params.p1_case1, params.p2_case1
    l, h, R, Rd = params.l, params.h, params.R, params.Ra
    mu_xy, mu_psi = asmc.mu_xy, asmc.mu_psi

    # Unpack state (Julia 1-based; matches Python v4 layout shifted by +1)
    Vx, Vy, psi_dot, psi = state[1], state[2], state[3], state[4]
    K_x, K_y, K_psi      = state[17], state[18], state[19]
    xo, yo               = state[20], state[21]

    # Disturbance estimate δ̂.  Super-twisting: δ̂ is the integrated observer
    # state directly (state[37:39]).  Linear: δ̂ = ζ + ω_o·v  (state[34:36]).
    v_body    = SVector(Vx, Vy, psi_dot)

    en = 0.0
    delta_hat = 0.0
    if eso.enable
        en = 1.0
        if eso.kind == :super_twisting
            delta_hat = en * SVector(state[37], state[38], state[39])
        else
            zeta_vec  = SVector(state[34], state[35], state[36])
            Omega     = SVector(eso.omega_o_x, eso.omega_o_y, eso.omega_o_psi)
            delta_hat = en * (zeta_vec + Omega .* v_body)
        end
    end
    
    # Effective yaw inertia (also used in the equivalent control; same value lives in M_aug[3,3])
    I_psi = Is + 4*(l + h)^2 / R^2 * J1
    
    ref = current_posref()
    # Positional errors (global frame)
    e_xo = xo - ref.xo(t)
    e_yo = yo - ref.yo(t)

    # Rotate global position errors into the body frame; e_psi uses sin(Δψ) for smooth wrap
    c_psi, s_psi = cos(psi), sin(psi)
    e_x = e_xo * c_psi + e_yo * s_psi
    e_y = -e_xo * s_psi + e_yo * c_psi
    d_psi = psi - ref.psi(t)
    e_psi = 2 * tan(d_psi/2) * (1 + 2 * (1 - cos(d_psi)))

    # Desired velocities in local frame
    Vx_d, Vy_d, omega_d = global_to_local_frame(t, psi, ref.Vxo, ref.Vyo, ref.om)

    # Velocity errors (local frame; coupled through yaw rate)
    edot_x   = Vx      - Vx_d + psi_dot * e_y
    edot_y   = Vy      - Vy_d - psi_dot * e_x
    edot_psi = psi_dot - omega_d
    edash_psi = (sec(d_psi/2))^2 * (3 - 2 * cos(d_psi)) + 4 * tan(d_psi/2) * sin(d_psi)

    # Dynamic λ — per-axis ranges (ψ > y > x)
    lam_x,   lam_dot_x   = get_dynamic_lambda(e_x,   edot_x,   asmc.lam_x_min,   asmc.lam_x_max,   mu_xy)
    lam_y,   lam_dot_y   = get_dynamic_lambda(e_y,   edot_y,   asmc.lam_y_min,   asmc.lam_y_max,   mu_xy)
    lam_psi, lam_dot_psi = get_dynamic_lambda(e_psi, edash_psi * edot_psi, asmc.lam_psi_min, asmc.lam_psi_max, mu_psi)

    # Sliding surfaces
    s_x   = edot_x   + lam_x   * e_x
    s_y   = edot_y   + lam_y   * e_y
    s_psi = edot_psi + lam_psi * e_psi

    # Smooth signum (per-axis boundary layer; yaw slightly thicker)
    ss_x   = tanh(s_x   / asmc.eps)
    ss_y   = tanh(s_y   / asmc.eps)
    ss_psi = tanh(s_psi / asmc.eps_psi)

    # Task-space switching wrench
    Mx_sw   = -K_x   * ss_x
    My_sw   = -K_y   * ss_y
    Mpsi_sw = -K_psi * ss_psi

    # Equivalent control — desired body-frame acceleration minus λ terms
    Ax_des, Ay_des, alpha_des = global_to_local_frame(t, psi, ref.Axo, ref.Ayo, ref.al)
    alpha_eq = alpha_des - lam_dot_psi * e_psi - lam_psi * edot_psi * edash_psi
    Ax_eq    = Ax_des    - lam_dot_x   * e_x   - lam_x   * edot_x   + psi_dot * (Vy_d - edot_y) - alpha_eq * e_y 
    Ay_eq    = Ay_des    - lam_dot_y   * e_y   - lam_y   * edot_y   - psi_dot * (Vx_d - edot_x) + alpha_eq * e_x

    # Base equivalent-control wrench (inverse-dynamics; no DOB compensation yet)
    Mx_eq = R * ((ms + 4*J1/R^2) * Ax_eq
                 - ms * psi_dot * Vy
                 - m  * aY * alpha_eq
                 - m  * aX * psi_dot^2
                 + 4  * p1 * Vx / R^2)

    My_eq = R * ((ms + 4*J1/R^2) * Ay_eq
                 + ms * psi_dot * Vx
                 + m  * aX * alpha_eq
                 - m  * aY * psi_dot^2
                 + (4 * p1 / R^2 + 8 * p2 / (R - Rd)^2) * Vy)

    M_psi_eq = (I_psi * alpha_eq
                - m * aY * (Ax_eq - psi_dot * Vy)
                + m * aX * (Ay_eq + psi_dot * Vx)
                + (4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-Rd)^2) * psi_dot)

    # MIMO DOB compensation  ΔM_c = -diag(R, R, 1)·M_aug·δ̂   (derivation eq 17, equivalently 18-20)
    M_aug_dh = params.M_aug * delta_hat
    Mx_eq    -= R * M_aug_dh[1]
    My_eq    -= R * M_aug_dh[2]
    M_psi_eq -=     M_aug_dh[3]

    # Map task-space wrenches to 4 wheel torques (mecanum O-configuration)
    lever = R / (l + h)
    M1_sw = 0.25 * (Mx_sw - My_sw - lever * Mpsi_sw)
    M2_sw = 0.25 * (Mx_sw + My_sw + lever * Mpsi_sw)
    M3_sw = 0.25 * (Mx_sw + My_sw - lever * Mpsi_sw)
    M4_sw = 0.25 * (Mx_sw - My_sw + lever * Mpsi_sw)

    M1_eq = 0.25 * (Mx_eq - My_eq - lever * M_psi_eq)
    M2_eq = 0.25 * (Mx_eq + My_eq + lever * M_psi_eq)
    M3_eq = 0.25 * (Mx_eq + My_eq - lever * M_psi_eq)
    M4_eq = 0.25 * (Mx_eq - My_eq + lever * M_psi_eq)

    Mi_sw = SVector(M1_sw, M2_sw, M3_sw, M4_sw)
    Mi_eq = SVector(M1_eq, M2_eq, M3_eq, M4_eq)

    # Smooth adaptive-gain dynamics, with per-axis cubic pushback & per-axis K_max
    base_dK_x   = asmc.gamma_x   * (s_x   * ss_x)
    base_dK_y   = asmc.gamma_y   * (s_y   * ss_y)
    base_dK_psi = asmc.gamma_psi * (s_psi * ss_psi)
    dK_x   = base_dK_x   * smooth_bound(K_x,   asmc.K_max_x)   - 0.1 * (K_x   / asmc.K_max_x)^3 - asmc.sigma_x   * (K_x   - asmc.K_x0   * 0.95) * exp(asmc.decay_k*(1 - s_x^2  /(9 * asmc.eps_psi^2)))
    dK_y   = base_dK_y   * smooth_bound(K_y,   asmc.K_max_y)   - 0.3 * (K_y   / asmc.K_max_y)^3 - asmc.sigma_y   * (K_y   - asmc.K_y0   * 0.95) * exp(asmc.decay_k*(1 - s_y^2  /(9 * asmc.eps_psi^2)))
    dK_psi = base_dK_psi * smooth_bound(K_psi, asmc.K_max_psi) - 0.5 * (K_psi / asmc.K_max_psi)^3 - asmc.sigma_psi * (K_psi - asmc.K_psi0 * 0.95) * exp(asmc.decay_k*(1 - s_psi^2/(9 * asmc.eps_psi^2)))

    return Mi_sw, Mi_eq, SVector(dK_x, dK_y, dK_psi), delta_hat
end

# ---------------------------
# Velocity-tracking ASMC (relative degree 1). Same signature & return tuple as
# asmc_torques, so it drops into the injected controller call in the RHS.
# Surface = body-frame velocity error (no position error, no dynamic λ); the
# equivalent control uses the body-frame accel feedforward directly.
# Requires body-frame reference getters from the spiral profiles:
#   ref.Vx, ref.Vy, ref.Wz                    (velocity, ref = current_ref())
#   ref.Ax, ref.Ay, ref.al                    (accel feedforward)
# Implemented mixed degree control as tracking pose without unbounded error is important for 
# ---------------------------
function asmc_torques_vel(state::AbstractVector, t::Real,
                          params::PlatformParams, asmc::ASMCParams, eso::ESOParams)
    # Unpack params
    m, ms, Is, J1 = params.m, params.ms, params.Is, params.J_wheel
    aX, aY = params.aX, params.aY
    p1, p2 = params.p1_case1, params.p2_case1
    l, h, R, Rd = params.l, params.h, params.R, params.Ra

    # Unpack state
    Vx, Vy, psi_dot, psi = state[1], state[2], state[3], state[4]
    K_x, K_y, K_psi = state[17], state[18], state[19]

    # Disturbance estimate δ̂ (identical to position controller)
    v_body = SVector(Vx, Vy, psi_dot)
    
    en = 0.0
    delta_hat = 0.0
    if eso.enable
        en = 1.0
        if eso.kind == :super_twisting
            delta_hat = en * SVector(state[37], state[38], state[39])
        else
            zeta_vec  = SVector(state[34], state[35], state[36])
            Omega     = SVector(eso.omega_o_x, eso.omega_o_y, eso.omega_o_psi)
            delta_hat = en * (zeta_vec + Omega .* v_body)
        end
    end

    I_psi = Is + 4*(l + h)^2 / R^2 * J1

    ref = current_ref()
    
    # ---- velocity-tracking surface (relative degree 1; body frame in translation, degree 2 in yaw) ----
    Vx_d    = ref.Vx(t)
    Vy_d    = ref.Vy(t)
    omega_d = ref.Wz(t)

    d_psi = psi - ref.psi(t)
    e_psi = 2 * tan(d_psi/2) * (1 + 2 * (1 - cos(d_psi)))
    edash_psi = (sec(d_psi/2))^2 * (3 - 2 * cos(d_psi)) + 4 * tan(d_psi/2) * sin(d_psi)
    edot_psi  = edash_psi * (psi_dot - ref.Wz(t))
    lam_psi, lam_dot_psi = get_dynamic_lambda(e_psi, edot_psi, asmc.lam_psi_min, asmc.lam_psi_max, asmc.mu_psi)
    
    s_x   = Vx      - Vx_d
    s_y   = Vy      - Vy_d
    s_psi = edot_psi + lam_psi * e_psi

    # Smooth signum (per-axis boundary layer; yaw slightly thicker)
    ss_x   = tanh(s_x   / asmc.eps)
    ss_y   = tanh(s_y   / asmc.eps)
    ss_psi = tanh(s_psi / asmc.eps_psi)

    # Task-space switching wrench
    Mx_sw   = -K_x   * ss_x
    My_sw   = -K_y   * ss_y
    Mpsi_sw = -K_psi * ss_psi

    # Equivalent control — body-frame desired-acceleration feedforward (no λ terms in translation)
    Ax_eq    = ref.Ax(t)
    Ay_eq    = ref.Ay(t)
    alpha_eq = ref.al(t) - lam_dot_psi * e_psi - lam_psi * edot_psi * edash_psi

    # Base equivalent-control wrench (inverse-dynamics; identical to position controller)
    Mx_eq = R * ((ms + 4*J1/R^2) * Ax_eq
                 - ms * psi_dot * Vy
                 - m  * aY * alpha_eq
                 - m  * aX * psi_dot^2
                 + 4  * p1 * Vx / R^2)

    My_eq = R * ((ms + 4*J1/R^2) * Ay_eq
                 + ms * psi_dot * Vx
                 + m  * aX * alpha_eq
                 - m  * aY * psi_dot^2
                 + (4 * p1 / R^2 + 8 * p2 / (R - Rd)^2) * Vy)

    M_psi_eq = (I_psi * alpha_eq
                - m * aY * (Ax_eq - psi_dot * Vy)
                + m * aX * (Ay_eq + psi_dot * Vx)
                + (4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-Rd)^2) * psi_dot)
                

    # MIMO DOB compensation  ΔM_c = -diag(R, R, 1)·M_aug·δ̂
    M_aug_dh = params.M_aug * delta_hat
    Mx_eq    -= R * M_aug_dh[1]
    My_eq    -= R * M_aug_dh[2]
    M_psi_eq -=     M_aug_dh[3]

    # Map task-space wrenches to 4 wheel torques (mecanum O-configuration)
    lever = R / (l + h)
    M1_sw = 0.25 * (Mx_sw - My_sw - lever * Mpsi_sw)
    M2_sw = 0.25 * (Mx_sw + My_sw + lever * Mpsi_sw)
    M3_sw = 0.25 * (Mx_sw + My_sw - lever * Mpsi_sw)
    M4_sw = 0.25 * (Mx_sw - My_sw + lever * Mpsi_sw)

    M1_eq = 0.25 * (Mx_eq - My_eq - lever * M_psi_eq)
    M2_eq = 0.25 * (Mx_eq + My_eq + lever * M_psi_eq)
    M3_eq = 0.25 * (Mx_eq + My_eq - lever * M_psi_eq)
    M4_eq = 0.25 * (Mx_eq - My_eq + lever * M_psi_eq)

    Mi_sw = SVector(M1_sw, M2_sw, M3_sw, M4_sw)
    Mi_eq = SVector(M1_eq, M2_eq, M3_eq, M4_eq)

    # Smooth adaptive-gain dynamics (identical structure; driven by the new s)
    base_dK_x   = asmc.gamma_x   * (s_x   * ss_x)
    base_dK_y   = asmc.gamma_y   * (s_y   * ss_y)
    base_dK_psi = asmc.gamma_psi * (s_psi * ss_psi)
    dK_x   = base_dK_x   * smooth_bound(K_x,   asmc.K_max_x)   - 0.1 * (K_x   / asmc.K_max_x)^3 - asmc.sigma_x   * (K_x   - asmc.K_x0   * 0.95) * exp(asmc.decay_k*(1 - s_x^2  /(9 * asmc.eps_psi^2)))
    dK_y   = base_dK_y   * smooth_bound(K_y,   asmc.K_max_y)   - 0.3 * (K_y   / asmc.K_max_y)^3 - asmc.sigma_y   * (K_y   - asmc.K_y0   * 0.95) * exp(asmc.decay_k*(1 - s_y^2  /(9 * asmc.eps_psi^2)))
    dK_psi = base_dK_psi * smooth_bound(K_psi, asmc.K_max_psi) - 0.5 * (K_psi / asmc.K_max_psi)^3 - asmc.sigma_psi * (K_psi - asmc.K_psi0 * 0.95) * exp(asmc.decay_k*(1 - s_psi^2/(9 * asmc.eps_psi^2)))

    return Mi_sw, Mi_eq, SVector(dK_x, dK_y, dK_psi), delta_hat
end

# ---------------------------
# In-place dynamics RHS for ODEProblem  (39-D: 21 base + 12 LuGre bristles + 6 observer states)
# p = (params, asmc, chi, p1, p2, coupling, lugre, eso, ctrl)   # ← ctrl added (9th element)
#   ctrl = asmc_torques  (position tracking)  |  asmc_torques_vel  (velocity tracking)
# state: [1:21] base; [22:25]=zx, [26:29]=zy, [30:33]=zs;
#        observer [34:36]=(ζ_x,ζ_y,ζ_ψ) if :linear, else (v̂_x,v̂_y,v̂_ψ); [37:39]=δ̂ if :super_twisting
# ---------------------------
function dynamics_full_mf_asmc!(du, u, p, t)
    params, asmc, chi, p1, p2, coupling, lugre, eso, ctrl = p   # ← unpack ctrl

    Vx, Vy, psi_dot, psi = u[1], u[2], u[3], u[4]
    ti = SVector(u[5],  u[6],  u[7],  u[8])
    wi = SVector(u[9],  u[10], u[11], u[12])
    gi = SVector(u[13], u[14], u[15], u[16])
    zx = SVector(u[22], u[23], u[24], u[25])
    zy = SVector(u[26], u[27], u[28], u[29])
    zs = SVector(u[30], u[31], u[32], u[33])

    px, py = params.wc_x, params.wc_y
    R, Rd  = params.R, params.Ra
    delta  = params.delta
    aX, aY = params.aX, params.aY
    ms, m  = params.ms, params.m

    sdi = sin.(delta); cdi = cos.(delta); tdi = tan.(delta)
    ti_t  = sawtooth_approx.(ti)
    sti_t = sin.(ti_t);  cti_t = cos.(ti_t);  tti_t = tan.(ti_t)
    DYi   = Rd .* tdi .* tti_t

    Vpi_x = @. Vx - psi_dot * (py + DYi) - wi * R +
               gi * sdi * (Rd * cti_t - R) + DYi * gi * cdi * sti_t
    Vpi_y = @. Vy + psi_dot * px +
               gi * cdi * (R * cti_t - Rd)
    wzi   = @. psi_dot - gi * (-sti_t * cdi)

    Ni = params.N_per_roller

    # Per-wheel DYNAMIC LuGre: forces + bristle derivatives.
    fr = ntuple(4) do i
        lugre_dyn_rates(lugre, coupling, params.f_coulomb, Ni[i], chi,
                        wzi[i], Vpi_x[i], Vpi_y[i], zx[i], zy[i], zs[i])
    end
    Fx_i = SVector(fr[1][1], fr[2][1], fr[3][1], fr[4][1])
    Fy_i = SVector(fr[1][2], fr[2][2], fr[3][2], fr[4][2])
    Mz_i = SVector(fr[1][3], fr[2][3], fr[3][3], fr[4][3])
    dzx  = SVector(fr[1][4], fr[2][4], fr[3][4], fr[4][4])
    dzy  = SVector(fr[1][5], fr[2][5], fr[3][5], fr[4][5])
    dzs  = SVector(fr[1][6], fr[2][6], fr[3][6], fr[4][6])

    Mi_sw, Mi_eq, dK, _ = ctrl(u, t, params, asmc, eso)   # ← injected controller
    Mi_total = Mi_sw .+ Mi_eq
    max_tau  = params.Max_torque
    Mi_sat   = smooth_sat.(Mi_total, max_tau, 4)

    # ---- MIMO DOB auxiliary-state ODE  (derivation eq 12) ----
    l, h  = params.l, params.h
    lever = R / (l + h)
    M_x_applied   =  Mi_sat[1] + Mi_sat[2] + Mi_sat[3] + Mi_sat[4]
    M_y_applied   = -Mi_sat[1] + Mi_sat[2] + Mi_sat[3] - Mi_sat[4]
    M_psi_applied = (-Mi_sat[1] + Mi_sat[2] - Mi_sat[3] + Mi_sat[4]) / lever
    F_phys = SVector(M_x_applied / R, M_y_applied / R, M_psi_applied)
    v_body = SVector(Vx, Vy, psi_dot)
    en     = eso.enable ? 1.0 : 0.0
    a_nom  = params.M_aug_inv * F_phys

    Mz_rolleri = @. px * Fy_i - py * Fx_i + Mz_i
    RHS0 = sum(Fx_i)          + ms * psi_dot * Vy  + m * aX * psi_dot^2
    RHS1 = sum(Fy_i)          - ms * psi_dot * Vx  + m * aY * psi_dot^2
    RHS2 = sum(Mz_rolleri)    - m * psi_dot * (aX * Vx + aY * Vy)
    dv = params.M_inv * SVector(RHS0, RHS1, RHS2)

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
    du[17] = dK[1]; du[18] = dK[2]; du[19] = dK[3]
    du[20] = Vx * cos(psi) - Vy * sin(psi)
    du[21] = Vx * sin(psi) + Vy * cos(psi)
    @inbounds for i in 1:4
        du[21+i] = dzx[i]   # u[22:25]
        du[25+i] = dzy[i]   # u[26:29]
        du[29+i] = dzs[i]   # u[30:33]
    end
    if eso.kind == :super_twisting
        v_hat = SVector(u[34], u[35], u[36])
        d_hat = SVector(u[37], u[38], u[39])
        e_obs = v_body - v_hat
        k1    = SVector(eso.k1_x, eso.k1_y, eso.k1_psi)
        k2    = SVector(eso.k2_x, eso.k2_y, eso.k2_psi)
        phi1  = reg_sqrt.(e_obs, eso.eps_obs)
        phi2  = reg_sign.(e_obs, eso.eps_obs)
        dvhat = en * (a_nom + d_hat + k1 .* phi1)
        ddhat = en * (k2 .* phi2)
        du[34] = dvhat[1]; du[35] = dvhat[2]; du[36] = dvhat[3]
        du[37] = ddhat[1]; du[38] = ddhat[2]; du[39] = ddhat[3]
    else
        zeta_vec = SVector(u[34], u[35], u[36])
        Omega    = SVector(eso.omega_o_x, eso.omega_o_y, eso.omega_o_psi)
        dzeta    = en * (-Omega .* zeta_vec - Omega.^2 .* v_body - Omega .* a_nom)
        du[34] = dzeta[1]; du[35] = dzeta[2]; du[36] = dzeta[3]
        du[37] = 0.0;      du[38] = 0.0;      du[39] = 0.0
    end
    return nothing
end

# ---------------------------
# DIAGNOSTICS (post-processing on `sol`) — three quantities, no change to the RHS:
#   (1) t_dwell_i = (2π / N_rollers) / |ωᵢ|   — time one roller stays in contact.
#   (2) Roller-spin torque balance.  The spin ODE is
#         J_eff·γ̈ᵢ = Tdamp + Tfx + Tfy + Tmz ,   J_eff = J_roller · N_rollers.
#       Tnet := J_eff·γ̈ᵢ is the inertial torque (= net of the four).  The spin is
#       slaved (algebraic) where  resid = |Tnet| / (|Tdamp|+|Tfx|+|Tfy|+|Tmz|) ≪ 1,
#       i.e. the contact/damping terms cancel and inertia carries a negligible residual.
#       Single-roller inertial torque is Tnet / N_rollers (J_roller·γ̈) — the literal
#       "J_roller·γ̇-rate" — reported via Jeff so you can divide if you want it.
#   (3) Per-wheel control split:
#         M_eff = no-slip inverse-dynamics equivalent control,
#         M_dob = MIMO-DOB compensation  (recovered from δ̂:  ΔW = -diag(R,R,1)·M_aug·δ̂,
#                 mapped through the same 0.25 O-config allocation),
#         M_sw  = sliding switching wrench,   M_sat = applied (saturated) torque.
#       M_eff = Mᵢ_eq − M_dob.  Small M_dob/M_sw relative to M_eff ⇒ the no-slip
#       equivalent control already does the work, i.e. the slip model adds little here.
# `coupling/lugre/eso/ctrl/p1/p2` must match the run that produced `sol`.
# ---------------------------
function compute_roller_ctrl_diag(sol, params::PlatformParams, asmc::ASMCParams,
                                  eso::ESOParams, chi::Real, coupling::Symbol,
                                  lugre::LuGreParams, ctrl::Function, p1::Real, p2::Real)
    N  = length(sol.t)
    Nr = params.rollers_per_wheel
    Jeff = params.J_roller * Nr

    tdwell = zeros(4, N)
    Tdamp  = zeros(4, N); Tfx = zeros(4, N); Tfy = zeros(4, N); Tmz = zeros(4, N)
    Tnet   = zeros(4, N); resid = zeros(4, N)
    Meff   = zeros(4, N); Mdob = zeros(4, N); Msw = zeros(4, N); Msat = zeros(4, N)

    sdi = sin.(params.delta); cdi = cos.(params.delta); tdi = tan.(params.delta)
    px, py = params.wc_x, params.wc_y
    R, Rd  = params.R, params.Ra
    l, h   = params.l, params.h
    lever  = R / (l + h)
    w_floor = 1e-4    # avoid Inf when a wheel is ~stationary (no roller handoff there)

    @inbounds for k in 1:N
        t = sol.t[k]; u = sol.u[k]
        Vx, Vy, psi_dot, psi = u[1], u[2], u[3], u[4]
        ti = SVector(u[5],  u[6],  u[7],  u[8])
        wi = SVector(u[9],  u[10], u[11], u[12])
        gi = SVector(u[13], u[14], u[15], u[16])
        zx = SVector(u[22], u[23], u[24], u[25])
        zy = SVector(u[26], u[27], u[28], u[29])
        zs = SVector(u[30], u[31], u[32], u[33])

        ti_t  = sawtooth_approx.(ti)
        sti_t = sin.(ti_t); cti_t = cos.(ti_t); tti_t = tan.(ti_t)
        DYi   = Rd .* tdi .* tti_t

        Vpi_x = @. Vx - psi_dot * (py + DYi) - wi * R +
                   gi * sdi * (Rd * cti_t - R) + DYi * gi * cdi * sti_t
        Vpi_y = @. Vy + psi_dot * px + gi * cdi * (R * cti_t - Rd)
        wzi   = @. psi_dot - gi * (-sti_t * cdi)

        for i in 1:4
            fx, fy, mz, _, _, _ = lugre_dyn_rates(lugre, coupling, params.f_coulomb,
                                                  params.N_per_roller[i], chi,
                                                  wzi[i], Vpi_x[i], Vpi_y[i], zx[i], zy[i], zs[i])
            # roller-spin torque contributions (signs as they enter J_eff·γ̈ = Σ)
            A = sdi[i] * (R - Rd * cti_t[i]) - DYi[i] * sti_t[i] * cdi[i]
            B = cdi[i] * (Rd - R * cti_t[i])
            C = sti_t[i] * cdi[i]
            td = -p2 * gi[i]; tx = -fx * A; ty = -fy * B; tm = -mz * C
            net = td + tx + ty + tm                       # = J_eff·γ̈ᵢ
            Tdamp[i,k]=td; Tfx[i,k]=tx; Tfy[i,k]=ty; Tmz[i,k]=tm; Tnet[i,k]=net
            resid[i,k] = abs(net) / (abs(td)+abs(tx)+abs(ty)+abs(tm) + eps())
            tdwell[i,k] = (2π / Nr) / max(abs(wi[i]), w_floor)
        end

        Mi_sw, Mi_eq, _, dhat = ctrl(u, t, params, asmc, eso)
        Mdh = params.M_aug * dhat                          # SVector{3}
        dWx = -R * Mdh[1]; dWy = -R * Mdh[2]; dWp = -Mdh[3]
        md = SVector(0.25*(dWx - dWy - lever*dWp),
                     0.25*(dWx + dWy + lever*dWp),
                     0.25*(dWx + dWy - lever*dWp),
                     0.25*(dWx - dWy + lever*dWp))
        for i in 1:4
            Mdob[i,k] = md[i]
            Meff[i,k] = Mi_eq[i] - md[i]                   # no-slip inverse-dynamics part
            Msw[i,k]  = Mi_sw[i]
            Msat[i,k] = params.Max_torque * tanh((Mi_sw[i] + Mi_eq[i]) / params.Max_torque)
        end
    end
    return (; tdwell, Tdamp, Tfx, Tfy, Tmz, Tnet, resid, Meff, Mdob, Msw, Msat, Jeff, Nr)
end


# ---------------------------
# Initial-condition construction (39-D state)
# ---------------------------
function build_initial_state(params::PlatformParams, asmc::ASMCParams)
    u0 = zeros(39)        # 21 base + 12 LuGre bristles + 6 observer states
    ref = Profiles.active_ref()
    u0[1] = 0.0            # Vx     — all profiles start from rest by design
    u0[2] = 0.0            # Vy
    u0[3] = 0.0            # psi_dot (profiles guarantee Wz(0)=0)
    u0[4] = ref.psi(0.0)   # ψ₀ = reference heading at t=0 ⇒ no initial yaw transient
    u0[5:8]  .= 0.1        # θᵢ
    u0[9:12] .= 0.0        # ωᵢ
    u0[13:16].= 0.0        # γᵢ
    u0[17] = 1.2 * asmc.K_x0
    u0[18] = 1.2 * asmc.K_y0
    u0[19] = 1.2 * asmc.K_psi0
    # u0[20:21] (world position Xo, Yo) stay 0 — VelRef profiles carry no position
    # reference, and PosRef profiles (ellipse) start their path AT the origin.
    # carry no position reference; the world path is whatever ∫R(ψ)v dt gives.
    return u0
end


# Per-group absolute tolerances — UNCHANGED from the previous cell
const _AT = (
    body_vel = 1.0e-8,   psi      = 1.0e-7,
    wtheta   = 1.0e-8,   womega   = 1.0e-7,
    gamma    = 1.0e-7,   gains    = 1.0e-7,
    worldpos = 1.0e-7,   bristle  = 1.0e-10,
    bristle_rot = 1.0e-7, observer = 1.0e-7,  dist_est = 1.0e-7,
)
const ABSTOL = vcat(
    fill(_AT.body_vel, 3), fill(_AT.psi, 1),
    fill(_AT.wtheta, 4),   fill(_AT.womega, 4),
    fill(_AT.gamma, 4),    fill(_AT.gains, 3),
    fill(_AT.worldpos, 2), fill(_AT.bristle, 8),
    fill(_AT.bristle_rot, 4), fill(_AT.observer, 3), fill(_AT.dist_est, 3),
)   # 3+1+4+4+4+3+2+8+4+3+3 = 39
  # counts: 3+1+4+4+4+3+2+12+3+3 = 39

# ---------------------------
# Single-run driver — T and tstops come from the active reference (either kind)
# ---------------------------
function run_one_chi(chi::Real, params::PlatformParams, asmc::ASMCParams;
                     friction_model::Symbol = friction_model, lugre::LuGreParams = lugre,
                     eso::ESOParams = ESOParams(),
                     T::Real = Profiles.active_ref().T_total,
                     friction_case::Int = friction_case,
                     dt_save::Real = 5e-4,                 # 2 kHz master resolution
                     reltol::Real = 1e-8, abstol::Union{Real,AbstractVector} = ABSTOL,
                     solver = TRBDF2())
    p1, p2 = friction_case == 1 ? (params.p1_case1, params.p2_case1) :
                                   (params.p1_case2, params.p2_case2)
    u0 = build_initial_state(params, asmc)
    t_eval = collect(range(0.0, T; length = round(Int, T / dt_save) + 1))
    ctrl = is_velref() ? asmc_torques_vel : asmc_torques   # reference kind decides
    p = (params, asmc, chi, p1, p2, coupling_of(friction_model), lugre, eso, ctrl)

    prob = ODEProblem(dynamics_full_mf_asmc!, u0, (0.0, T), p)

    prog = ProgressMeter.Progress(100; desc=@sprintf("%-16s chi=%.4f ", String(friction_model), chi))
    last_pct = Ref(0)
    function update_prog!(integrator)
        pct = clamp(floor(Int, 100 * integrator.t / T), 0, 100)
        if pct > last_pct[]
            ProgressMeter.update!(prog, pct); last_pct[] = pct
        end
        return nothing
    end
    pbar_cb = PeriodicCallback(update_prog!, T/100; initial_affect = false)

    sol = solve(prob, solver;
                reltol = reltol, abstol = abstol, saveat = t_eval,
                tstops = Profiles.active_ref().tstops,        # profile phase boundaries
                callback = pbar_cb, dtmax = 0.001, maxiters = 10^7)
    ProgressMeter.finish!(prog)
    return sol
end

# --- [skipped: sweep/plot/reload cell 27] ---

# --- [skipped: sweep/plot/reload cell 29] ---

# --- [skipped: sweep/plot/reload cell 31] ---

# --- [skipped: sweep/plot/reload cell 33] ---

