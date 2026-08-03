# =============================================================================
# hybrid_ctrl_v2/controllers_v2.jl — ControllerV2Mod (Stage 3 MPC extension +
# ASMC v2 + PID v2 + MPC v2)
# =============================================================================
# `hybrid_ctrl/controllers.jl` is NEVER edited (brief §11 preservation
# constraint, reaffirmed by every subsequent brief in this file). Adding
# fields to `MPCController`/`ASMCController`/`PIDController` is structurally
# impossible without editing that file (Julia struct fields are fixed at
# definition), so this module defines NEW structs (`MPCControllerV2`,
# `ASMCControllerV2`, `PIDControllerV2`) and extends the EXISTING generic
# functions (`ControllerMod.mpc_wrench!`/`asmc_wrench!`/`pid_wrench!`) with
# new methods dispatching on the new types -- standard Julia multiple-
# dispatch extension, zero edits to the original file.
#
# `MPCControllerV2`/its `mpc_wrench!` method were created for an earlier
# Stage-3 `P_terminal` brief and are now further extended below (instructions/
# mpc-v2-bryson-weights-terminal-cost.md) -- moved to the END of this file
# (new "MPC v2" section) since that brief's `bryson_R`/`u_eq_horizon` need
# `PhysicalLimits`/`capability_wrench`, defined further down in the ASMC v2
# section.
#
# Must be `include`d AFTER tune_controller.jl (so `Main.ControllerMod`,
# `Main.PlantMod`, `Main.Profiles`, `Main.TOL` already exist) -- see
# tune_controller_v2.jl.
# =============================================================================
module ControllerV2Mod

using StaticArrays
using LinearAlgebra
using SparseArrays
using OSQP
using MatrixEquations

export MPCControllerV2, PhysicalLimits, capability_wrench, kmax_schedule,
       lambda_schedule, decay_parameters, ASMCControllerV2,
       imc_gains, imax_from_measured, vcmd_limits, pose_outer_loop_v2, PIDControllerV2,
       bryson_Q_pose, bryson_R, u_eq_horizon, terminal_cost, DARECache

# =============================================================================
# ASMC v2 — physically-derived gain bounds (instructions/asmc-v2-physical-
# gain-bounds.md). `hybrid_ctrl/controllers.jl` is NEVER edited: ASMCControllerV2
# is a NEW struct and `Main.ControllerMod.asmc_wrench!` gets a NEW method for
# it, exactly the MPCControllerV2/mpc_wrench! precedent above.
# =============================================================================

# -----------------------------------------------------------------------------
# PhysicalLimits — friction-circle constants, traced to
# docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex (case-1 params,
# read from `params` where possible rather than hardcoded so they track any
# future base.toml change; the .tex's numeric equations are cited inline).
# -----------------------------------------------------------------------------
"""
    PhysicalLimits

Fields and provenance (docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex):
    kappa       = sqrt(2)*p2/(R-Ra)^2                    (E54) -- geometric/viscous, mu-independent
    mu_N3       = mu * params.N_per_roller[3]            (E54 sec 3.4) -- rear-left binding wheel,
                  READ from params (not hardcoded 34.8) -- verified to reproduce 34.8 at mu=0.5
    h, l, R, Ra                                          (Table 1)
    lever       = R/(l+h)                                mixer allocation
    m_tilde     = params.M_aug[1,1]                       (E52) READ from params, not hardcoded 45.0
    I_psi       = params.M_aug[3,3]                       READ from params, not the .tex's I_psi/I_s~1.31 ratio
    a_cap       = (2.93, 2.86, 9.62) .* (mu/0.5)          (E53) friction-circle intercepts at v=0,
                  F_perp=0; the .tex's reference values are AT mu=0.5, scaled linearly here since a
                  friction-circle limit is proportional to mu*N with everything else fixed
    alloc_ratio = capability_wrench(this) normalized      per-axis wrench capability, ~1:1:8.6
    safety_margin = 0.9                                   (E56) factor of safety on the friction
                  circle -- the SINGLE design margin in the v2 controllers (brief §6 update),
                  applied to mu_N3 wherever a controller checks against the AVAILABLE circle
                  (kmax_schedule's F_par_avail here; a future PID v2 gate would consume the
                  same `lim.safety_margin` for its own vcmd_limits). NOT applied to a_cap/
                  capability_wrench/K_max_base -- those stay the unmargined "free circle, v=0"
                  theoretical values per (E53); the .tex's own E56 note is that s shrinks the
                  AVAILABLE headroom (F_perp caps and F_parallel headroom together), which is
                  exactly kmax_schedule's per-tick operating-point gate, not the rest-state intercepts.

NOTE a_cap is deliberately NOT the 1.0 m/s^2 out-of-plane-load-transfer operating cap of .tex
section 3.5 -- that is a dataset-fidelity budget, not a hardware limit (brief §6).

EXTENDED for the PID v2 brief (instructions/pid-v2-imc-cascade-two-variants.md
§6/§7.1) -- added fields, existing ones/constructor signature unchanged:
    m_eff = (R*m_tilde, R*m_tilde, I_psi)                  ~ (2.25, 2.25, 5.89)
    d_eff = (R*4p1/R^2, R*(4p1/R^2+8p2/(R-Ra)^2),
             4p1(l+h)^2/R^2+8p2h^2/(R-Ra)^2)                ~ (8.8, 19.8, 38) -- the SAME
            D_eq diagonal already used in ControllerMod.asmc_wrench!'s Mx_eq/My_eq/M_psi_eq
            drag terms (.tex §2), R-scaled/I_psi-scaled the same way m_eff is
    tau_open = m_eff ./ d_eff                               ~ (0.256, 0.114, 0.155) s

Also carries the BARE-mass quantities (m, ms, Is, aX, aY) straight from
`params`, needed by `vcmd_limits`'s E55 demand vector (which -- unlike
`kmax_schedule`'s `W_ff` approximation -- must use the bare mass exactly, not
the augmented m_tilde/I_psi; E52's bare-vs-augmented caution).
"""
struct PhysicalLimits
    kappa::Float64
    mu_N3::Float64
    h::Float64
    l::Float64
    R::Float64
    Ra::Float64
    lever::Float64
    m_tilde::Float64
    I_psi::Float64
    a_cap::SVector{3,Float64}
    alloc_ratio::SVector{3,Float64}
    safety_margin::Float64
    m_eff::SVector{3,Float64}
    d_eff::SVector{3,Float64}
    tau_open::SVector{3,Float64}
    m::Float64
    ms::Float64
    Is::Float64
    aX::Float64
    aY::Float64
end

function PhysicalLimits(params; mu::Float64=0.5, safety_margin::Float64=0.9)
    R, Ra, h, l = params.R, params.Ra, params.h, params.l
    p1 = params.p1_case1
    p2 = params.p2_case1   # friction_case=1 convention (matches this codebase's default elsewhere)
    kappa = sqrt(2) * p2 / (R - Ra)^2
    mu_N3 = mu * params.N_per_roller[3]
    lever = R / (l + h)
    m_tilde = params.M_aug[1, 1]
    I_psi   = params.M_aug[3, 3]
    a_cap = SVector(2.93, 2.86, 9.62) .* (mu / 0.5)

    d_x_raw   = 4*p1/R^2
    d_y_raw   = 4*p1/R^2 + 8*p2/(R-Ra)^2
    d_psi_raw = 4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-Ra)^2
    m_eff = SVector(R*m_tilde, R*m_tilde, I_psi)
    d_eff = SVector(R*d_x_raw, R*d_y_raw, d_psi_raw)
    tau_open = m_eff ./ d_eff

    Mx_max = R * m_tilde * a_cap[1]
    My_max = R * m_tilde * a_cap[2]
    Mpsi_max = I_psi * a_cap[3]
    cap = SVector(Mx_max, My_max, Mpsi_max)
    alloc_ratio = cap ./ cap[1]

    return PhysicalLimits(kappa, mu_N3, h, l, R, Ra, lever, m_tilde, I_psi, a_cap, alloc_ratio,
                          safety_margin, m_eff, d_eff, tau_open,
                          params.m, params.ms, params.Is, params.aX, params.aY)
end

"""
    capability_wrench(lim::PhysicalLimits) -> SVector{3,Float64}

Per-axis task-space wrench capability at a free circle (v=0, zero feedforward),
in the controller's own R/I_psi-scaled units. Expected ~(6.59, 6.43, 56.7) N*m.
"""
capability_wrench(lim::PhysicalLimits) =
    SVector(lim.R * lim.m_tilde * lim.a_cap[1],
            lim.R * lim.m_tilde * lim.a_cap[2],
            lim.I_psi * lim.a_cap[3])

# =============================================================================
# PID v2 — IMC cascade reparameterization (instructions/pid-v2-imc-cascade-
# two-variants.md). Reuses `PhysicalLimits`/`_allocation_constant` from the
# ASMC v2 section above unchanged.
# =============================================================================

"""
    imc_gains(lim::PhysicalLimits, lam_inner::SVector{3,Float64}, N::Real) -> (Kp, Ki, Kp_pos)

IMC-PI pole cancellation, per axis, on the first-order plant
`m_eff*v_dot + d_eff*v = W`:

    Kp = m_eff / lam_inner        Ki = d_eff / lam_inner        (Kd = 0, a RESULT)

The controller zero -Ki/Kp = -d_eff/m_eff = -1/tau_open lands exactly on the
plant pole, so `Ki/Kp == 1/lim.tau_open` identically for ANY lam_inner --
this is the pole-cancellation identity the brief's §10 checks.

Outer loop (integrator plant p_dot=v): `Kp_pos = 1/(N*lam_inner)`, giving
`omega_n = 1/(lam_inner*sqrt(N))`, `zeta = sqrt(N)/2` -- N=4 is exactly
critical damping.
"""
function imc_gains(lim::PhysicalLimits, lam_inner::SVector{3,Float64}, N::Real)
    Kp = lim.m_eff ./ lam_inner
    Ki = lim.d_eff ./ lam_inner
    Kp_pos = 1.0 ./ (N .* lam_inner)
    return Kp, Ki, Kp_pos
end

"""
    imax_from_measured(lim::PhysicalLimits, Ki::SVector{3,Float64}, feedforward::Bool) -> SVector{3,Float64}

Anti-windup bound sized from the MEASURED p95 of the per-wheel wrench the
integral must supply (brief §6):

    PID-FB : p95(|Msat|)        = 5.279 N*m/wheel  (integral supplies drag + Coriolis + disturbance)
    PID-CT : p95(|Msat - M_eq|) = 2.389 N*m/wheel  (feedforward supplies the model; integral supplies the residual only)

NOT re-measured in this pass -- using the brief's own stated design values
directly (same judgment call as the ASMC brief's `K_floor`), since these are
cited alongside an independent analytical cross-check (see below) that this
implementation reproduces to <0.2%.

Converted through the SAME capability-proportional allocation as `K_floor`
(`_allocation_constant`): `per_wheel = alloc_const*c`, task-space wrench
budget `= c .* alloc_ratio`. `I_max` is then set so `Ki*I_max` equals that
budget (the integral's actual contribution to the wrench is `Ki*I_pid`, so
the bound that matters is on `Ki*I_max`, not `I_max` alone):

    I_max = (c .* lim.alloc_ratio) ./ Ki

CROSS-CHECK (brief, reproduced by `_tmp/pid_v2_validation.jl`): measured-p95
route gives `c_FB = 5.279/alloc_const ~ 6.77` task-space, against an
independent analytical steady-demand estimate `d_eff_x*v_max + disturbance =
8.8*0.6+1.5 = 6.78` -- agreement to 0.2% from two unrelated routes.

EMPIRICAL ADJUSTMENT (post-implementation, per-user direction): the PID-CT
budget as originally stated (`2.389` N*m/wheel) saturated 25.4% of ticks on
an EASY ellipse trajectory in `_tmp/pid_v2_validation.jl` -- well above the
brief's "should be rare" expectation, and on the GENTLE trajectory rather
than the stress one, which is the wrong direction for a genuine physical
bound. Doubled to `4.778` N*m/wheel here; re-evaluated and confirmed
`frac(I_sat)` drops to 0.0 on that same trajectory (see chat response). This
is a deliberate widening of the anti-windup BOUND, not a retuned gain (`Kp`/
`Ki` are untouched) -- `I_max` remains a bound on rare saturation, not a
performance knob, so this still needs re-checking once real tuning happens.
"""
function imax_from_measured(lim::PhysicalLimits, Ki::SVector{3,Float64}, feedforward::Bool)
    p95_per_wheel = feedforward ? 2.389*2 : 5.279   # CT doubled -- see docstring EMPIRICAL ADJUSTMENT note
    c = p95_per_wheel / _allocation_constant(lim)
    wrench_budget = c .* lim.alloc_ratio
    return wrench_budget ./ Ki
end

"""
    kmax_schedule(lim, V_y, psidot, W_ff) -> SVector{3,Float64}

State-dependent gain ceiling from the binding-wheel gate (E54).

STEP 2 UPDATE (instructions/asmc-v2-physical-gain-bounds.md §6, K_max
calculation revision): `F_par_avail = sqrt(max(0, (s*mu_N3)^2 - F_perp3^2))`
with `s = lim.safety_margin` (0.9 by default) -- a factor of safety on the
AVAILABLE friction circle per (E56), NOT on velocities or on K_max_base/
capability_wrench separately (those stay the unmargined "free circle, v=0"
values -- see the `PhysicalLimits` docstring). `s` is the SINGLE design
margin in the v2 controllers: a future PID v2 `vcmd_limits` gate would consume
the SAME `lim.safety_margin`, just applying it to a demand vector via a
scalar rather than distributing a ceiling across axes via `alloc_ratio`.

NOTE this shifts the brief's §10 binding-wheel-table numbers (both the
absolute `F_par,avail` column AND the %-of-circle column, since the margin
only factors out cleanly at `F_perp3=0`): row 1 (V_y=psidot=0) becomes
`0.9*34.8=31.32 N` / 90% exactly, but the other rows do NOT simply scale by
0.9 (e.g. row 2, V_y=0.4: `sqrt(0.9^2*34.8^2-15.56^2)=27.18 N` = 78.1% of the
RAW circle, not `0.9*89%=80.1%`) -- `_tmp/asmc_v2_validation.jl` checks the
UPDATED formula directly rather than the brief's original (pre-margin) table.

Step 3 (subtract the feedforward's own draw on the binding wheel): `W_ff =
(Mx_eq, My_eq, Mpsi_eq)` is in the controller's own R/I_psi-scaled units;
`Mx_eq/R`, `My_eq/R` recover body-frame FORCE (undoing the R*(...) the
equivalent-control law applies), `Mpsi_eq` is already a body-frame MOMENT
matching (E54)'s `b_Omega` directly. Uses the SAME wheel-3 allocator row
(0.354, 0.354, -0.918) as (E54).

DOCUMENTED APPROXIMATION: (E54)'s b-vector uses the BARE mass (m_s, I_s) for
the contact-circle constraint (E52's bare-vs-augmented caution), while `W_ff`
is computed against the AUGMENTED mass (m_tilde, I_psi) inside `asmc_wrench!`.
The augmented/bare ratios are O(1) (~1.26 translational, ~1.31 yaw per the
.tex), so this is a bounded, second-order approximation on a gating term whose
DOMINANT effect -- F_perp3's velocity-slaved circle consumption (steps 1-2) --
is exact (verified against the .tex's own table, not approximated).
"""
function kmax_schedule(lim::PhysicalLimits, V_y::Real, psidot::Real, W_ff::SVector{3,Float64})
    F_perp3 = lim.kappa * (V_y - lim.h * psidot)
    mu_N3_margined = lim.safety_margin * lim.mu_N3
    F_par_avail = sqrt(max(0.0, mu_N3_margined^2 - F_perp3^2))

    Fx_ff = W_ff[1] / lim.R
    Fy_ff = W_ff[2] / lim.R
    Mz_ff = W_ff[3]
    F_par3_ff = 0.354 * (Fx_ff + Fy_ff) - 0.918 * Mz_ff
    F_avail = max(0.0, F_par_avail - abs(F_par3_ff))

    # frac is measured against the RAW (unmargined) mu_N3, so it correctly
    # tops out at `safety_margin` (0.9), not 1.0, at the free circle (F_perp3=0,
    # W_ff=0) -- capability_wrench/K_max_base stay the unmargined reference;
    # the margin is entirely expressed through this fraction.
    frac = lim.mu_N3 > 0 ? F_avail / lim.mu_N3 : 0.0
    return capability_wrench(lim) .* frac
end

"""
    vcmd_limits(lim, xhat, dt, vcmd_prev, vcmd_raw) -> (V_cmd, gamma, guard_hit)

Feasibility-limited velocity command via the SAME binding-wheel gate (E54) as
`kmax_schedule`, but as a SINGLE COMBINED constraint over velocity AND
acceleration together (brief §6, "the only structural change in this brief").

PER-TICK ALGORITHM:
  1. `a_req = (vcmd_raw - vcmd_prev)/dt` -- the implied acceleration demand.
  2. Demand vector `b(V_hat, a_req)` (E55), BARE mass (m_s, m, I_s) -- unlike
     `kmax_schedule`'s `W_ff` (already R/I_psi-scaled, hence its documented
     bare-vs-augmented approximation), `vcmd_limits` computes `b` directly
     from `xhat`/`a_req` in physical (N, N*m) units, so E55 applies EXACTLY,
     no approximation.
  3. `F_perp3 = kappa*(V_y - h*psidot)` (velocity only, fixed this tick);
     `F_par3` is AFFINE in `a_req` -- split `b` into its velocity-only part
     (`a=0`) and its full value at `a=a_req`, giving `F_par3(gamma) =
     F_par3_0 + gamma*dF` for `a_req` scaled by `gamma in [0,1]`.
  4. `budget = (s*mu_N3)^2 - F_perp3^2`, `s = lim.safety_margin` (0.9, the
     SAME single margin `kmax_schedule` uses). `budget < 0` (F_perp3 alone
     exceeds the margined circle): `gamma=0`, hold `V_cmd=vcmd_prev`,
     `guard_hit=true` -- a reportable finding (brief §9), not silenced.
  5. Otherwise solve `|F_par3(gamma)| = sqrt(budget)` for the LARGEST
     `gamma in [0,1]` (F_par3 is monotone in gamma since it's affine, so
     there's one binding root in the direction gamma is heading); if
     `|F_par3_0| > sqrt(budget)` already (can't even hold current course),
     also `guard_hit=true`, `gamma=0`.
  6. `V_cmd = vcmd_prev + gamma*a_req*dt`.

WHY A SINGLE SCALAR gamma: scaling all three components together PRESERVES
THE DIRECTION of the demanded acceleration; per-axis clipping would distort
it and push the platform off the path.

NORMALIZER CAUTION: this does NOT use `V_y,crit=0.63` anywhere -- that is the
COMBINED steady-strafe point (F_par forced equal to F_perp), not a velocity
scale for this gate; using it would double-count.
"""
function vcmd_limits(lim::PhysicalLimits, xhat, dt::Real,
                     vcmd_prev::SVector{3,Float64}, vcmd_raw::SVector{3,Float64})
    Vx, Vy, psidot = xhat[1], xhat[2], xhat[3]
    a_req = (vcmd_raw .- vcmd_prev) ./ dt

    m, ms, Is, aX, aY = lim.m, lim.ms, lim.Is, lim.aX, lim.aY

    # E55 demand vector, split into velocity-only (a=0) and full (a=a_req)
    # evaluations -- F_par3 is affine in `a`, so this gives the two points
    # needed to solve for gamma without ever forming the Jacobian explicitly.
    _b(ax, ay, alpha) = (
        ms*ax - m*aY*alpha - ms*psidot*Vy - m*aX*psidot^2,
        ms*ay + m*aX*alpha + ms*psidot*Vx - m*aY*psidot^2 - 110.0*Vy,
        -m*aY*ax + m*aX*ay + Is*alpha + m*psidot*(aX*Vx + aY*Vy) - 2.20*psidot,
    )
    _Fpar3(b) = 0.354*(b[1] + b[2]) - 0.918*b[3]

    b0 = _b(0.0, 0.0, 0.0)
    b1 = _b(a_req[1], a_req[2], a_req[3])
    F_par3_0 = _Fpar3(b0)
    F_par3_1 = _Fpar3(b1)
    dF = F_par3_1 - F_par3_0

    F_perp3 = lim.kappa * (Vy - lim.h * psidot)
    mu_N3_m = lim.safety_margin * lim.mu_N3
    budget = mu_N3_m^2 - F_perp3^2

    if budget < 0.0 || abs(F_par3_0) > sqrt(max(budget, 0.0))
        # Degenerate: the platform is already outside its (margined) circle,
        # or even holding steady (a=0) would violate it. Reportable finding
        # (brief §9), not silenced.
        return vcmd_prev, 0.0, true
    end

    target = sqrt(budget)
    gamma = if dF == 0.0
        1.0   # F_par3 independent of a_req; already within budget at a=0 (checked above)
    elseif dF > 0.0
        (target - F_par3_0) / dF
    else
        (-target - F_par3_0) / dF
    end
    gamma = clamp(gamma, 0.0, 1.0)

    V_cmd = vcmd_prev .+ gamma .* a_req .* dt
    return V_cmd, gamma, false
end
"""
    lambda_schedule(e, edot, lam_max, v_max) -> (lam, lam_dot)

Saturation-aware sliding-surface slope: `lam(e) = min(lam_max, v_max/|e|)`.
Branch test uses `lam_max*|e| <= v_max` (a multiplication, not a division) so
no division-by-zero guard is needed -- at e=0 the lam_max branch is always
selected (lam_max*0 <= v_max trivially), giving `lam_dot=0` with no division
ever evaluated. Satisfies `lam*|e| <= v_max` everywhere (equality in the
hyperbolic branch, strict inequality in the capped branch).
"""
function lambda_schedule(e::Real, edot::Real, lam_max::Real, v_max::Real)
    if lam_max * abs(e) <= v_max
        return lam_max, 0.0
    else
        ae = abs(e)
        lam = v_max / ae
        lam_dot = -v_max * edot * sign(e) / e^2
        return lam, lam_dot
    end
end

"""
    decay_parameters(K_max_base, K_floor, tau_relax, tau_ceiling, decay_k) -> NamedTuple

sigma is IDENTICAL across axes (no range-dependence in its own formula);
cubic_coeff is RANGE-PROPORTIONAL (K_max_base-K_floor)/tau_ceiling, giving
equal decay times per axis despite the 8.5x range difference (brief §6).
"""
function decay_parameters(K_max_base::SVector{3,Float64}, K_floor::SVector{3,Float64},
                          tau_relax::Real, tau_ceiling::Real, decay_k::Real)
    sigma = 1.0 / (tau_relax * exp(decay_k))
    decay_sigma = SVector(sigma, sigma, sigma)
    cubic_coeff = (K_max_base .- K_floor) ./ tau_ceiling
    return (decay_sigma=decay_sigma, cubic_coeff=cubic_coeff)
end

"""
    _smooth_bound_v2(K, K_max) -> Float64

Anti-windup growth gate with a FRACTIONAL knee (0.98*K_max) instead of v1's
absolute -2.0 offset (`ControllerMod._smooth_bound`, never edited -- this is
a new function). Puts the knee at 98% of ceiling on EVERY axis regardless of
K_max's physical value (brief §6: v1's absolute -2.0 put the knee at 70% on
x but 96.5% on psi once K_max is physically derived).
"""
_smooth_bound_v2(K::Real, K_max::Real) = 0.5 - 0.5 * tanh(K - 0.98 * K_max)

"""
    _allocation_constant(lim) -> Float64

Shared allocation constant `0.25*(alloc_ratio_x + alloc_ratio_y + lever*alloc_ratio_psi)`
mapping a task-space wrench on the capability-proportional ratio to the
per-wheel torque it costs (`per_wheel = alloc_const * c` for `W = c*alloc_ratio`).
Used by BOTH `_default_k_floor` (ASMC v2 brief) and `imax_from_measured`
(PID v2 brief, instructions/pid-v2-imc-cascade-two-variants.md §6) -- the two
controllers differ only in which measured per-wheel statistic they convert
through this SAME allocation.
"""
_allocation_constant(lim::PhysicalLimits) =
    0.25 * (lim.alloc_ratio[1] + lim.alloc_ratio[2] + lim.lever * lim.alloc_ratio[3])

"""
    _default_k_floor(lim) -> SVector{3,Float64}

K_floor design derivation (brief §7.1): measured switching-demand statistics
on the stored simulation dataset give a per-sample |Msat-Meq| median (p50)
~0.294 N*m/wheel; the working budget is set at 1.2 N*m/wheel (NOT re-measured
in this pass -- using the brief's own stated design values directly, unlike
`tau_slip` in the sensor brief which was freshly measured here). Converted
through the allocation `0.25*(K_x+K_y+lever*K_psi) = alloc_const*c`, so
`c = 1.2/alloc_const`, and K_floor = c .* alloc_ratio (same ratio as
K_max_base, matching the brief's stated ~4.3x adaptation range).
"""
function _default_k_floor(lim::PhysicalLimits)
    c = 1.2 / _allocation_constant(lim)
    return c .* lim.alloc_ratio
end

# -----------------------------------------------------------------------------
# ASMCControllerV2
# -----------------------------------------------------------------------------
"""
    ASMCControllerV2

Same `(bus, xhat, ref, params, dt; mode)` contract as `ASMCController`, but
gain bounds are functions of the estimated state and `lim` rather than
constants (brief §1). Field-declaration order matters here: `@kwdef`
evaluates defaults in order, so `tau_relax`/`tau_ceiling`/`decay_k` are
declared before the derived `K_max_base`/`K_floor`/`decay_sigma`/
`cubic_coeff` that depend on them.

`eps`/`eps_psi` are SPECIFIED (pinned at v1's existing values), not freshly
re-measured from noise data this pass -- a judgment call, disclosed here
rather than silently assumed (unlike `tau_slip` in the sensor brief, which
WAS freshly measured).

`start_at_floor` (brief §8 step 8): `bus.K` resets to zeros in the
never-edited `bus.jl`. DELIBERATE CHOICE made here: default `false` (keep the
zero start), because the brief itself notes the startup transient is "one of
the few places gamma is observable" -- and gamma_x/y/psi are exactly what
`ASMC_SPACE_V2` tunes. Set `true` to instead inject `0.95*K_floor` on the
first tick and remove the transient (a legitimate choice for a non-tuning
deployment run, just not the default here).
"""
Base.@kwdef mutable struct ASMCControllerV2
    lim::PhysicalLimits

    # Tuned (ASMC_SPACE_V2)
    lam_x_max::Float64   = 1.5
    lam_y_max::Float64   = 2.5
    lam_psi_max::Float64 = 5.0
    gamma_x::Float64     = 8.0
    gamma_y::Float64     = 15.0
    gamma_psi::Float64   = 25.0

    # Specified (design constants / measured-noise proxies, not searched)
    eps::Float64         = 0.0175
    eps_psi::Float64      = 0.08
    tau_relax::Float64    = 2.0    # sigma-leak time constant (design choice, must be reported)
    tau_ceiling::Float64  = 0.5    # cubic-barrier time constant (design choice, must be reported)
    decay_k::Float64      = 0.25
    v_max_axis::SVector{3,Float64} = SVector(0.63, 0.63, 3.8)  # platform's own V_y,crit / psidot_max caps (.tex §3.2-3.3)
    rate_hz::Float64      = 1000.0

    # Derived-at-construction (depends on lim + the specified fields above)
    K_max_base::SVector{3,Float64}  = capability_wrench(lim)
    K_floor::SVector{3,Float64}     = _default_k_floor(lim)
    decay_sigma::SVector{3,Float64} = decay_parameters(capability_wrench(lim), _default_k_floor(lim), tau_relax, tau_ceiling, decay_k).decay_sigma
    cubic_coeff::SVector{3,Float64} = decay_parameters(capability_wrench(lim), _default_k_floor(lim), tau_relax, tau_ceiling, decay_k).cubic_coeff

    # Target equilibrium (leakage set-point, NOT the initial condition despite
    # the K_x0 name -- brief §7.1). Cubic depresses the actual resting K below
    # 0.95*K_x0; verify numerically (see _tmp/asmc_v2_validation.jl) and adjust
    # if the depression needs compensating.
    K_x0::Float64   = K_floor[1] / 0.95
    K_y0::Float64   = K_floor[2] / 0.95
    K_psi0::Float64 = K_floor[3] / 0.95

    # Schedule config
    use_scheduled_kmax::Bool = true
    kmax_lpf_tau::Float64    = 0.05   # low-pass tau on the estimated V_y/psidot schedule inputs (brief §9)
    start_at_floor::Bool     = false  # K initial-condition choice -- see struct docstring

    use_dhat::Bool = false

    # Internal state
    Vy_filt::MVector{1,Float64}     = MVector(0.0)
    psidot_filt::MVector{1,Float64} = MVector(0.0)
    initialized::Bool = false

    # Optional per-tick K/K_max_sched instrumentation (brief §10 "max(K) logged
    # per axis, scheduled ceiling reached at least occasionally"). `bus.K` is
    # not in `SchedulerMod`'s probe-log schema and that file is never edited,
    # so this self-contained logging hook lives on the controller instance
    # instead -- zero cost when `log_K=false` (the default; hot-path safe).
    log_K::Bool = false
    K_log::Vector{SVector{3,Float64}} = SVector{3,Float64}[]
    K_max_sched_log::Vector{SVector{3,Float64}} = SVector{3,Float64}[]
end

"""
    Main.ControllerMod.asmc_wrench!(bus, xhat, ref, params, asmc::ASMCControllerV2, dt; mode=:pose)

Same control law as `ControllerMod.asmc_wrench!` (surfaces, equivalent
control, switching wrench, gain-adaptation ODE all structurally unchanged),
with the v2 replacements: `lambda_schedule` instead of `_get_dynamic_lambda`;
a per-tick `kmax_schedule` ceiling (low-pass filtered inputs) under a LAZY
clamp `K_max_eff = max(K, K_max_sched)` instead of the constant `K_max_*`;
`_smooth_bound_v2` (fractional knee) instead of `_smooth_bound`; and the
sigma/cubic decay terms using `asmc.decay_sigma`/`asmc.cubic_coeff` against
`K_max_eff`, with the sigma exponential using EACH AXIS'S OWN boundary layer
(`eps` for x/y, `eps_psi` for psi -- v1 uses `eps_psi` in all three, brief's
required fix).

STABILITY NOTE (brief §9): the lazy clamp makes this a projection-based
adaptive law with a TIME-VARYING bound (Ioannou & Sun parameter projection),
not the constant-bound case the standard adaptive-SMC Lyapunov argument
assumes -- flagged here rather than inheriting the fixed-bound proof.
"""
function Main.ControllerMod.asmc_wrench!(bus, xhat, ref, params, asmc::ASMCControllerV2, dt;
                                         mode::Symbol=:pose)
    if !asmc.initialized
        if asmc.start_at_floor
            bus.K = 0.95 .* asmc.K_floor
        end
        asmc.initialized = true
    end

    m, ms, Is, J1 = params.m, params.ms, params.Is, params.J_wheel
    aX, aY = params.aX, params.aY
    p1, p2 = params.p1_case1, params.p2_case1
    l, h, R = params.l, params.h, params.R
    I_psi_local = asmc.lim.I_psi

    Vx, Vy, psi_dot, psi = xhat[1], xhat[2], xhat[3], xhat[4]
    K_x, K_y, K_psi = bus.K[1], bus.K[2], bus.K[3]

    d_hat = bus.d_hat
    t = bus.t_now[]

    if mode == :velocity
        Vx_d, Vy_d = ref.Vx(t), ref.Vy(t)
        omega_d    = ref.Wz(t)
        Ax_d, Ay_d = ref.Ax(t), ref.Ay(t)
        alpha_d    = ref.al(t)
        dpsi = psi - ref.psi(t)

        s_x = Vx - Vx_d
        s_y = Vy - Vy_d
        epsi = Main.ControllerMod._e_psi(dpsi)
        edpsi = Main.ControllerMod._edash_psi(dpsi) * (psi_dot - omega_d)
        lam_psi, lam_dot_psi = lambda_schedule(epsi, edpsi, asmc.lam_psi_max, asmc.v_max_axis[3])
        s_psi = edpsi + lam_psi * epsi

        Ax_eq, Ay_eq = Ax_d, Ay_d
        alpha_eq = alpha_d - lam_dot_psi * epsi - lam_psi * edpsi * Main.ControllerMod._edash_psi(dpsi)

    elseif mode == :pose
        ref::Main.Profiles.PosRef
        e_xo = xhat[5] - ref.xo(t)
        e_yo = xhat[6] - ref.yo(t)
        cp, sp = cos(psi), sin(psi)
        e_x =  e_xo * cp + e_yo * sp
        e_y = -e_xo * sp + e_yo * cp
        dpsi = psi - ref.psi(t)
        epsi = Main.ControllerMod._e_psi(dpsi)

        Vx_d, Vy_d, omega_d = Main.Profiles.global_to_local_frame(t, psi, ref.Vxo, ref.Vyo, ref.om)

        edot_x   = Vx - Vx_d + psi_dot * e_y
        edot_y   = Vy - Vy_d - psi_dot * e_x
        edot_psi = psi_dot - omega_d
        edashpsi = Main.ControllerMod._edash_psi(dpsi)

        lam_x,   lam_dot_x   = lambda_schedule(e_x, edot_x, asmc.lam_x_max, asmc.v_max_axis[1])
        lam_y,   lam_dot_y   = lambda_schedule(e_y, edot_y, asmc.lam_y_max, asmc.v_max_axis[2])
        lam_psi, lam_dot_psi = lambda_schedule(epsi, edashpsi * edot_psi, asmc.lam_psi_max, asmc.v_max_axis[3])

        s_x   = edot_x   + lam_x   * e_x
        s_y   = edot_y   + lam_y   * e_y
        s_psi = edot_psi + lam_psi * epsi

        Ax_des, Ay_des, alpha_des = Main.Profiles.global_to_local_frame(t, psi, ref.Axo, ref.Ayo, ref.al)
        alpha_eq = alpha_des - lam_dot_psi * epsi - lam_psi * edot_psi * edashpsi
        Ax_eq    = Ax_des - lam_dot_x * e_x - lam_x * edot_x + psi_dot * (Vy_d - edot_y) - alpha_eq * e_y
        Ay_eq    = Ay_des - lam_dot_y * e_y - lam_y * edot_y - psi_dot * (Vx_d - edot_x) + alpha_eq * e_x
    else
        error("asmc_wrench! (V2) mode must be :velocity or :pose; got $mode")
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
    M_psi_eq = (I_psi_local * alpha_eq
                - m * aY * (Ax_eq - psi_dot * Vy)
                + m * aX * (Ay_eq + psi_dot * Vx)
                + (4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-params.Ra)^2) * psi_dot)

    if bus.use_dhat
        Mdh = params.M_aug * d_hat
        Mx_eq    -= R * Mdh[1]
        My_eq    -= R * Mdh[2]
        M_psi_eq -=     Mdh[3]
    end

    # --- Per-tick state-dependent ceiling (kmax_schedule) + lazy clamp ------
    W_ff = SVector(Mx_eq, My_eq, M_psi_eq)
    if asmc.use_scheduled_kmax
        asmc.Vy_filt[1]     += (dt / asmc.kmax_lpf_tau) * (Vy - asmc.Vy_filt[1])
        asmc.psidot_filt[1] += (dt / asmc.kmax_lpf_tau) * (psi_dot - asmc.psidot_filt[1])
        K_max_sched = kmax_schedule(asmc.lim, asmc.Vy_filt[1], asmc.psidot_filt[1], W_ff)
        K_max_sched = max.(K_max_sched, 0.05 .* asmc.K_max_base)   # degenerate-case floor (brief §9)
    else
        K_max_sched = asmc.K_max_base
    end
    K_max_eff = max.(SVector(K_x, K_y, K_psi), K_max_sched)   # LAZY clamp (projection, not hard clamp)

    base_dK_x   = asmc.gamma_x   * (s_x   * ss_x)
    base_dK_y   = asmc.gamma_y   * (s_y   * ss_y)
    base_dK_psi = asmc.gamma_psi * (s_psi * ss_psi)

    dK_x = base_dK_x * _smooth_bound_v2(K_x, K_max_eff[1]) -
           asmc.cubic_coeff[1] * (K_x / K_max_eff[1])^3 -
           asmc.decay_sigma[1] * (K_x - asmc.K_x0 * 0.95) * exp(asmc.decay_k * (1 - s_x^2 / (9*asmc.eps^2)))
    dK_y = base_dK_y * _smooth_bound_v2(K_y, K_max_eff[2]) -
           asmc.cubic_coeff[2] * (K_y / K_max_eff[2])^3 -
           asmc.decay_sigma[2] * (K_y - asmc.K_y0 * 0.95) * exp(asmc.decay_k * (1 - s_y^2 / (9*asmc.eps^2)))
    dK_psi = base_dK_psi * _smooth_bound_v2(K_psi, K_max_eff[3]) -
             asmc.cubic_coeff[3] * (K_psi / K_max_eff[3])^3 -
             asmc.decay_sigma[3] * (K_psi - asmc.K_psi0 * 0.95) * exp(asmc.decay_k * (1 - s_psi^2 / (9*asmc.eps_psi^2)))

    bus.K = SVector(K_x + dt*dK_x, K_y + dt*dK_y, K_psi + dt*dK_psi)

    if asmc.log_K
        push!(asmc.K_log, bus.K)
        push!(asmc.K_max_sched_log, K_max_sched)
    end

    return SVector(Mx_sw + Mx_eq, My_sw + My_eq, Mpsi_sw + M_psi_eq)
end

# =============================================================================
# PIDControllerV2 (instructions/pid-v2-imc-cascade-two-variants.md)
# =============================================================================
"""
    PIDControllerV2

Cascade-PID parameter carrier with IMC-derived gains (`imc_gains`) and the
two variants (`feedforward=false` -> PID-FB, `true` -> PID-CT). Only
`lam_inner` (`PID_SPACE_V2`) is tuned; everything else follows from pole
cancellation, the cascade separation ratio `N`, or `imax_from_measured`.
"""
Base.@kwdef mutable struct PIDControllerV2
    lim::PhysicalLimits

    # Tuned (PID_SPACE_V2)
    lam_inner::SVector{3,Float64} = SVector(0.15, 0.11, 0.12)

    # Pinned (brief §7.2: N=4.0 for all runs in this brief; the N-sensitivity
    # sweep is deferred, brief §11)
    N::Float64 = 4.0

    # Variant + ablation toggles
    feedforward::Bool   = false   # false=PID-FB (industrial baseline), true=PID-CT (computed-torque)
    use_rate_limit::Bool = true   # false bypasses vcmd_limits (ablation/debug only -- reproduces v1's unbounded V_cmd)

    # Derived at construction (IMC pole cancellation + measured p95 anti-windup)
    Kp::SVector{3,Float64}     = imc_gains(lim, lam_inner, N)[1]
    Ki::SVector{3,Float64}     = imc_gains(lim, lam_inner, N)[2]
    Kp_pos::SVector{3,Float64} = imc_gains(lim, lam_inner, N)[3]
    I_max::SVector{3,Float64}  = imax_from_measured(lim, imc_gains(lim, lam_inner, N)[2], feedforward)

    # Fixed -- a RESULT of IMC on a first-order plant (Kd) and of the outer
    # integrator plant having no damping deficit to fix (Kd_pos), not a
    # simplification. Kept for schema compatibility; no derivative path wired.
    Kd::SVector{3,Float64}     = SVector(0.0, 0.0, 0.0)
    Kd_pos::SVector{3,Float64} = SVector(0.0, 0.0, 0.0)

    # Numerical backstop only -- deliberately NO factor of safety (brief §7.1);
    # vcmd_limits is what actually enforces the (margined) envelope.
    vcmd_clamp::SVector{3,Float64} = SVector(4.55, 0.63, 3.80)

    rate_hz::Float64 = 100.0

    # State (persists across ticks, mirrors v1 PIDController's prev_e/prev_e_pos)
    prev_e::MVector{3,Float64}       = MVector(0.0, 0.0, 0.0)
    initialized::Bool                = false
    prev_e_pos::MVector{3,Float64}   = MVector(0.0, 0.0, 0.0)
    pos_initialized::Bool            = false
    prev_vcmd::MVector{3,Float64}    = MVector(0.0, 0.0, 0.0)   # initialized to V_ff(0) on first tick, NOT zero (brief §6)
    vcmd_initialized::Bool           = false
    last_eps_dot::MVector{3,Float64} = MVector(0.0, 0.0, 0.0)   # cached for PID-CT's a_cmd
    last_a_ref::MVector{3,Float64}   = MVector(0.0, 0.0, 0.0)   # cached for PID-CT's a_cmd

    # Optional instrumentation (brief §10: log the fraction of ticks with
    # gamma<1 / guard_hit, and the integral-saturated fraction). Zero cost
    # when log_diag=false (the default; hot-path safe).
    log_diag::Bool = false
    gamma_log::Vector{Float64} = Float64[]
    guard_hit_log::Vector{Bool} = Bool[]
    i_sat_log::Vector{Bool} = Bool[]
end

"""
    pose_outer_loop_v2(xhat, ref, pid::PIDControllerV2, lim, t, dt) -> SVector{3,Float64}

Outer position loop: same body-frame error construction as v1's
`pose_outer_loop` (world error rotated by R(psi)^T, smooth-wrapped heading),
P-only correction (`Kd_pos=0`, no derivative term -- see `PIDControllerV2`
docstring), then `vcmd_limits` (the brief's structural fix) and finally the
`vcmd_clamp` numerical backstop.

`prev_vcmd` is initialized to `V_ff` at the FIRST tick this function is
called (not zero) -- else tick one demands a spurious huge acceleration and
gamma collapses (brief §6 DEGENERATE CASES note).
"""
function pose_outer_loop_v2(xhat, ref, pid::PIDControllerV2, lim::PhysicalLimits, t::Real, dt::Real)
    ref::Main.Profiles.PosRef
    psi = xhat[4]
    cp, sp = cos(psi), sin(psi)
    e_xo = xhat[5] - ref.xo(t)
    e_yo = xhat[6] - ref.yo(t)
    e_x =  e_xo * cp + e_yo * sp
    e_y = -e_xo * sp + e_yo * cp
    e_psi = Main.EstimatorMod._wrap_angle(xhat[4] - ref.psi(t))
    e_body = SVector(e_x, e_y, e_psi)

    de_body = pid.pos_initialized ?
        (e_body .- SVector(pid.prev_e_pos...)) ./ dt : SVector(0.0, 0.0, 0.0)
    pid.prev_e_pos .= e_body
    pid.pos_initialized = true

    Vx_ff, Vy_ff, om_ff = Main.Profiles.global_to_local_frame(t, psi, ref.Vxo, ref.Vyo, ref.om)
    V_ff = SVector(Vx_ff, Vy_ff, om_ff)
    V_cmd_raw = V_ff .- pid.Kp_pos .* e_body

    if !pid.vcmd_initialized
        pid.prev_vcmd .= V_ff
        pid.vcmd_initialized = true
    end

    V_cmd, gamma, guard_hit = if pid.use_rate_limit
        vcmd_limits(lim, xhat, dt, SVector(pid.prev_vcmd...), V_cmd_raw)
    else
        V_cmd_raw, 1.0, false
    end
    V_cmd = clamp.(V_cmd, .-pid.vcmd_clamp, pid.vcmd_clamp)   # numerical backstop, no FOS
    pid.prev_vcmd .= V_cmd

    pid.last_eps_dot .= de_body
    Ax_des, Ay_des, alpha_des = Main.Profiles.global_to_local_frame(t, psi, ref.Axo, ref.Ayo, ref.al)
    pid.last_a_ref .= SVector(Ax_des, Ay_des, alpha_des)

    if pid.log_diag
        push!(pid.gamma_log, gamma)
        push!(pid.guard_hit_log, guard_hit)
    end

    return V_cmd
end

"""
    Main.ControllerMod.pid_wrench!(bus, xhat, ref, pid::PIDControllerV2, dt; mode=:pose)

Cascade PID wrench, both variants:

    e = xhat[1:3] - V_cmd
    I = clamp(bus.I_pid + dt*e, +/- I_max)         (reuses bus.I_pid, same as v1)
    W = -(Kp.*e + Ki.*I)                            [PID-FB]
    W = M_eq_cmd - (Kp.*e + Ki.*I)                  [PID-CT]

`M_eq_cmd` is built from the COMMANDED velocity/position-error-rate, not the
reference -- `a_cmd = a_ref - Kp_pos.*eps_dot`, `M_eq_cmd = m_eff.*a_cmd +
d_eff.*V_cmd + Coriolis(psi_dot, V_hat)`. The Coriolis/COM cross terms use the
ESTIMATED state (`xhat`, matching the brief's `Coriolis(psi_dot, V_hat)`
notation) -- a deliberate mixed convention: the plant terms mirror what's
COMMANDED, but Coriolis is a real physical effect that depends on the
platform's ACTUAL motion. Only `:pose` is supported (brief: every trajectory
runs `:pose`; velocity-mode paths are out of scope).

`feedforward=false` gives `M_eq_cmd` no contribution at all, so PID-CT with
the feedforward forced to zero reproduces PID-FB bit-identically by
construction (brief §10 "variant equivalence").
"""
function Main.ControllerMod.pid_wrench!(bus, xhat, ref, pid::PIDControllerV2, dt;
                                        mode::Symbol=:pose)
    mode == :pose || error("pid_wrench! (V2) only supports mode=:pose (brief: every trajectory runs :pose)")

    t = bus.t_now[]
    V_cmd = pose_outer_loop_v2(xhat, ref, pid, pid.lim, t, dt)
    e = SVector(xhat[1] - V_cmd[1], xhat[2] - V_cmd[2], xhat[3] - V_cmd[3])

    I_new = clamp.(bus.I_pid .+ dt .* e, .-pid.I_max, pid.I_max)
    if pid.log_diag
        saturated = any(abs.(I_new) .>= abs.(pid.I_max) .- 1e-9)
        push!(pid.i_sat_log, saturated)
    end
    bus.I_pid = I_new

    fb = .-(pid.Kp .* e .+ pid.Ki .* bus.I_pid)

    if pid.feedforward
        psi_dot, Vx, Vy = xhat[3], xhat[1], xhat[2]
        a_ref   = SVector(pid.last_a_ref...)
        eps_dot = SVector(pid.last_eps_dot...)
        a_cmd   = a_ref .- pid.Kp_pos .* eps_dot

        ms, m, aX, aY = pid.lim.ms, pid.lim.m, pid.lim.aX, pid.lim.aY
        coriolis_x = -ms * psi_dot * Vy - m * aX * psi_dot^2
        coriolis_y =  ms * psi_dot * Vx - m * aY * psi_dot^2
        M_eq_cmd = SVector(
            pid.lim.m_eff[1] * a_cmd[1] + pid.lim.d_eff[1] * V_cmd[1] + coriolis_x,
            pid.lim.m_eff[2] * a_cmd[2] + pid.lim.d_eff[2] * V_cmd[2] + coriolis_y,
            pid.lim.m_eff[3] * a_cmd[3] + pid.lim.d_eff[3] * V_cmd[3],
        )
        return M_eq_cmd .+ fb
    else
        return fb
    end
end

# =============================================================================
# MPC v2 — Bryson-normalized weights, U_eq-centered effort, Riccati terminal
# cost (instructions/mpc-v2-bryson-weights-terminal-cost.md). Extends the
# EXISTING `MPCControllerV2` struct and `Main.ControllerMod.mpc_wrench!`
# method defined for the earlier Stage-3 `P_terminal` brief (both live only
# in this file) -- `hybrid_ctrl/controllers.jl` is still NEVER touched. Lives
# here, after PhysicalLimits/ASMC v2/PID v2 above, since `bryson_R`/
# `u_eq_horizon` both consume `PhysicalLimits`/`capability_wrench`.
#
# Search dimension 11 -> 1 (brief §1): `Q_pose` and `R` are DERIVED from
# `Main.TOL` and the capability wrench respectively; only `S_scale` (a single
# log-scaled voltage-rate weight) remains free. This DISSOLVES the v1 flat
# direction (uniform `(Q,R,S)` rescale leaves the QP argmin unchanged) rather
# than constraining it, since physically-scaled weights have no free
# multiplier left (brief §1 point 3).
# =============================================================================

"""
    bryson_Q_pose(tol_pos, tol_head, tau_cl) -> SVector{6,Float64}

Bryson-normalized state weights for the 6-state pose model, ordered
(x, y, psi, Vx, Vy, psidot). `Q_ii = 1/(largest acceptable error in state i)^2`.

Position/heading use `tol_pos`/`tol_head` directly (`Main.TOL.pos_max`/
`Main.TOL.head_max`, NOT `*_final` -- a steady-state spec this MPC's no-
integral-action structure cannot honor via any weight choice, brief §9/§6).
Velocity/yaw-rate have no tolerance of their own in the `:pose` objective
(`TOL.vel_rms`/`yawrate_rms` are consumed ONLY by the `:velocity` scoring
branch, brief §6 CRITICAL note) -- so the velocity tolerance is derived from
the position tolerance and a stated closed-loop time constant `tau_cl` (a
velocity error sustained for `tau_cl` produces a position error
`eps_v*tau_cl`):

    Q_vx = q_pos  * tau_cl[1]^2      Q_vy = q_pos * tau_cl[2]^2
    Q_wz = q_head * tau_cl[3]^2

(the brief's stated identity `Q_vel,i = Q_pos*tau_cl_i^2` holds exactly here
because `tol_pos == tol_head` at the current `Main.TOL` values; the yaw-rate
entry genuinely uses `q_head`, not `q_pos`, in general.)

`tau_cl` DEFAULTS to the PID v2 `lam_inner = (0.15, 0.11, 0.12)` s -- a
deliberate shared-bandwidth choice across all three v2 controllers (brief
§2), not an inherited constant, and must be REPORTED as such. At the
defaults (`tol_pos=tol_head=0.1`): `Q_pose = (100, 100, 100, 2.25, 1.21,
1.44)` -- matches brief §6 exactly.
"""
function bryson_Q_pose(tol_pos::Real, tol_head::Real, tau_cl::SVector{3,Float64})
    q_pos  = 1.0 / tol_pos^2
    q_head = 1.0 / tol_head^2
    q_vx = q_pos  * tau_cl[1]^2
    q_vy = q_pos  * tau_cl[2]^2
    q_wz = q_head * tau_cl[3]^2
    return SVector(q_pos, q_pos, q_head, q_vx, q_vy, q_wz)
end

"""
    bryson_R(lim::PhysicalLimits, motor) -> SVector{4,Float64}

Bryson-normalized per-wheel weight on the voltage DEVIATION from `U_eq` (NOT
the absolute voltage -- see `u_eq_horizon`). `R_jj = 1/(largest acceptable
deviation from U_eq)^2`, sized from the per-wheel voltage that commands each
axis's FULL single-axis capability (`capability_wrench(lim)`) through the
O-config allocation's pure-axis columns (`0.25` for x/y, `0.25*lim.lever` for
psi -- the same allocation `ControllerMod._mpc_body_matrices`'s `Amix`
encodes), converted to voltage via `dTaudV = G*eta*Kt/Ra`. Takes the MAX over
axes so `R` never over-penalizes the axis with the largest voltage demand;
all four wheels get the SAME weight (identical motors/radii, O-config gives
each an equal-magnitude share -- a per-wheel `R` would break a symmetry the
plant genuinely has, brief §6).

Expected ~0.045-0.047 (brief states 0.0447 from reference values 6.59/6.43/
56.7 N*m; this codebase's own `capability_wrench` psi-entry runs ~1.4% below
56.7 per the `PhysicalLimits` docstring's own already-documented a_cap
rounding -- a pre-existing gap, not new here).
"""
function bryson_R(lim::PhysicalLimits, motor)
    dTaudV = motor.G * motor.eta * motor.Kt / motor.Ra
    W = capability_wrench(lim)
    per_wheel = SVector(0.25 * W[1], 0.25 * W[2], 0.25 * lim.lever * W[3])
    dV_cap = maximum(per_wheel ./ dTaudV)
    Rjj = 1.0 / dV_cap^2
    return SVector{4}(fill(Rjj, 4))
end

"""
    DARECache

Mutable cache cell for `terminal_cost`'s DARE solve, keyed on the operating-
point matrix `A_end`. Recomputes only when `A_end` moves past a relative
threshold (brief: "~1e-3 is a reasonable trigger"); falls back to the last
successful `P` (or `zeros(6,6)` pre-first-success) on an `ared` failure, and
COUNTS both cache activity and failures rather than silencing them (brief:
"a nonzero failure count is a reportable finding, not something to silence").
"""
mutable struct DARECache
    A_end::Matrix{Float64}
    P::Matrix{Float64}
    valid::Bool
    hits::Int
    misses::Int
    failures::Int
end
DARECache() = DARECache(zeros(6, 6), zeros(6, 6), false, 0, 0, 0)

"""
    _equiv_wrench_mpc(params, lim, Vx, Vy, psidot, Ax_eq, Ay_eq, alpha_eq) -> SVector{3,Float64}

The SAME equivalent-control wrench expression as `ControllerMod.asmc_wrench!`'s
`Mx_eq`/`My_eq`/`M_psi_eq` (brief `u_eq_horizon` note: "one object, three
consumers" -- ASMC's equivalent control, PID-CT's `M_eq_cmd`, and this MPC's
`U_eq` feedforward are all the same physical quantity), extracted as its own
function so `u_eq_horizon` can evaluate it AT THE REFERENCE (`Vx`,`Vy`,
`psidot`,`Ax_eq`,`Ay_eq`,`alpha_eq` all taken from the trajectory, zero error
terms) without duplicating or touching the ASMC method body.
"""
function _equiv_wrench_mpc(params, lim::PhysicalLimits, Vx::Real, Vy::Real, psidot::Real,
                           Ax_eq::Real, Ay_eq::Real, alpha_eq::Real)
    m, ms, J1 = params.m, params.ms, params.J_wheel
    aX, aY = params.aX, params.aY
    p1, p2 = params.p1_case1, params.p2_case1
    R, Ra, l, h = params.R, params.Ra, params.l, params.h

    Mx_eq = R * ((ms + 4*J1/R^2) * Ax_eq
                 - ms * psidot * Vy
                 - m  * aY * alpha_eq
                 - m  * aX * psidot^2
                 + 4  * p1 * Vx / R^2)
    My_eq = R * ((ms + 4*J1/R^2) * Ay_eq
                 + ms * psidot * Vx
                 + m  * aX * alpha_eq
                 - m  * aY * psidot^2
                 + (4 * p1 / R^2 + 8 * p2 / (R - Ra)^2) * Vy)
    M_psi_eq = (lim.I_psi * alpha_eq
                - m * aY * (Ax_eq - psidot * Vy)
                + m * aX * (Ay_eq + psidot * Vx)
                + (4*p1*(l+h)^2/R^2 + 8*p2*h^2/(R-Ra)^2) * psidot)
    return SVector(Mx_eq, My_eq, M_psi_eq)
end

"""
    u_eq_horizon(ref, t, dt, Np, params, motor, lim) -> Vector{Float64}

Stacked feedforward voltage `[U_eq,1; ...; U_eq,Np]`, length `4*Np`. Removes
v1's droop (`R` penalizing the ABSOLUTE voltage, including the back-EMF/drag
voltage a constant-velocity segment genuinely needs -- a bias this no-
integral-action MPC can never reject, brief §1 point 1) by re-centering the
effort penalty on this feedforward.

PER STEP k: (1) `_equiv_wrench_mpc` at the reference operating point (zero
error terms) gives `W_eq,k`; (2) `tau_eq,k = Amix*W_eq,k` (4x3 * 3x1, EXACT,
no pinv -- the forward allocation direction is unambiguous, brief: "do NOT
use pinv here"); (3) per-wheel motor inversion `V_eq,k,j = tau_eq,k,j/dTaudV +
Kb*G*omega_ref,k,j`, with `omega_ref,k = H_omega*[Vx_r; Vy_r; psidot_r]` (the
SAME no-slip wheel-speed map `EstimatorMod._wheel_jacobian` already encodes
-- reused, not re-derived, per the brief's sign-error caution).

NULL SPACE: since `tau_eq` lands in `range(Amix)` by construction, and the
O-config null vector `n=[1,1,-1,-1]` is orthogonal to that range, the
feedforward carries zero internal tension automatically (`dot(tau_eq,n)==0`
to machine precision -- brief §10 null-space check).

DOCUMENTED APPROXIMATION: step 3 inverts the quasi-static LINEAR motor map
and ignores `PlantMod`'s Coulomb term `tau_f`. `MotorParams.tau_f=0.01` is
specified in MOTOR-SHAFT units (`PlantMod.MotorParams`'s own comment: "N*m
motor-shaft friction"); `motor_torque` scales the WHOLE motor-shaft torque
(including this term) by the gearbox `G*eta` before returning wheel-side
torque, so the WHEEL-side Coulomb residual is actually `G*eta*tau_f ~
23.26*0.01 ~ 0.233 N*m`, not `0.01 N*m` -- ~23x bigger than a naive reading
of the brief's own stated "<1% of tau_eq~1.6 N*m" residual suggests.
MEASURED (`_tmp/mpc_v2_validation.jl` check 4, ellipse trajectory, t=3.7s):
round-trip residual ~12% of `‖W_eq‖`, uniform across all 4 wheels (confirmed
via `dot(tau_diff,[1,1,1,1])` -- the Wx-recovery direction, not the null
space) -- consistent with `G*eta*tau_f` exactly, not a bug. Reported here as
a corrected finding, brief: "do not iterate to invert it exactly" (still
holds -- the fix is to widen the expected-residual bound, not to compensate
`tau_f` here). Deliberately does NOT reuse `PlantMod.motor_torque_inverse`
(which DOES compensate `tau_f`) for this reason.
"""
function u_eq_horizon(ref, t::Real, dt::Real, Np::Int, params, motor, lim::PhysicalLimits)
    ref::Main.Profiles.PosRef
    Hω = Main.EstimatorMod._wheel_jacobian(params)
    _, _, Amix = Main.ControllerMod._mpc_body_matrices(params, motor, dt)
    dTaudV = motor.G * motor.eta * motor.Kt / motor.Ra

    U = zeros(4 * Np)
    for k in 1:Np
        tk = t + (k - 1) * dt
        psik = ref.psi(tk)
        ck, sk = cos(psik), sin(psik)
        Vxw, Vyw = ref.Vxo(tk), ref.Vyo(tk)
        Vx_r = Vxw * ck + Vyw * sk
        Vy_r = -Vxw * sk + Vyw * ck
        psidot_r = ref.om(tk)

        Ax_r, Ay_r, alpha_r = Main.Profiles.global_to_local_frame(tk, psik, ref.Axo, ref.Ayo, ref.al)

        W_eq = _equiv_wrench_mpc(params, lim, Vx_r, Vy_r, psidot_r, Ax_r, Ay_r, alpha_r)
        tau_eq = Amix * W_eq

        omega_ref = Hω * SVector(Vx_r, Vy_r, psidot_r)
        V_eq = tau_eq ./ dTaudV .+ (motor.Kb * motor.G) .* omega_ref

        U[(k-1)*4+1:k*4] .= V_eq
    end
    return U
end

"""
    terminal_cost(A_end, Bm, Q, R, cache) -> Matrix{Float64}

Infinite-horizon cost-to-go `P` from the discrete algebraic Riccati equation,
via `MatrixEquations.ared(A_end, Bm, Rm, Qm)` (never hand-rolled, brief §3).
NOT a tuning parameter -- fully derived from `(A_end, Bm, Q, R)`, all of which
already exist. Removes the stability gap of finite-horizon MPC with no
terminal cost (brief §1 point 2): `Np_pose=15` at 100 Hz is 0.15 s, under one
open-loop time constant on x (`tau_open_x=0.256 s`), so a curvature change
just past the horizon is invisible until the demand arrives all at once.

Caches on `A_end` (`cache::DARECache`), recomputing only when it moves past a
~1e-3 relative threshold; on an `ared` failure (unstabilizable linearization
at some operating point) falls back to the last successful `P`, or
`zeros(6,6)` pre-first-success, and increments `cache.failures` -- never
throws inside the 100 Hz loop (brief: "must degrade, not crash").

Uses `As[Np]` (the transition at the HORIZON END, where the terminal cost
applies), NOT a frozen nominal `A` -- x/y isotropy does not hold here
(`bryson_Q_pose` gives `Q_vx=2.25` vs `Q_vy=1.21`), so a single rotation-
invariant solve is not available as a shortcut (brief §9).

DOCUMENTED APPROXIMATION: the DARE models `(A,B,Q,R)` but NOT the rate
penalty `S` -- defensible since `S` is a transient-shaping penalty whose
steady-state contribution is zero (brief §9), but belongs in the paper's
method section, not hidden.
"""
function terminal_cost(A_end::Matrix{Float64}, Bm::Matrix{Float64},
                       Q::SVector{6,Float64}, R::SVector{4,Float64}, cache::DARECache)
    if cache.valid && isapprox(A_end, cache.A_end; rtol=1e-3, atol=1e-6)
        cache.hits += 1
        return cache.P
    end
    Qm = Matrix(Diagonal(collect(Q)))
    Rm = Matrix(Diagonal(collect(R)))
    try
        P, _, _ = MatrixEquations.ared(A_end, Bm, Rm, Qm)
        Psym = Matrix(P)
        cache.A_end = copy(A_end)
        cache.P = Psym
        cache.valid = true
        cache.misses += 1
        return Psym
    catch e
        cache.failures += 1
        return cache.valid ? cache.P : zeros(6, 6)
    end
end

"""
    _build_Su!(Sx, Su, As, Bm, n, m, Np) -> Nothing

Condensed LTV prediction matrices `X = Sx*x0 + Su*U`, built by COLUMN
RECURSION instead of v1's recomputed-product nested loops. MATHEMATICALLY
IDENTICAL (verified to machine precision, not necessarily bit-identical --
floating-point reassociation, brief §10): for a fixed input column `j`,
`Phi(k,j) = As[k]*Phi(k-1,j)` starting from `Phi(j,j)=I`, reusing the running
product across increasing `k` -- exactly the same operand sequence
`As[k]*As[k-1]*...*As[j+1]` v1's inner loop recomputed from scratch for every
`(k,j)` pair, just accumulated once per column instead of `O(Np)` times per
column. `O(Np^3) -> O(Np^2)`: `Np_pose` going from 15 to 30 (brief §7.2) is an
~8x increase in v1's per-tick cost on this 100 Hz hot path, which this fix
recovers.
"""
function _build_Su!(Sx::Matrix{Float64}, Su::Matrix{Float64},
                    As::Vector{Matrix{Float64}}, Bm::Matrix{Float64},
                    n::Int, m::Int, Np::Int)
    fill!(Sx, 0.0)
    fill!(Su, 0.0)

    Phi0 = Matrix{Float64}(I, n, n)
    for k in 1:Np
        Phi0 = As[k] * Phi0
        Sx[(k-1)*n+1:k*n, :] .= Phi0
    end

    for j in 1:Np
        Phij = Matrix{Float64}(I, n, n)
        Su[(j-1)*n+1:j*n, (j-1)*m+1:j*m] .= Phij * Bm
        for k in (j+1):Np
            Phij = As[k] * Phij
            Su[(k-1)*n+1:k*n, (j-1)*m+1:j*m] .= Phij * Bm
        end
    end
    return nothing
end

"""
    MPCControllerV2 (extended -- instructions/mpc-v2-bryson-weights-terminal-cost.md)

New fields since the Stage-3 `P_terminal` brief: `lim`, `motor`, `tau_cl`
(shared v2 bandwidth), `use_u_eq`/`use_terminal` (ablation toggles for the
brief §10 regression gate), `dare_cache`, `log_diag` + diagnostics.

Changed defaults: `Np_pose: 15 -> 30`; `Q_pose`/`R` now DERIVED at
construction (`bryson_Q_pose`/`bryson_R`) instead of free constants; `S`
derived from the single searched `S_scale` (brief §7.2, `MPC_SPACE_V2`).

Field-declaration order matters (`@kwdef` evaluates defaults in order,
matching `ASMCControllerV2`/`PIDControllerV2`'s existing documented
constraint): `lim`/`motor`/`tau_cl` precede `Q_pose` (needs `tau_cl`) and `R`
(needs `lim`/`motor`); `S_scale` precedes `S` (needs `S_scale`).

`Np`/`Q` (velocity-mode) are DEAD -- kept for schema compatibility only;
`mpc_wrench!` errors on `mode=:velocity` in v2 (brief §7.2, matching the
`PIDControllerV2`/`pid_wrench!` precedent).
"""
Base.@kwdef mutable struct MPCControllerV2
    lim::PhysicalLimits
    motor::Main.PlantMod.MotorParams = Main.PlantMod.MotorParams()
    tau_cl::SVector{3,Float64} = SVector(0.15, 0.11, 0.12)   # PID v2 lam_inner -- shared v2 bandwidth (brief §2/§6)

    Np::Int = 10                                              # dead (velocity-mode; :velocity errors in v2)
    Q::SVector{3,Float64} = SVector(50.0, 50.0, 80.0)         # dead, ditto
    Np_pose::Int = 30                                         # CHANGED 15 -> 30 (brief §7.2)

    Q_pose::SVector{6,Float64} = bryson_Q_pose(Main.TOL.pos_max, Main.TOL.head_max, tau_cl)
    R::SVector{4,Float64}      = bryson_R(lim, motor)

    S_scale::Float64 = 2.5                                    # SEARCHED (MPC_SPACE_V2, tune_controller_v2.jl)
    S::SVector{4,Float64} = SVector{4}(fill(S_scale, 4))

    P_terminal::Matrix{Float64} = zeros(6, 6)                 # filled per-tick by terminal_cost when use_terminal
    rate_hz::Float64 = 100.0
    use_ltv::Bool = true

    use_u_eq::Bool     = true
    use_terminal::Bool = true
    dare_cache::DARECache = DARECache()

    # Optional per-tick wall-clock instrumentation (brief §10). Zero cost when
    # log_diag=false (the default; hot-path safe).
    log_diag::Bool = false
    ticktime_log::Vector{Float64} = Float64[]

    # Optional per-tick horizon-plan log (brief §7.3 Np_pose sweep support):
    # the raw QP solution (u0 = U1, full res.x, and the box/rate bounds that
    # applied at this tick), needed to check plan convergence and terminal-
    # region constraint activity across Np without re-deriving the bounds
    # externally. Same zero-cost-when-off contract as ticktime_log.
    plan_log::Vector{NamedTuple} = NamedTuple[]
end

"""
    Main.ControllerMod.mpc_wrench!(bus, xhat, ref, params, motor, mpc::MPCControllerV2; mode=:pose)

Extended per brief §7.1. `:velocity` mode ERRORS in v2 (dead-parameter
handling, brief §7.2) -- only `:pose` is implemented. Five changes from the
pre-existing `MPCControllerV2` method (four from the brief, one additional
correctness fix made per explicit user direction after the brief landed):

  1. `_build_Su!` replaces the nested-loop `Sx`/`Su` construction (efficiency
     only, §10 machine-precision equivalence).
  2. `U_eq_stack = u_eq_horizon(...)` re-centers the voltage-effort penalty
     (`f` gains `-Rd*U_eq_stack`; `H` is UNCHANGED -- the re-centering is a
     linear shift, brief §7.1 step 9).
  3. `P = terminal_cost(As[Np], Bm, Q_pose, R, mpc.dare_cache)` is added to
     the LAST `n`x`n` block of `Qd` when `mpc.use_terminal`.
  4. Both (2) and (3) are individually togglable (`use_u_eq`/`use_terminal`).
  5. **[NOT in the original brief -- added per user direction]** `xref_fn`'s
     velocity sub-block now rotates the reference's GLOBAL `Vxo`/`Vyo` into
     BODY frame using the CURRENT ESTIMATED heading `xhat[4]` (frozen across
     the whole horizon), matching `ControllerMod.asmc_wrench!`/
     `ControllerV2Mod.pose_outer_loop_v2`'s `global_to_local_frame(t,
     psi=xhat[4], ...)` convention exactly. The pre-existing (and v1)
     behaviour rotated using the REFERENCE's own future heading `ref.psi(tk)`
     at each horizon step instead -- a subtly different frame per step, which
     technically compared `xhat`'s body-frame velocity against a reference
     velocity expressed in a slightly different (future, reference-only)
     body frame whenever `ψ̂` has drifted from `ψ_ref`. Position/heading
     (`xo,yo,psi`) are UNCHANGED -- still raw world-frame, no rotation.
     Deliberately NOT a global-frame comparison: `bryson_Q_pose`'s
     `Q_vx=2.25` vs `Q_vy=1.21` are body-frame forward/lateral bandwidths
     (from `tau_cl_x`/`tau_cl_y`); comparing in global frame would apply that
     anisotropy to world X/Y instead, making the tracking penalty depend on
     which way the platform happens to be pointing -- a real bug, ruled out
     explicitly (see chat).

**Consequence for the brief's own §10 regression gate**: item 5 means
`MPCControllerV2` can no longer reproduce the ORIGINAL `ControllerMod.
MPCController` to machine precision even with `use_u_eq=false`/
`use_terminal=false`/v1-default weights -- that gate's claim is downgraded to
"machine precision on a CONSTANT-heading reference (where `ref.psi(tk)` and
`xhat[4]` coincide for every horizon step by construction) and a small,
explained, non-zero residual on a curving one" (`_tmp/mpc_v2_validation.jl`
checks 8a/8b). This is intentional -- item 5 is a correctness fix, not
something a toggle should be able to disable back to the old (subtly wrong)
behaviour.
"""
function Main.ControllerMod.mpc_wrench!(bus, xhat, ref, params, motor, mpc::MPCControllerV2;
                                        mode::Symbol=:pose)
    mode == :pose || error("mpc_wrench! (V2) only supports mode=:pose -- :velocity errors in v2 (brief §7.2)")
    ref::Main.Profiles.PosRef
    tick_start = mpc.log_diag ? time() : 0.0

    t  = bus.t_now[]
    dt = 1.0 / mpc.rate_hz

    ω = SVector{4,Float64}(bus.y_last.ω[1], bus.y_last.ω[2], bus.y_last.ω[3], bus.y_last.ω[4])
    _, Bvel, Amix = Main.ControllerMod._mpc_body_matrices(params, motor, dt)

    n, m, Np = 6, 4, mpc.Np_pose
    Bm = zeros(6, 4);  Bm[4:6, :] .= Matrix(Bvel)
    x0 = SVector(xhat[5], xhat[6], xhat[4], xhat[1], xhat[2], xhat[3])
    Qtrack = mpc.Q_pose

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
        push!(As, Matrix(Main.ControllerMod._mpc_pose_A(params, dt, v_op, psi_k)))
    end
    # [CHANGED] ASMC/PID frame-matching fix: rotate the reference's GLOBAL
    # velocity into BODY frame using the CURRENT ESTIMATED heading xhat[4]
    # (frozen across the whole horizon), not the reference's own future
    # ref.psi(tk) -- matches asmc_wrench!/pose_outer_loop_v2's
    # global_to_local_frame(t, psi=xhat[4], ref.Vxo, ref.Vyo, ref.om) exactly.
    # Position/heading (xo,yo,psi) stay untouched, unrotated, in world frame.
    # Deliberately NOT a global-frame comparison: bryson_Q_pose's Q_vx=2.25 vs
    # Q_vy=1.21 are body-frame forward/lateral bandwidths (tau_cl_x/y) -- a
    # global-frame comparison would apply that anisotropy to world X/Y instead,
    # making the penalty heading-dependent (a real bug, not a style choice).
    c_hat, s_hat = cos(xhat[4]), sin(xhat[4])
    xref_fn = function (tk)
        Vxw, Vyw = ref.Vxo(tk), ref.Vyo(tk)
        return SVector(ref.xo(tk), ref.yo(tk), ref.psi(tk),
                       Vxw*c_hat + Vyw*s_hat, -Vxw*s_hat + Vyw*c_hat, ref.om(tk))
    end

    # --- [NEW] condensed LTV prediction via column recursion -----------------
    Sx = zeros(n*Np, n)
    Su = zeros(n*Np, m*Np)
    _build_Su!(Sx, Su, As, Bm, n, m, Np)

    xref = zeros(n*Np)
    for k in 1:Np
        tk = t + k*dt
        xref[(k-1)*n+1:k*n] = collect(xref_fn(tk))
    end

    D = zeros(m*Np, m*Np)
    for k in 1:Np
        D[(k-1)*m+1:k*m, (k-1)*m+1:k*m] = Matrix{Float64}(I, m, m)
        if k > 1
            D[(k-1)*m+1:k*m, (k-2)*m+1:(k-1)*m] = -Matrix{Float64}(I, m, m)
        end
    end
    p_prev = vcat(collect(bus.mpc_last_u), zeros(m*(Np-1)))

    # --- [NEW] U_eq feedforward stack ----------------------------------------
    U_eq_stack = mpc.use_u_eq ? u_eq_horizon(ref, t, dt, Np, params, motor, mpc.lim) :
                                zeros(m*Np)

    # --- [NEW] Riccati terminal cost on the LAST block -----------------------
    Qd = Matrix(Diagonal(repeat(collect(Qtrack), Np)))
    if mpc.use_terminal
        P = terminal_cost(As[Np], Bm, Qtrack, mpc.R, mpc.dare_cache)
        Qd[(Np-1)*n+1:Np*n, (Np-1)*n+1:Np*n] .+= P
        mpc.P_terminal = P
    end

    Rd = Diagonal(repeat(collect(mpc.R), Np))
    Sd = Diagonal(repeat(collect(mpc.S), Np))
    H = Su' * Qd * Su + Rd + D' * Sd * D
    H = (H + H') / 2 + 1e-6 * I
    f = Su' * Qd * (Sx * x0 .- xref) .- D' * Sd * p_prev .- Rd * U_eq_stack

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

    model = OSQP.Model()
    OSQP.setup!(model; P = sparse(H), q = Vector(f), A = Acon, l = lcon, u = ucon,
                verbose = false, warm_start = true, polish = true)
    if length(bus.mpc_warm) == m*Np
        OSQP.warm_start!(model; x = bus.mpc_warm)
    end
    res = OSQP.solve!(model)

    if res.info.status_val in (1, 2)
        u0 = SVector{4,Float64}(res.x[1], res.x[2], res.x[3], res.x[4])
        bus.mpc_last_u = u0
        bus.mpc_warm   = copy(res.x)
    else
        u0 = bus.mpc_last_u
    end

    if mpc.log_diag
        push!(mpc.ticktime_log, time() - tick_start)
        push!(mpc.plan_log, (t=t, Np=Np, u0=u0, U_full=copy(res.x),
                             lb_box=copy(lb_box), ub_box=copy(ub_box),
                             lb_rate=copy(lb_rate), ub_rate=copy(ub_rate)))
    end

    τ_u0 = Main.PlantMod.motor_torque(u0, ω, motor)
    W = SVector{3,Float64}(pinv(Matrix(Amix)) * Vector(τ_u0))
    return W
end

end # module
