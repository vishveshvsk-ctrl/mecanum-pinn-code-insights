# ASMC v2 — Physically-Derived Gain Bounds, State-Scheduled Ceiling, and Surface Reparameterization

> **Generated:** 2026-07-29
> **Stack:** Julia 1.12.5 — `code_insights/` project, `hybrid_ctrl_v2/` module tree. No new packages.
> **Scope:** `ControllerV2Mod` extension only. `hybrid_ctrl/controllers.jl` is never edited.
> **Source of physical constants:** `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex`

---

## 1. Overview

Five changes to the adaptive sliding-mode controller, all replacing eyeballed or
frozen constants with values derived from the platform's friction-circle physics:

1. **`K_max` derived, not searched.** The current pinned `(150, 150, 300)` exceeds
   the deliverable wrench by roughly 25×, which silently disables both anti-windup
   mechanisms in the gain law. Replaced by friction-circle-derived values in the
   ratio **1 : 1 : 8.6**.
2. **`K_max` scheduled on state.** The traction circle available to the controller
   shrinks as `V_y` and `ψ̇` consume it. A fixed ceiling is valid at one operating
   point only.
3. **`K_floor` derived** from measured switching demand, allocated on the same ratio.
4. **Sliding-surface reparameterization.** `λ` moves from frozen struct default into
   the tuned set, and its error schedule changes from Gaussian to the
   saturation-derived hyperbolic form.
5. **Gain-decay parameters and the anti-windup knee derived.** Three corrections:
   the σ-leakage exponential must use each axis's own `eps` (v1 uses `eps_psi`
   everywhere); the cubic coefficient must be **proportional to each axis's range**
   (a uniform value gives decay times differing 8.5×); and `_smooth_bound`'s absolute
   `−2.0` knee must become fractional (`0.98·K_max`), or growth on x is throttled at
   70% of its ceiling while ψ is throttled at 96.5%.

**System-level contract:** `ASMCControllerV2` consumes the same
`(bus, xhat, ref, params, dt; mode)` interface as `ASMCController` and returns the
same `SVector{3}` task-space wrench, but its gain bounds are functions of the
estimated state and the platform's analytical limits rather than constants.

**Effect on the tuning space:** `ASMC_SPACE` drops `K_max_{x,y,psi}` (derived) and
the `lam_*_min` / `mu_xy` / `mu_psi` schedule parameters (derived), and gains
`lam_{x,y,psi}_max`. Net: the same order of dimensionality, but spanning the
directions the objective is actually sensitive to.

---

## 2. Architecture Pattern

**New struct + dispatch extension, matching the existing `MPCControllerV2` precedent.**

`ControllerV2Mod` already demonstrates the pattern: define `MPCControllerV2` with the
extra fields, then add a method to the *existing generic*
`Main.ControllerMod.mpc_wrench!` dispatching on the new type. Zero edits to
`hybrid_ctrl/controllers.jl`. Do exactly the same for `ASMCControllerV2` and
`Main.ControllerMod.asmc_wrench!`.

Within the new method, the organizing idea is **derive, don't tune**: every bound is
computed from a documented physical quantity at call time, and only the genuinely
free parameters (surface slopes, adaptation rates) remain in the search space.

---

## 3. Technology Constraints

- **Julia 1.12.5**, existing project. **No new packages.**
- `ControllerV2Mod` is `include`d *after* `tune_controller.jl`, so `Main.ControllerMod`,
  `Main.PlantMod`, `Main.Profiles` and `PlatformParams` are already in scope.
- Must remain allocation-light: this runs at **1000 Hz** inside the ODE callback.
  `SVector`/`SMatrix` throughout; no heap allocation in the hot path.
- `LinearAlgebra.BLAS.set_num_threads(1)` — retain.
- **Explicit exclusions:** no edits to `hybrid_ctrl/controllers.jl`, no plant/sensor/
  estimator changes, no new estimator, no re-tuning within this brief.

---

## 4. Component Breakdown

### `ASMCControllerV2` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** `Base.@kwdef mutable struct`
- **Responsibility:** ASMC parameter carrier with physically-derived bounds replacing
  the fixed `K_max_*`, `lam_*_min/max`, `mu_*` fields.
- **Key fields:**
  - tuned: `lam_x_max`, `lam_y_max`, `lam_psi_max`, `gamma_x`, `gamma_y`, `gamma_psi`
  - derived-at-construction: `K_max_base::SVector{3}`, `K_floor::SVector{3}`,
    `decay_sigma::SVector{3}`, `cubic_coeff::SVector{3}`
  - specified: `eps`, `eps_psi` (set from measured noise, not searched), `tau_relax`,
    `tau_ceiling`, `v_max_axis::SVector{3}`, `rate_hz`
  - schedule config: `use_scheduled_kmax::Bool`, `kmax_lpf_tau`
- **Depends on:** `PhysicalLimits`

### `PhysicalLimits` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** `Base.@kwdef struct` + constructor from `params`
- **Responsibility:** Hold the friction-circle constants from the analytical-limits
  document and expose the binding-wheel gate.
- **Key fields:** `kappa`, `h`, `l`, `mu_N3`, `lever`, `m_tilde`, `I_psi`,
  `a_cap::SVector{3}`, `alloc_ratio::SVector{3}`
- **Depends on:** nothing (leaf); values traced to the .tex document (§7)

### `kmax_schedule` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** function
- **Responsibility:** Per-tick state-dependent gain ceiling from the binding-wheel gate.
- **Inputs:** `lim::PhysicalLimits`, estimated `V_y`, `ψ̇`, and the feedforward wrench
  already computed in `asmc_wrench!`
- **Outputs:** `SVector{3,Float64}` — per-axis ceiling
- **Depends on:** `PhysicalLimits`

### `lambda_schedule` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** function
- **Responsibility:** Saturation-aware surface slope, replacing `_get_dynamic_lambda`.
- **Inputs:** error `e`, error rate `edot`, `lam_max`, `v_max` for that axis
- **Outputs:** `(lam, lam_dot)` — the slope and its time derivative (the equivalent-control
  terms need `lam_dot`, exactly as the v1 law does)
- **Depends on:** nothing

### `Main.ControllerMod.asmc_wrench!` method on `ASMCControllerV2`
- **Type:** dispatch extension
- **Responsibility:** The control law itself — same wrench contract, new bounds.
- **Inputs:** `bus`, `xhat`, `ref`, `params`, `asmc::ASMCControllerV2`, `dt`; `mode=:pose`
- **Outputs:** `SVector{3,Float64}` task-space wrench
- **Depends on:** all of the above

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl_v2/
│   ├── controllers_v2.jl        # MODIFIED — add PhysicalLimits, ASMCControllerV2,
│   │                            #            kmax_schedule, lambda_schedule,
│   │                            #            asmc_wrench! method. MPCControllerV2 untouched.
│   └── tune_controller_v2.jl    # MODIFIED — ASMC_SPACE_V2, build_controller dispatch
├── docs/
│   └── Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex   # READ ONLY — constant source
└── _tmp/
    └── asmc_v2_validation.jl    # NEW — the §10 checks
```

No new files in `hybrid_ctrl_v2/`; everything lands in the existing `ControllerV2Mod`.

---

## 6. Key Interfaces

Signatures only; bodies are stubs.

```julia
"""
    PhysicalLimits(params; mu=0.5)

Friction-circle constants for the youBot O-config platform, traced to
docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex. Every field below
must carry an inline comment naming its source equation.

Fields and their provenance:
    kappa      = sqrt(2)*p2/(R-Ra)^2 = 38.9        (E54)
    mu_N3      = 34.8 N   binding wheel (rear-left) (E54, section 3.4)
    h, l       = 0.235, 0.15 m                      (Table 1)
    lever      = R/(l+h) = 0.130                    mixer allocation
    a_cap      = (2.93, 2.86, 9.62)                 (E53) friction-circle intercepts
    alloc_ratio= (1.0, 1.0, 8.6)                    per-axis wrench capability

NOTE the a_cap values are the FRICTION-CIRCLE intercepts, deliberately NOT the
1.0 m/s^2 operating cap of section 3.5. That cap is a dataset-fidelity budget
(keeping neglected out-of-plane load transfer under 5%), not a hardware limit;
sizing controller authority against it would forfeit authority the platform
physically has.
"""
function PhysicalLimits(params; mu::Float64=0.5) end

"""
    capability_wrench(lim::PhysicalLimits) -> SVector{3,Float64}

Per-axis task-space wrench capability at a free circle, in the controller's own
units (N*m for all three -- the translational channels carry the R factor from
Mx_eq = R*(m_tilde*Ax + ...), the yaw channel uses I_psi directly):

    Mx_max = R * m_tilde * a_cap[1]
    My_max = R * m_tilde * a_cap[2]
    Mpsi_max = I_psi * a_cap[3]

Expected values ~ (6.59, 6.43, 56.7) N*m, i.e. the 1 : 1 : 8.6 ratio. Yaw carries
roughly 8.6x the translational wrench capability; an equal-K allocation therefore
under-uses the yaw axis by about 6x.
"""
function capability_wrench(lim::PhysicalLimits) end

"""
    kmax_schedule(lim, V_y, psidot, W_ff) -> SVector{3,Float64}

State-dependent gain ceiling from the binding-wheel gate (E54).

Steps:
  1. F_perp3     = kappa * (V_y - h*psidot)                velocity-slaved consumption
  2. F_par_avail = sqrt(max(0, (s*mu_N3)^2 - F_perp3^2))   remaining traction, s = 0.9
  3. subtract the feedforward's own draw on the binding wheel, from W_ff
  4. map the residual back to a per-axis wrench budget using alloc_ratio

s = 0.9 is the SINGLE design margin in the v2 controllers, applied to the friction
circle where the .tex (E56) intends a factor of safety -- NOT to velocities or to
K_max separately. The PID's vcmd_limits consumes this same gate at the same s; the two
differ only in what they do with the result (this distributes a CEILING across axes via
alloc_ratio; the PID scales a DEMAND VECTOR by a scalar gamma).

Two properties worth knowing when reading the output:
  - The ceiling collapses toward zero as V_y approaches V_y_crit = 0.63 m/s,
    where the steady strafe requirement alone fills the circle.
  - F_perp3 VANISHES when V_y = h*psidot ~ 0.235*psidot, so a combined
    lateral+yaw maneuver can load the binding wheel LESS than pure strafe. The
    worst case is therefore not "both axes at maximum".

A cheaper alternative to steps 1-2 is the 729-node velocity-grid lookup that
accompanies the analytical-limits notes; it is velocity-only, so it cannot
include step 3.
"""
function kmax_schedule(lim::PhysicalLimits, V_y::Real, psidot::Real,
                       W_ff::SVector{3,Float64}) end

"""
    lambda_schedule(e, edot, lam_max, v_max) -> Tuple{Float64,Float64}

Saturation-aware sliding-surface slope, replacing `_get_dynamic_lambda`.

On the surface s = edot + lam*e = 0 the error decays as exp(-lam*t), so lam is
1/tau for the tracking time constant. The slope is capped by what the actuator can
actually deliver: the surface demands a corrective velocity lam*|e|, so

    lam(e) = min(lam_max, v_max / |e|)          HYPERBOLIC in the error

This replaces the v1 Gaussian schedule
lam = lam_min + (lam_max - lam_min)*exp(-mu*e^2). The Gaussian form has the right
INTENT (soften the surface at large error to avoid demanding unachievable
corrective velocity) but its lam_min = 0.1 corresponds to a demand 60x below the
platform's ~0.6 m/s capability -- correct only at an error of 6 metres. The
hyperbolic form derives the softening from the saturation limit instead of a free
parameter, removing lam_min, mu_xy and mu_psi from the search.

Returns (lam, lam_dot). lam_dot is required by the equivalent-control terms
(alpha_eq, Ax_eq, Ay_eq) exactly as in the v1 law; differentiate the min() branch
that is active, and guard |e| -> 0 where the hyperbolic branch is inactive.
"""
function lambda_schedule(e::Real, edot::Real, lam_max::Real, v_max::Real) end

"""
    decay_parameters(K_max_base, K_floor, tau_relax, tau_ceiling, decay_k) -> NamedTuple

Gain-decay parameters. The law has TWO decay terms with COMPLEMENTARY roles, and
after the eps fix below they act in DISJOINT regimes -- so both must be sized.

    sigma term :  -sigma * (K - 0.95*K_floor) * exp(decay_k*(1 - s^2/(9*eps_axis^2)))
    cubic term :  -c * (K/K_max)^3

REQUIRED FIX -- use each axis's OWN boundary layer in the exponential. v1 uses
eps_psi in all three channels, including x and y which switch on eps
(ss_x = tanh(s_x/eps)). The leak-off scale must match the axis's own boundary
layer, because |s| < eps is what "converged on this surface" means for that axis.
Use eps for x and y, eps_psi for psi.

    9*eps^2     = 0.00276  (eps = 0.0175)   -> sigma collapses by |s| ~ 0.15
    9*eps_psi^2 = 0.0576   (eps_psi = 0.08) -> collapses by |s| ~ 0.7

After the fix the sigma term is a CONVERGED-STATE leak: it relaxes K toward the
floor while tracking well, and switches off while the controller is still fighting
a large error. That is correct -- do not strip authority still in use. The bound
DURING an event is _smooth_bound_v2 (the anti-windup), not the leakage: K climbs
to the ceiling, holds, then relaxes once |s| falls back inside the boundary layer.

SIZING -- c must be PROPORTIONAL TO RANGE, not uniform:

    sigma_i = 1 / (tau_relax * exp(decay_k))
    c_i     = (K_max_base_i - K_floor_i) / tau_ceiling

Why not a uniform c: dK = -c at the ceiling regardless of K_max, but the distance
to travel is the axis RANGE, and those differ 8.5x (x,y: 5.1; psi: 43.5). Under the
v1 coefficients (0.1, 0.3, 0.5) scaled by any common factor f the decay times are
51/f : 17/f : 87/f -- no single f equalizes them. Range-proportional c keeps
tau_ceiling as ONE design parameter while giving equal decay times per axis.

    at tau_ceiling = 0.5 s :   c = (10.2, 10.2, 87)   vs v1 (0.1, 0.3, 0.5)

ENGAGEMENT PROFILE -- the cubic is top-heavy by construction, so it does not switch
on abruptly:

    K/K_max :  0.23(floor)   0.5     0.7     0.9     1.0
    strength:     1.2%      12.5%    34%     73%    100%

Near the floor it contributes almost nothing; it crosses over with the sigma term
at about half the ceiling and dominates only in the top third. sigma handles gentle
relaxation in the lower half, the cubic is a progressive barrier in the upper half.

SIDE EFFECT to compensate: the cubic acts at all K, so it competes with sigma near
the floor and depresses the resting equilibrium below 0.95*K_floor (about -8% at
c=10, -33% at c=100 on x). Set K_x0/K_y0/K_psi0 to the TARGET equilibrium rather
than assuming it equals K_floor, and verify the resting K numerically.

Both time constants are design choices and must be reported, not hidden. Equal
decay times across axes is a reasonable default, but yaw arguably should recover
faster since its surface time constant is 3-5x shorter than translation -- decide
that deliberately rather than inheriting it.
"""
function decay_parameters(K_max_base::SVector{3}, K_floor::SVector{3},
                          tau_relax::Real, tau_ceiling::Real, decay_k::Real) end

"""
    _smooth_bound_v2(K, K_max) -> Float64

Anti-windup growth gate, replacing ControllerMod._smooth_bound. Same tanh shape,
but the knee is a FRACTION of the ceiling rather than an absolute offset:

    v1 :  0.5 - 0.5*tanh( K - (K_max - 2.0) )      <- absolute -2.0
    v2 :  0.5 - 0.5*tanh( K - 0.98*K_max )         <- fractional knee

WHY: the absolute -2.0 places the knee at a different RELATIVE position on every
axis once K_max is physically derived --

    K_max_x   = 6.6   ->  knee at 4.6   =  70%   of ceiling
    K_max_psi = 56.7  ->  knee at 54.7  =  96.5% of ceiling

so growth on x would be halved at 70% of the ceiling, throttling adaptation long
before the axis reaches the authority derived for it. This was invisible in v1 only
because K_max was uniformly large: at (150,150,300) the knee sat at 98.7/98.7/99.3%,
consistent by accident. Making K_max physical breaks that accident; 0.98 restores
the original design intent on every axis.

Note the tanh WIDTH remains absolute (~1 unit of K), so the transition spans about
15% of the range on x but only ~2% on psi. That is acceptable -- a sharper gate on
yaw behaves closer to a hard clamp, and the function stays C-infinity either way --
but if a softer yaw gate is wanted later, scale the tanh argument by K_max too.

_smooth_bound lives in the never-edited hybrid_ctrl/controllers.jl, so this is a new
function in ControllerV2Mod rather than a modification.
"""
function _smooth_bound_v2(K::Real, K_max::Real) end

"""
    Main.ControllerMod.asmc_wrench!(bus, xhat, ref, params, asmc::ASMCControllerV2, dt; mode=:pose)

Adaptive sliding-mode task-space wrench. Same contract as the v1 method.

Differences from ControllerMod.asmc_wrench!:
  - surfaces use lambda_schedule instead of _get_dynamic_lambda
  - the gain ceiling is kmax_schedule(...) evaluated per tick, not a constant
  - the ceiling is applied as a LAZY clamp (see below)
  - the growth gate is _smooth_bound_v2 (fractional 0.98*K_max knee)
  - decay sigma and cubic coefficients come from decay_parameters
    (tau_relax and tau_ceiling respectively)

LAZY CLAMP -- deliberate design choice. When the scheduled ceiling falls below the
current K, do NOT force K down. Instead raise the working ceiling:

    K_max_eff = max(K, K_max_sched)

This avoids a step discontinuity in commanded torque, and it is self-correcting:
with K_max_eff = K the cubic leakage term (K/K_max)^3 = 1 sits at its MAXIMUM, so
the scheme automatically applies the strongest available kickback while
_smooth_bound(K, K_max_eff) ~ 0.018 halts further growth. This is a soft
(projection-style) bound rather than a hard clamp.

Consequence to be aware of: K may exceed the deliverable ceiling for as long as the
leakage takes to remove it, during which the command saturates. That duration is
tau_ceiling, since after the eps fix the cubic is the only decay acting during an
event. That is why decay_parameters matters.
"""
function Main.ControllerMod.asmc_wrench!(bus, xhat, ref, params,
                                         asmc::ASMCControllerV2, dt; mode::Symbol=:pose) end
```

---

## 7. Data Flow

### 7.1 Derived constants (computed once, at construction)

| Quantity | Value | Source |
|---|---|---|
| `a_cap` | (2.93, 2.86, 9.62) | .tex (E53) friction-circle intercepts |
| `m̃` | 45.0 kg | `m_s + 4J_w/R²`, Table 1 |
| `I_ψ` | ≈5.89 kg·m² | from `I_ψ/I_s ≈ 1.31`, Table 1 — **verify against `params`** |
| capability wrench | (6.59, 6.43, 56.7) N·m | `capability_wrench` |
| **alloc ratio** | **1 : 1 : 8.6** | normalized capability |
| **`K_max_base`** | **(6.6, 6.6, 56.7) N·m** | free circle, zero feedforward |
| **`K_floor`** | **(1.5, 1.5, 13.2) N·m** | measured switching demand, same ratio |
| adaptation range | **≈4.3×** | `K_max_base / K_floor` |
| `κ` | 38.9 | (E54) |
| `μN₃` | 34.8 N | (E54) binding wheel |
| `lever` | 0.130 | `R/(l+h)` |

`K_floor` traces to the measured switching-demand statistics on the stored dataset
(per-sample `|Msat − M_eq|`, p50 ≈ 0.294 N·m per wheel, with the working budget set
at 1.2 N·m per wheel). Convert through the allocation:
`0.25·(K_x + K_y + lever·K_ψ) = 0.7795·c`, so `c = 1.2/0.7795 ≈ 1.54`.

**Report `K_floor` as a percentile of measured switching demand**, computed properly
per-sample — not as "0.6 doubled", which traces back to an invalid subtraction of
absolute quantiles.

Sanity check to include in the validation: `K_max_base` costs
`0.7795 × 6.59 ≈ 5.14 N·m` per wheel against a `τ_max ≈ 9.53 N·m` motor limit, so the
switching term at full authority uses ~54% of motor torque, leaving ~46% for
feedforward. That is tight, and it is exactly why step 3 of `kmax_schedule`
(subtracting the feedforward draw) matters rather than being a refinement.

### 7.2 Per-tick flow

1. Compute pose errors and `ė` as in v1.
2. **`lambda_schedule`** per axis → `(λ, λ̇)`, feeding the surfaces `s` and the
   equivalent-control terms unchanged in structure.
3. Compute `M*_eq` (feedforward) exactly as v1.
4. **`kmax_schedule(lim, V̂_y, ψ̂̇, W_ff)`** → per-axis ceiling. Low-pass its inputs
   (`kmax_lpf_tau`) — see §9.
5. **Lazy clamp**: `K_max_eff = max.(bus.K, K_max_sched)`.
6. Gain update as v1, but with `K_max_eff` in `_smooth_bound_v2` and the derived
   `decay_sigma` on the σ term (the dominant decay path), pulling toward
   `0.95·K_floor`; the cubic term stays as a secondary near-ceiling bite.
7. Switching wrench `−K.*tanh.(s./ε)`, summed with `M*_eq`, returned.

### 7.3 What moves in and out of the search space

```
REMOVED (derived):   K_max_x, K_max_y, K_max_psi
                     lam_x_min, lam_y_min, lam_psi_min, mu_xy, mu_psi
                     K_x0, K_y0, K_psi0            (= K_floor, derived)
                     leakage coefficients 0.1/0.3/0.5

ADDED   (tuned):     lam_x_max, lam_y_max, lam_psi_max

RETAINED(tuned):     gamma_x, gamma_y, gamma_psi
SPECIFIED:           eps, eps_psi (from measured noise), tau_relax, tau_ceiling
```

Six tuned dimensions, spanning surface slopes (performance) and adaptation rates
(robustness) — the directions the objective is actually sensitive to.

**Testable prediction to record before running:** with `λ_x` free, `gamma_x` and
`gamma_y` should tighten from their v1 cv of ~55% toward the ~8% already achieved by
`gamma_psi`. The mechanism is that a stiffer surface makes switching authority
matter translationally, giving those parameters an interior optimum they currently
lack. **If they stay flat, this hypothesis is wrong** and the don't-care conclusion
stands — record the prediction either way so the result is falsifiable rather than
post-hoc.

---

## 8. Implementation Sequence

1. **`PhysicalLimits` + `capability_wrench`** — leaf, pure. Validate the derived
   capability against the .tex intercepts and check `I_ψ` against `params` before
   anything depends on it.
2. **`lambda_schedule`** — leaf. Validate the `λ·|e| ≤ v_max` property and the
   `λ̇` derivative numerically against finite differences.
3. **`decay_parameters` and `_smooth_bound_v2`** — leaves. Apply the eps fix (eps on
   x/y, eps_psi on ψ) and validate each regime separately by integrating the gain law
   in isolation: the σ path at small |s|, the cubic path at |s| large enough that σ
   is off. Check the `_smooth_bound_v2` knee sits at 98% of the ceiling on all axes.
4. **`kmax_schedule`** — depends on #1. Validate against the table in §10.
5. **`ASMCControllerV2` struct** — depends on #1–#4.
6. **`asmc_wrench!` method** — depends on all. Smoke against a single trajectory
   before touching the tuner.
7. **`tune_controller_v2.jl`**: `ASMC_SPACE_V2` and the `build_controller` branch.
8. **Decide the `K` initial condition.** `bus.K` currently resets to **zeros**
   (`bus.jl:55`), so the controller starts every trajectory with no switching
   authority and adapts up. Note `K_x0` in v1 was never the initial value — it is the
   σ-leakage set-point, despite the name. Initializing to `0.95·K_floor` removes the
   startup transient, but that transient is one of the few places `gamma` is
   observable. Decide deliberately and record which was chosen.

---

## 9. Numerical, Real-Time & Stability Considerations

*(Domain-adapted — this is a hard-real-time Julia control law, not a learning task.)*

- **1 kHz hot path, zero allocation.** `SVector`/`SMatrix` only. `kmax_schedule` adds
  one `sqrt` and a handful of flops per tick — fine — but must not allocate.
- **The schedule runs on ESTIMATED states.** `V̂_y` and `ψ̂̇` carry estimator noise
  straight into the ceiling, and a jittering ceiling produces gain jitter. Low-pass
  the schedule inputs (`kmax_lpf_tau`), and given the ESKF findings, treat this as a
  first-order concern rather than a refinement.
- **Guard the schedule's degenerate cases.** `F_par_avail` goes to zero as
  `|F_⊥,3| → μN₃`; clamp the ceiling to a small positive floor so the controller never
  loses all switching authority, and log when that guard fires — it means the platform
  is at its physical envelope, which is a result worth surfacing rather than hiding.
- **`lambda_schedule` at `e → 0`.** The hyperbolic branch diverges; the `min` with
  `lam_max` handles it, but the `λ̇` expression must switch branches cleanly. A
  discontinuous `λ̇` injects a torque spike through the equivalent-control terms.
- **Lazy clamp is a projection with a time-varying set.** The standard adaptive SMC
  Lyapunov argument assumes a *constant* gain bound. A state-dependent one makes this a
  projection-based adaptive law — well established (Ioannou & Sun parameter
  projection), but it must be cited rather than inheriting the fixed-bound proof. Flag
  this wherever a stability claim appears.
- **Saturation is not benign for an adaptive law.** Under saturation `|s|` stays large,
  so `dK` stays positive and `K` ratchets up while producing no additional torque —
  windup in the adaptation. `_smooth_bound` *is* the anti-windup clamp and `K_max` is
  its threshold; that, not "robustness reserve", is what `K_max` is for. Setting it
  above the deliverable disables it.
- **Determinism.** No RNG in this path. Keep BLAS single-threaded.

---

## 10. Success Criteria

- [ ] `capability_wrench` returns ≈(6.59, 6.43, 56.7) N·m; ratio ≈1 : 1 : 8.6
- [ ] `I_ψ` verified against `params` rather than inherited from the .tex ratio
- [ ] **`kmax_schedule` reproduces the binding-wheel table:**

      | V_y | ψ̇ | expected F_∥,avail | % of circle |
      |---|---|---|---|
      | 0 | 0 | 34.8 | 100% |
      | 0.4 | 0 | 31.1 | 89% |
      | 0.6 | 0 | 25.8 | 74% |
      | 0 | 2.0 | 29.6 | 85% |
      | 0.4 | 2.0 | 34.7 | 99.7% |

- [ ] The `V_y = h·ψ̇` cancellation is reproduced (last row above) — confirms the sign
      convention in the gate
- [ ] `lambda_schedule` satisfies `λ·|e| ≤ v_max` everywhere, and `λ̇` matches a finite
      difference of `λ` to within tolerance across the branch switch
- [ ] The σ-exponential uses `eps` on x/y and `eps_psi` on ψ, collapsing at |s| ≈ 0.15
      and ≈ 0.7 respectively — confirming the leak-off scale tracks each axis's own
      boundary layer
- [ ] `decay_parameters` reproduces `tau_relax` at small |s| (σ active) and
      `tau_ceiling` at large |s| (σ off, cubic alone), **equally on all three axes** —
      the check that range-proportional `c` worked
- [ ] `_smooth_bound_v2` knee sits at 98% of `K_max` on every axis, and `K` is observed
      to reach at least ~95% of its ceiling under a demanding trajectory (if it never
      does, the gate is still throttling too early)
- [ ] Resting `K` with no disturbance is measured and matches the intended floor; if it
      sits below, `K_*0` needs raising to compensate for cubic depression
- [ ] Smoke: one trajectory runs to completion, `max(K)` logged per axis, and the
      scheduled ceiling is *reached* at least occasionally — if `K` never approaches it,
      the ceiling is still non-binding and the sizing needs revisiting
- [ ] `n_params(ASMC_SPACE_V2) == 6`
- [ ] Prediction recorded (§7.3) before the tuning run, and its outcome reported
      whichever way it falls

---

## 11. Out of Scope

- **Re-tuning.** This brief changes what is tunable; a later run does the tuning.
- **`hybrid_ctrl/controllers.jl`** — never edited. New behaviour lives in
  `ControllerV2Mod` via dispatch, matching the `MPCControllerV2` precedent.
- **PID and MPC.** `MPCControllerV2` is untouched here.
- **Estimator work** — separate brief; note only that the schedule consumes `x̂`, so
  estimator quality now directly affects the gain ceiling.
- **The yaw load-transfer limit.** §3.5 of the analytical-limits document derives the
  out-of-plane bound for *translational* acceleration only; the yaw equivalent is not
  derived. Since the 1 : 1 : 8.6 allocation puts ~8.6× more authority on yaw, that gap
  is worth closing before the yaw ceiling is trusted at full value — but it is a
  separate analytical task, not part of this implementation.
- **Velocity-mode paths.** Every trajectory runs `:pose`.
