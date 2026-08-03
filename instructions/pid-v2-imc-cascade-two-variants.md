# PID v2 — IMC Cascade Reparameterization, V_cmd Limiting, and the Feedforward Ablation

> **Generated:** 2026-07-29
> **Stack:** Julia 1.12.5 — `code_insights/` project, `hybrid_ctrl_v2/` module tree. No new packages.
> **Scope:** `ControllerV2Mod` extension only. `hybrid_ctrl/controllers.jl` is never edited.
> **Physical constants:** `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex`

---

## 1. Overview

Four changes to the cascade PID, and one new variant:

1. **IMC reparameterization.** 18 searched gains collapse to **3 tuned parameters**
   (`λ_inner` per axis). Every other gain is derived from the plant model.
2. **`V_cmd` limited by the combined friction-circle gate (E54).** The outer loop
   currently emits an unbounded velocity command. A single quadratic gate over
   velocity *and* acceleration together yields a scalar scale factor `γ` that
   preserves the demanded acceleration direction. This is the *only structural*
   change, and it is what makes inner-loop saturation preventable at all.
3. **`I_max` from measured p95** of the wrench the integral must supply — replacing a
   default that permits ~450× the deliverable integral authority.
4. **`Kd = Kd_pos = 0`**, derived rather than assumed.
5. **Two variants for ablation:** `PID-FB` (pure feedback, industrial baseline) and
   `PID-CT` (computed-torque / equivalent-control feedforward, structurally matched to
   the ASMC).

**System-level contract:** `PIDControllerV2` consumes the same
`(bus, xhat, ref, pid, dt; mode)` interface as `PIDController` and returns the same
`SVector{3}` task-space wrench. Its gains are functions of `λ_inner`, the pinned
separation ratio `N`, and the plant's analytical limits.

**Why two variants.** The ASMC's `M_eq` is *equivalent control*, not reference
feedforward — it applies the full model to a surface-corrected acceleration
`a_eq = a_ref − λ·ε̇ − λ̇·ε`. `PID-CT` reproduces that structure with
`a_cmd = a_ref − Kp_pos·ε̇` (identical, with `Kp_pos ↔ λ`). This makes the three
controllers a clean 2-step ablation:

| variant | model | residual handler | the gap measures |
|---|---|---|---|
| PID-FB | ✗ | PI | — (baseline) |
| PID-CT | ✓ | PI | **FB→CT = value of the model** |
| ASMC | ✓ | adaptive switching | **CT→ASMC = value of adaptation** |

Without `PID-CT`, an ASMC-vs-PID comparison conflates the model and the adaptation,
and a reviewer will say so.

---

## 2. Architecture Pattern

**New struct + dispatch extension**, matching the `MPCControllerV2` / `ASMCControllerV2`
precedent already in `ControllerV2Mod`. Define `PIDControllerV2`, then add a method to
the existing generic `Main.ControllerMod.pid_wrench!` dispatching on the new type. Zero
edits to `hybrid_ctrl/controllers.jl`.

The organizing principle is the same as the ASMC brief: **derive, don't tune.** Only
`λ_inner` remains free, because only it represents a genuine performance/robustness
trade. Everything else follows from IMC pole cancellation, the cascade separation, or a
measured statistic.

---

## 3. Technology Constraints

- **Julia 1.12.5**, existing project. **No new packages.**
- `ControllerV2Mod` is included after `tune_controller.jl`; `Main.ControllerMod`,
  `Main.PlantMod`, `Main.Profiles`, `PlatformParams` are in scope.
- Runs at **100 Hz** (`f_pid`) inside the ODE callback — allocation-light, `SVector`
  throughout, but far less hot than the ASMC's 1 kHz path.
- Reuses `PhysicalLimits` and the binding-wheel gate from the ASMC v2 brief. **Build
  that first** — this brief depends on it.
- **Exclusions:** no edits to v1 files; no plant/sensor/estimator changes; no re-tuning
  within this brief; the `N` sensitivity sweep is deferred (§11).

---

## 4. Component Breakdown

### `PIDControllerV2` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** `Base.@kwdef mutable struct`
- **Responsibility:** Cascade-PID parameter carrier with IMC-derived gains and the two
  variant modes.
- **Key fields:**
  - tuned: `lam_inner::SVector{3,Float64}`
  - pinned: `N::Float64` (separation ratio, = 4.0)
  - derived at construction: `Kp`, `Ki`, `Kp_pos`, `I_max` (all `SVector{3}`)
  - fixed: `Kd = zeros(3)`, `Kd_pos = zeros(3)`
  - limits: `vcmd_clamp::SVector{3}`, `use_rate_limit::Bool`
  - variant: `feedforward::Bool` (false = PID-FB, true = PID-CT)
  - state: `prev_e`, `prev_e_pos`, `prev_vcmd`, `initialized`, `pos_initialized`
- **Depends on:** `PhysicalLimits` (ASMC v2 brief)

### `imc_gains` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** function
- **Responsibility:** Derive all cascade gains from `λ_inner` and `N`.
- **Inputs:** `lim::PhysicalLimits`, `lam_inner::SVector{3}`, `N::Real`
- **Outputs:** `(Kp, Ki, Kp_pos)` each `SVector{3,Float64}`
- **Depends on:** `PhysicalLimits`

### `vcmd_limits` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** function
- **Responsibility:** Limit the outer loop's velocity command via the combined
  friction-circle gate (E54) — the structural fix.
- **Inputs:** `lim::PhysicalLimits`, `xhat` (needs `V̂x`, `V̂y`, `ψ̂̇`), `dt`,
  `vcmd_prev::SVector{3}`, `vcmd_raw::SVector{3}`
- **Outputs:** `(V_cmd::SVector{3,Float64}, gamma::Float64, guard_hit::Bool)`
- **Depends on:** `PhysicalLimits`, and the same binding-wheel gate as `kmax_schedule`

### `pose_outer_loop_v2` (`hybrid_ctrl_v2/controllers_v2.jl`)
- **Type:** function
- **Responsibility:** Outer position loop — body-frame error, P-only correction, then
  the E54 feasibility gate via `vcmd_limits`.
- **Inputs:** `xhat`, `ref::PosRef`, `pid::PIDControllerV2`, `lim`, `t`, `dt`
- **Outputs:** `V_cmd::SVector{3,Float64}` (limited)
- **Depends on:** `vcmd_limits`

### `Main.ControllerMod.pid_wrench!` method on `PIDControllerV2`
- **Type:** dispatch extension
- **Responsibility:** The control law; both variants.
- **Inputs:** `bus`, `xhat`, `ref`, `pid::PIDControllerV2`, `dt`; `mode=:pose`
- **Outputs:** `SVector{3,Float64}` task-space wrench
- **Depends on:** all of the above

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl_v2/
│   ├── controllers_v2.jl        # MODIFIED — add PIDControllerV2, imc_gains,
│   │                            #   vcmd_limits, pose_outer_loop_v2, pid_wrench! method.
│   │                            #   MPCControllerV2 / ASMCControllerV2 untouched.
│   └── tune_controller_v2.jl    # MODIFIED — PID_SPACE_V2 (3 dims), build_controller branch
├── docs/
│   └── Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex   # READ ONLY
└── _tmp/
    └── pid_v2_validation.jl     # NEW — the §10 checks
```

---

## 6. Key Interfaces

```julia
"""
    imc_gains(lim::PhysicalLimits, lam_inner::SVector{3}, N::Real)
        -> (Kp, Ki, Kp_pos)

Derive every cascade gain from the inner-loop time constants and the separation ratio.

Per-axis the plant is first order,  m_eff*v̇ + d_eff*v = W , with

    m_eff = (R*m_tilde, R*m_tilde, I_psi)      = (2.25, 2.25, 5.89)
    d_eff = (R*176,     R*396,     38)         = (8.8,  19.8, 38)
    tau_open = m_eff/d_eff                     = (0.256, 0.114, 0.155) s

IMC-PI for G(s) = (1/d)/(tau*s + 1) with desired closed-loop time constant lambda:

    Kp = m_eff / lambda        Ki = d_eff / lambda        Kd = 0

The controller zero then sits at -Ki/Kp = -d/m = -1/tau_open, exactly on the plant
pole. After cancellation L(s) = 1/(lambda*s) and the closed loop is 1/(lambda*s+1).
Kd = 0 is a RESULT, not a simplification: IMC prescribes no derivative for a
first-order plant, and the only unmodelled lag is the 100 Hz half-sample (~5 ms,
3-5% of lambda).

Outer loop, on the integrator plant p-dot = v:

    Kp_pos = 1/(N * lam_inner)        Kd_pos = 0

N is the separation ratio and it IS the damping specification: closing the outer
loop on the inner gives

    omega_n = 1/(lam_inner*sqrt(N))        zeta = sqrt(N)/2

so N = 4 is exactly critical damping (the origin of the "3-5x" rule of thumb).
Kd_pos = 0 follows from that -- there is no damping deficit for lead to fix, and
derivative action on a drifting pose estimate is the single worst noise path in the
system.

Note also zeta*omega_n = 1/(2*lam_inner), INDEPENDENT of N: the outer loop cannot
settle faster than ~2*lam_inner however Kp_pos is chosen. If pose tracking is too
slow the fix is a faster inner loop, never a stiffer outer gain.
"""
function imc_gains(lim::PhysicalLimits, lam_inner::SVector{3,Float64}, N::Real) end


"""
    vcmd_limits(lim, xhat, dt, vcmd_prev, vcmd_raw) -> (V_cmd, gamma, guard_hit)

Feasibility-limited velocity command, using the binding-wheel gate (E54) as a SINGLE
COMBINED constraint over velocity and acceleration together.

THIS IS THE ONLY STRUCTURAL CHANGE IN THIS BRIEF. v1's pose_outer_loop emits V_cmd with
no bound at all, so a large pose error commands an arbitrarily large velocity setpoint,
the inner loop saturates, and the integral winds up.

PER-TICK ALGORITHM

  1. Implied acceleration demand -- what the outer loop is actually asking of the plant:

         a_req = (vcmd_raw - vcmd_prev) / dt

     a_req mixes the reference's own acceleration with the position correction. The
     platform cannot distinguish them and neither should the gate.

  2. Demand vector b(v, a_req)  (E55), at the ESTIMATED velocity:

         b_x = m_s*a_x - m*a_Y*alpha - m_s*psidot*V_y - m*a_X*psidot^2
         b_y = m_s*a_y + m*a_X*alpha + m_s*psidot*V_x - m*a_Y*psidot^2 - 110.0*V_y
         b_O = -m*a_Y*a_x + m*a_X*a_y + I_s*alpha + m*psidot*(a_X*V_x + a_Y*V_y) - 2.20*psidot

  3. Binding-wheel gate (E54):

         F_perp3 = kappa*(V_y - h*psidot)                velocity only -- fixed this tick
         F_par3  = 0.354*(b_x + b_y) - 0.918*b_O         affine in a_req
         budget  = (s*mu_N3)^2 - F_perp3^2               s = 0.9

  4. Scale factor -- F_par3 is AFFINE in a_req, so split it and solve one scalar equation:

         F_par3(gamma) = F_par3(0) + gamma*dF     F_par3(0) = velocity-only part
                                                  dF        = acceleration contribution
         solve |F_par3(gamma)| = sqrt(budget)     for gamma in [0, 1]

  5. Apply:   V_cmd = vcmd_prev + gamma * a_req * dt

WHY A SINGLE SCALAR gamma, NOT PER-AXIS CLIPPING: scaling all three components by the
same factor PRESERVES THE DIRECTION of the demanded acceleration -- the platform keeps
heading where the trajectory wants, just less hard. Per-axis clipping distorts the
direction and pushes the platform off the path.

WHY THE GATE IS QUADRATIC AND COMBINED, not a per-axis linear budget:
  - The friction circle is a circle. A linear (diamond) approximation is safe but ~2x
    conservative: at V_y = 50% of its cap the linear form allows 0.40*a_cap where the
    circle allows 0.83*a_cap. That bites hardest on y, the axis with least room.
  - V_y and psidot do NOT add independently -- they enter as (V_y - h*psidot), so a
    combined lateral+yaw maneuver can load the binding wheel LESS than either alone.
    Separate per-axis terms miss that cancellation and reject feasible states.
  - The .tex section 4 states the true velocity set is a superellipse with 1 < n < 2:
    n = 1 is the conservative diamond, n = 2 the circle the physics enforces.

NORMALIZER CAUTION: do NOT use V_y_crit = 0.63 m/s as the velocity scale. That is the
COMBINED steady-strafe point, where F_par is forced equal to F_perp. The pure-F_perp
intercept is mu_N3/kappa = 0.894 m/s. Using 0.63 double-counts.

DEGENERATE CASES
  - budget < 0 (F_perp3 alone exceeds the circle): set gamma = 0, hold V_cmd, return
    guard_hit = true. The platform is already outside its envelope -- a reportable
    finding, not something to silence.
  - vcmd_prev must persist across ticks and initialize to V_ff(0), NOT zero, or tick one
    demands a huge acceleration and gamma collapses spuriously.

RELATION TO THE MAGNITUDE CLAMP: this gate enforces the velocity envelope on its own --
as V_y approaches its intercept, budget -> 0 and V_cmd stops rising. The separate
magnitude clamp (pid.vcmd_clamp) is therefore only a NUMERICAL BACKSTOP against the
positive floor placed on budget, and carries no safety factor of its own. s = 0.9 is the
SINGLE margin in the design, applied to the circle where the .tex intends it.

SHARED MACHINERY: steps 3-4 use the same binding-wheel gate as the ASMC's kmax_schedule.
Build it once in PhysicalLimits. The two controllers differ only in what they do with the
result: the ASMC distributes a CEILING across axes via alloc_ratio, the PID scales a
DEMAND VECTOR by gamma.
"""
function vcmd_limits(lim::PhysicalLimits, xhat, dt::Real,
                     vcmd_prev::SVector{3,Float64},
                     vcmd_raw::SVector{3,Float64}) end


"""
    imax_from_measured(Ki::SVector{3}, feedforward::Bool) -> SVector{3,Float64}

Anti-windup bound, sized from the MEASURED p95 of the wrench the integral must supply.

I_max clamps the integrator STATE (units: metres -- an accumulated velocity error),
not the torque. The integral's contribution to the wrench is Ki*I_pid, so the bound
that matters is Ki*I_max.

    PID-FB : integral supplies drag + Coriolis + disturbance
             -> p95(|Msat|)          = 5.279 N*m/wheel
    PID-CT : feedforward supplies the model; integral supplies the residual only
             -> p95(|Msat - M_eq|)   = 2.389 N*m/wheel

Convert per-wheel torque to task-space wrench on the capability-proportional
allocation (1 : 1 : 8.6, the same ratio as the ASMC):

    per_wheel = 0.25*(W_x + W_y + lever*W_psi) = 0.7795*c   ,  W = c*(1, 1, 8.6)

CROSS-CHECK worth reproducing: the measured p95 route gives c = 5.279/0.7795 = 6.77
task-space for PID-FB, against an independent analytical steady-demand estimate of
d_eff*v_max + disturbance = 8.8*0.6 + 1.5 = 6.78. Agreement to 0.2% from two
unrelated routes -- if the implementation does not reproduce that, the unit
convention is wrong somewhere.

v1 default I_max = (50, 50, 30) permits Ki*I_max ~ 2935 N*m against a wrench
capability of ~6.6. That is the windup defect this replaces.
"""
function imax_from_measured(Ki::SVector{3,Float64}, feedforward::Bool) end


"""
    pose_outer_loop_v2(xhat, ref, pid, lim, t, dt) -> SVector{3,Float64}

Outer position loop. Same body-frame error construction as v1's pose_outer_loop
(world error rotated by R(psi)^T, smooth-wrapped heading), then:

    V_cmd_raw = V_ff - Kp_pos .* e_body            (Kd_pos = 0, no derivative term)
    V_cmd     = vcmd_limits(...)                    clamp + rate limit

Kp_pos is MANDATORY and cannot be replaced by Kd_pos: the outer plant is a pure
integrator, so derivative-only feedback gives L(s) = Kd (a constant), leaving
S(0) = 1/(1+Kd) != 0 -- a permanent position offset. Only proportional action places
a pole off the origin and gives zero steady-state position error.

Useful identity for anyone reading the closed loop: the inner-loop error becomes

    e = v - V_cmd = eps_dot + Kp_pos*eps

i.e. the inner error IS a sliding surface on the position error, with Kp_pos playing
the role of the ASMC's lambda. The full cascade characteristic polynomial factors as

    m*(s + 1/tau_open) * (s^2 + s/lam_inner + Kp_pos/lam_inner)

-- the cancelled plant pole, plus the outer second-order pair.
"""
function pose_outer_loop_v2(xhat, ref, pid::PIDControllerV2, lim::PhysicalLimits,
                            t::Real, dt::Real) end


"""
    Main.ControllerMod.pid_wrench!(bus, xhat, ref, pid::PIDControllerV2, dt; mode=:pose)

Cascade PID wrench, both variants.

    e   = xhat[1:3] - V_cmd
    I   = clamp(I + dt*e, +/- I_max)
    W   = -(Kp.*e + Ki.*I)                        [PID-FB]
    W   = M_eq_cmd - (Kp.*e + Ki.*I)              [PID-CT]

with the PID-CT feedforward built from the COMMANDED velocity, not the reference:

    a_cmd  = a_ref - Kp_pos .* eps_dot
    M_eq_cmd = m_eff .* a_cmd + d_eff .* V_cmd + Coriolis(psi_dot, V_hat)

Building it from V_cmd rather than V_ref is deliberate: it reproduces the ASMC's
surface-corrected equivalent control a_eq = a_ref - lambda*eps_dot - lambda_dot*eps
(identical with Kp_pos <-> lambda), so the CT variant is a structurally matched
sibling of the ASMC rather than a differently-shaped controller.

Reuse the Coriolis/COM terms from ControllerMod's asmc_wrench! expressions rather
than re-deriving them; they are the same physical terms.

Kd is zero in both variants -- keep the field for schema compatibility but do not
wire a derivative path.
"""
function Main.ControllerMod.pid_wrench!(bus, xhat, ref, pid::PIDControllerV2, dt;
                                        mode::Symbol=:pose) end
```

---

## 7. Data Flow

### 7.1 Derived constants

| Quantity | x | y | ψ | Source |
|---|---|---|---|---|
| `m_eff` | 2.25 | 2.25 | 5.89 | `R·m̃`, `I_ψ` |
| `d_eff` | 8.8 | 19.8 | 38 | `R·D_eq`, `D_eq` (.tex §2) |
| `τ_open` | 0.256 | 0.114 | 0.155 | `m_eff/d_eff` |
| `a_cap` | 2.93 | 2.86 | 9.62 | .tex (E53) |
| `W_cap` | 6.59 | 6.43 | 56.7 | `m_eff·a_cap` |
| `vcmd_clamp` | 4.55 | 0.63 | 3.80 | numerical backstop only — **no FOS** |
| `s` (circle margin) | — | 0.9 | — | the single design margin, applied in E54 |
| `μN₃` | — | 34.8 N | — | .tex (E54) binding wheel |
| `κ` | — | 38.9 | — | .tex (E54) |

### 7.2 The three tuned parameters and their windows

```
lam_inner ∈ [0.05, tau_open]     per axis

  x :  [0.05, 0.256]     ← widest genuine freedom
  y :  [0.05, 0.114]
  psi: [0.05, 0.155]
```

Upper bound is `τ_open` and is genuinely **per-axis**: beyond it the closed loop is
slower than doing nothing. Lower bound is the **sample-rate** bound (≈5 sample
periods at 100 Hz) and is **uniform** — all the per-axis structure lives in the
upper bound.

Two other candidate lower bounds were checked and are slack. Recorded here so the
next reader does not have to re-derive them:

- **Saturation** (`λ ≥ e_max/a_cap`) does not apply once the `V_cmd` rate limit is
  in place: the inner loop lags the ramping command by `λ·a_avail`, so the demanded
  acceleration is `a_avail` regardless of `λ`. Self-consistent, and it is the reason
  the rate limit is worth adding. (Without the rate limit the bound is circular
  anyway — the properly derived `e_max = a_cap·τ_open` gives `λ ≥ τ_open`, colliding
  with the upper bound and collapsing the window to a point.)
- **Noise amplification** *is* genuinely per-axis, since `Kp = m_eff/λ`. At
  `λ = 0.05` the P-term wrench noise is **3.3% of `W_cap` on x versus 0.6% on ψ** —
  x is ~5× more noise-sensitive relative to its own capability. But capping wrench
  noise at 5% of capability gives `λ_x ≥ 0.033`, `λ_ψ ≥ 0.006`, both below the
  sample-rate bound. **Revisit this if `f_pid` is ever raised**, since the
  sample-rate bound falls with rate while the noise bound does not.

`N = 4.0` is **pinned** for all runs in this brief.

### 7.3 Derived gains at mid-window `λ = (0.15, 0.11, 0.12)`

```
Kp      = m_eff/λ         = ( 15.0,  20.5,  49.1)     v1 default (20, 25, 15)
Ki      = d_eff/λ         = ( 58.7, 180.0, 317.0)     v1 default ( 1,  1, 0.5)   ~100x low
Kd      = 0                                            v1 default ( 2,  2,  1 )
Kp_pos  = 1/(4λ)          = (  1.67,  2.27,  2.08)     v1 default ( 1,  1,  2 )
Kd_pos  = 0                                            v1 default (0.5,0.5, 1 )

I_max (PID-FB)            = (  0.115, 0.038, 0.184)    v1 default (50, 50, 30)   ~450x high
I_max (PID-CT)            = (  0.052, 0.017, 0.083)
```

### 7.4 Predicted outcomes — record before running

- **`PID-FB` velocity floor ≈ `λ·a`.** From the ramp analysis, `e_ss = a·d/Ki = λ·a`.
  At `λ ≈ 0.15` and typical trajectory accelerations `a ≈ 0.03 m/s²`, that is
  **4.5 mm/s** — matching the 3–6 mm/s floor measured in v1. Note this floor is
  independent of `Kp`, which explains why the v1 tuner pinned `Kp_psi` at its 800
  ceiling on every seed: it was pushing the one knob that could not help, because
  `Ki`'s useful range sat outside the search box.
- **`PID-CT` should not show that floor** — the drag ramp is cancelled by the
  feedforward, leaving error set by disturbance rejection.
- **`λ_y` should be tightly identified** across seeds (its window is only 14% wide),
  while `λ_x` has the most genuine freedom. If `λ_y` comes out with *high* cv, the
  window derivation is wrong.

---

## 8. Implementation Sequence

0. **Prerequisite:** `PhysicalLimits` and the binding-wheel gate from the ASMC v2
   brief must exist. This brief consumes both.
1. **`imc_gains`** — leaf, pure. Validate the pole-cancellation identity: with the
   derived gains, `Ki/Kp` must equal `1/τ_open` exactly on every axis.
2. **`imax_from_measured`** — leaf. Reproduce the 0.2% cross-check in §6 before
   trusting the unit conversion.
3. **`vcmd_limits`** — depends on the gate. Validate the `a_avail` table in §10.
4. **`pose_outer_loop_v2`** — depends on #3.
5. **`PIDControllerV2` struct** — depends on #1–#4.
6. **`pid_wrench!` method, PID-FB first.** Smoke one trajectory before adding the
   variant branch.
7. **PID-CT branch** — add the `M_eq_cmd` feedforward. Verify it reproduces PID-FB
   exactly when the feedforward is forced to zero.
8. **`PID_SPACE_V2`** (3 dims, per-axis bounds from §7.2) and the `build_controller`
   branch in `tune_controller_v2.jl`.

---

## 9. Numerical, Real-Time & Robustness Considerations

- **100 Hz loop**, so far less hot than the ASMC path — but still no allocation.
  `SVector` throughout.
- **The rate limiter needs state.** `vcmd_prev` must persist across ticks and be reset
  on initialization, exactly as `prev_e` / `prev_e_pos` are. A stale `vcmd_prev` at
  trajectory start produces a spurious first-tick rate violation.
- **Pole cancellation is inexact by construction.** `d_eff` comes from the *no-slip*
  `D_eq`, while the real plant has LuGre slip, so the PI zero lands near — not on —
  the plant pole, leaving a slow residual mode. Practical consequence: **do not chase
  very small λ.** Aggressive cancellation designs are the most sensitive to this
  mismatch, and yours is mismatched by design.
- **The cancelled mode is not gone.** It survives in the *disturbance* response even
  when invisible from the reference. Expect it to show up under noise and slip.
- **`a_avail` can go negative** (y axis above ~0.5 m/s). Clamp at a small positive
  floor and **log when that guard fires** — it means the platform is at its physical
  envelope, which is a reportable finding rather than something to hide.
- **The outer loop's low bandwidth is a feature.** `Kp_pos ≈ 2` means `λ_outer ≈ 0.5 s`,
  which low-pass filters position-estimate noise. This is why `Kd_pos = 0` matters:
  derivative action is the one thing that would undo that filtering, on the noisiest
  and most drift-prone signal in the system.
- **Integral saturation should be rare.** `I_max` rests on `D_eq` (no-slip, so it
  under-represents slip dissipation) and on a disturbance estimate from a dataset with
  known bound issues. It is a bound, not a performance gain, so approximate is
  acceptable — but log the saturated fraction and check it is small.
- **Determinism.** No RNG in this path; BLAS single-threaded.

---

## 10. Success Criteria

- [ ] **Pole cancellation:** `Ki/Kp == 1/τ_open` exactly on all three axes for any λ
- [ ] **`ζ = √N/2`:** with the derived `Kp_pos`, the outer-loop closed-loop damping
      measures 1.0 at `N = 4` (step-response overshoot ≈ 0)
- [ ] **I_max cross-check:** the measured-p95 route and the analytical steady-demand
      route agree to within ~1% on the x axis (6.77 vs 6.78 task-space)
- [ ] **Gate reproduces the worked example.** At `V̂ = (0.5, 0.3, 0.5)` with
      `a_req = (5, 5, 0)`: `F_⊥,3 ≈ 7.10 N`, `√budget ≈ 30.5 N`,
      `F_∥,3(0) ≈ −9.4 N`, `γ ≈ 0.33`, giving
      `V_cmd = (0.517, 0.317, 0.5)` instead of the raw `(0.55, 0.35, 0.5)`
- [ ] **Cancellation captured:** `F_⊥,3` vanishes at `V_y = h·ψ̇ ≈ 0.235·ψ̇` — the
      check that catches a sign error in the gate
- [ ] **Direction preserved:** when `γ < 1` the applied acceleration is parallel to
      `a_req` (per-axis clipping would break this)
- [ ] **`γ = 1` in normal tracking.** Log the fraction of ticks with `γ < 1` and with
      `guard_hit`; both should be small outside transients and initialization
- [ ] **Variant equivalence:** `PID-CT` with the feedforward forced to zero reproduces
      `PID-FB` bit-identically
- [ ] **`n_params(PID_SPACE_V2) == 3`**
- [ ] Integral saturated fraction logged and small (< a few %) on the training set
- [ ] The §7.4 predictions recorded before the tuning run and reported afterwards
      **whichever way they fall**

---

## 11. Out of Scope

- **The `N` sensitivity sweep.** Deferred by decision: run it *after* the FB-vs-CT
  ablation settles which PID variant goes in the paper, then sweep
  `N ∈ {2, 3, 4, 6, 9}` (ζ ∈ [0.71, 1.5]) with the tuned λ's frozen. Five evaluations,
  one dimension. Worth doing because the model mismatch pushes ζ in *both* directions —
  slip reduces loop gain (raising ζ) while LuGre bristle lag adds phase (lowering it) —
  and which dominates is empirical.
- **Re-tuning.** This brief changes what is tunable; a later run tunes it.
- **`hybrid_ctrl/controllers.jl`** — never edited.
- **ASMC and MPC** — separate briefs. `PhysicalLimits` is shared, not duplicated.
- **Relay auto-tuning** as an alternative baseline — discussed, not adopted.
- **Velocity-mode paths.** Every trajectory runs `:pose`.
