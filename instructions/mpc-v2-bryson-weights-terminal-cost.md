# MPC v2 — Bryson-Normalized Weights, U_eq-Centered Effort, Riccati Terminal Cost

> **Generated:** 2026-07-31
> **Stack:** Julia 1.x, OSQP.jl, StaticArrays, LinearAlgebra, SparseArrays, MatrixEquations.jl (new)
> **Scope:** Controller (`hybrid_ctrl_v2/controllers_v2.jl` MPC section) + tuner search space (`tune_controller_v2.jl`)
> **Companion briefs:** `asmc-v2-physical-gain-bounds.md`, `pid-v2-imc-cascade-two-variants.md`

---

## 1. Overview

`ControllerMod.mpc_wrench!` is a condensed receding-horizon QP over per-wheel motor
voltages. It works, but **every one of its 11 tuned parameters was searched as a free
number with no physical referent**, and the search space had an exact flat direction in
it. This brief replaces the search with derivation.

Four changes, in decreasing order of how much they matter:

1. **`R` is re-centered on a feedforward voltage `U_eq`.** v1 penalizes `‖U‖²` — the
   *absolute* voltage, including the voltage the trajectory physically requires to
   overcome back-EMF and drag. Since this MPC has **no integral action anywhere**, that
   penalty produces a genuine steady-state droop, structurally identical to
   proportional-only control. It is why `R_scale`'s box had to be pinned at `[1e-3, 1]`:
   the parameter was crippled, not tuned. Penalizing `‖U − U_eq,k‖²` removes the bias
   and gives `R` a well-posed job (see §6, `bryson_R`).

2. **A Riccati terminal cost `P` is added.** Finite-horizon MPC with no terminal cost has
   **no stability guarantee**, even with a perfect model. `Np_pose = 15` at 100 Hz is
   0.15 s — under one open-loop time constant on x (`tau_open_x = 0.256 s`). The DARE
   solution makes the truncated cost equal the infinite-horizon cost, so the horizon
   stops being a myopia parameter and becomes purely a constraint/preview parameter.

3. **`Q_pose` and `R` are derived from the tuner's own tolerances (Bryson's rule)**
   rather than searched. This has a side effect worth naming explicitly: the v1 search
   space contained an **exact one-dimensional flat direction**. Scaling `(Q, R, S)` by
   any `c` scales `H` and `f` both linearly, so the objective becomes `c·J_orig` — same
   argmin, same constraints, bit-identical trajectory. In the log-parameterized box the
   flat set is a straight line along the all-ones direction spanning most of the box.
   dxNES gets zero ranking signal along it, never shrinks that axis, and burns
   evaluations. Worse, the flatness is broken only *numerically* — by the absolute
   `1e-6*I` regularizer (which does not scale with `c`) and by OSQP's absolute stopping
   tolerances — so the optimizer can lock onto solver artifact rather than control
   performance. That is a plausible route to the observed outcome (67 evaluations,
   converged to a **worse** basin than the coarse search). Bryson does not *fix* this
   degeneracy; it **dissolves** it, because weights with physical units have no free
   multiplier left.

4. **`Np_pose` moves from "never searched" to a diagnostic sweep**, and the search space
   collapses to a single parameter.

**Search dimension: 11 → 1.**

---

## 2. Architecture Pattern

**Physical derivation + minimal residual search**, matching the ASMC v2 and PID v2 briefs.

Each v2 controller now derives its parameters from an independent physical source and
leaves a small, interpretable residual search:

| controller | derived from | searched |
|---|---|---|
| ASMC v2 | friction circle (E53/E54/E56) | 6 (`lam_*_max`, `gamma_*`) |
| PID v2 | IMC pole cancellation on `m_eff/d_eff` | 3 (`lam_inner`) |
| **MPC v2** | **Bryson normalization on `TOL` + DARE** | **1 (`S_scale`)** |

This is a **methodological** point, not just a tuning convenience. The v1 comparison
pitted a physically-parameterized ASMC against a 15-parameter PID and an 11-parameter
MPC with a flat direction in it. Deriving all three from physics and leaving each a small
residual search is what makes the three-way comparison defensible in the paper.

Implementation follows the established v2 pattern exactly: `hybrid_ctrl/controllers.jl`
is **NEVER edited**. `MPCControllerV2` already exists in `hybrid_ctrl_v2/controllers_v2.jl`
with a `P_terminal::Matrix{Float64}` field and a `Main.ControllerMod.mpc_wrench!` method
dispatching on it. This brief extends that struct and that method.

---

## 3. Technology Constraints

- **Julia:** 1.x, existing `Project.toml`/`Manifest.toml` environment
- **New dependency:** `MatrixEquations.jl` — for `ared` (discrete algebraic Riccati).
  `ControlSystemsBase.dare` is an acceptable substitute if already in the manifest;
  prefer whichever avoids a new manifest entry. **Do not hand-roll a Riccati iteration.**
- **Existing:** `OSQP.jl`, `StaticArrays`, `LinearAlgebra`, `SparseArrays`
- **Reuses unchanged:** `ControllerMod._mpc_body_matrices`, `_mpc_vel_A`, `_mpc_pose_A`,
  `_coriolis_jac`, `PlantMod.motor_torque`; `ControllerV2Mod.PhysicalLimits`,
  `capability_wrench`
- **Explicit exclusions:** no edits to `hybrid_ctrl/controllers.jl`, `bus.jl`,
  `scheduler.jl`, `mixer.jl`; no soft-constraint/slack reformulation of the QP; no
  change to the OSQP settings block

---

## 4. Component Breakdown

All components live in `ControllerV2Mod` (`hybrid_ctrl_v2/controllers_v2.jl`), in a new
section following the existing MPC/ASMC/PID sections.

### `bryson_Q_pose`
- **Type:** `function` (pure)
- **Responsibility:** Derive the 6 pose-state weights from `TOL` plus a stated
  closed-loop time constant.
- **Inputs:** `tol_pos`, `tol_head` (m, rad); `tau_cl::SVector{3,Float64}` (s)
- **Outputs:** `SVector{6,Float64}` ordered `(x, y, psi, Vx, Vy, psidot)`
- **Depends on:** nothing

### `bryson_R`
- **Type:** `function` (pure)
- **Responsibility:** Derive the per-wheel voltage-deviation weight from the voltage that
  commands the platform's full single-axis acceleration capability.
- **Inputs:** `lim::PhysicalLimits`, `motor`
- **Outputs:** `SVector{4,Float64}` (four identical entries — see §9 on wheel symmetry)
- **Depends on:** `capability_wrench`

### `u_eq_horizon`
- **Type:** `function` (pure)
- **Responsibility:** Build the stacked feedforward voltage `[U_eq,1; …; U_eq,Np]` from
  the reference trajectory over the horizon.
- **Inputs:** `ref::PosRef`, `t`, `dt`, `Np`, `params`, `motor`, `lim`
- **Outputs:** `Vector{Float64}` length `4*Np`
- **Depends on:** `PhysicalLimits` (for `m_eff`/`d_eff`/COM terms), `_mpc_body_matrices`
  (for `Amix`)

### `terminal_cost`
- **Type:** `function` (pure, with an internal cache)
- **Responsibility:** Solve the DARE at the horizon-end operating point and return `P`.
- **Inputs:** `A_end::Matrix{Float64}` (6×6), `Bm::Matrix{Float64}` (6×4),
  `Q::SVector{6}`, `R::SVector{4}`, `cache`
- **Outputs:** `Matrix{Float64}` (6×6, symmetric PSD)
- **Depends on:** `MatrixEquations.ared`

### `_build_Su!`
- **Type:** `function` (in-place)
- **Responsibility:** Build the condensed prediction matrices `Sx`, `Su` by **column
  recursion** instead of the existing recomputed-product nested loops.
- **Inputs:** preallocated `Sx`, `Su`; `As::Vector{Matrix{Float64}}`; `Bm`; `n`, `m`, `Np`
- **Outputs:** mutates `Sx`, `Su` in place
- **Depends on:** nothing. **Why this exists:** the v1 construction recomputes each
  transition product from scratch, costing `O(Np³)` small matrix multiplies per tick.
  `Np_pose` is going from 15 to 30, which makes that ~8× more expensive on a 100 Hz hot
  path. Column recursion is `O(Np²)` and produces **numerically identical** matrices —
  this is an efficiency fix, not a behavior change, and §10 requires proving that.

### `MPCControllerV2` (existing struct — extended)
- **Type:** `Base.@kwdef mutable struct`
- **New fields:** `lim::PhysicalLimits`; `tau_cl::SVector{3,Float64}`;
  `use_u_eq::Bool`; `use_terminal::Bool`; `dare_cache`; `log_diag::Bool` plus diagnostic
  vectors
- **Changed defaults:** `Np_pose = 30`; `Q_pose`, `R` derived at construction from
  `bryson_Q_pose`/`bryson_R`; `S` from the searched `S_scale`
- **Preserved:** `Np`, `Q` (velocity-mode; dead — see §7.2), `rate_hz`, `use_ltv`,
  `P_terminal`

### `Main.ControllerMod.mpc_wrench!(…, mpc::MPCControllerV2; mode)` (existing method — extended)
- **Responsibility:** unchanged contract, four internal changes (§7)
- **Inputs/Outputs:** unchanged — returns `SVector{3,Float64}` task-space wrench

### `MPC_SPACE_V2` (in `tune_controller_v2.jl`)
- **Type:** `const` search-space tuple list
- **Responsibility:** the 1-D residual search

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl_v2/
│   ├── controllers_v2.jl              # EXTEND — new MPC v2 section + MPCControllerV2 fields
│   └── tune_controller_v2.jl          # EXTEND — MPC_SPACE_V2, grid backend, Np sweep entry
├── instructions/
│   └── mpc-v2-bryson-weights-terminal-cost.md   # this file
└── _tmp/
    ├── mpc_v2_validation.jl           # NEW — §10 checks, standalone
    └── mpc_v2_horizon_sweep.jl        # NEW — the Np_pose diagnostic (§7.3)
```

No new directories. `hybrid_ctrl/` is untouched.

---

## 6. Key Interfaces

Signatures and docstrings only; bodies are stubs.

```julia
"""
    bryson_Q_pose(tol_pos, tol_head, tau_cl) -> SVector{6,Float64}

Bryson-normalized state weights for the 6-state pose model, ordered
(x, y, psi, Vx, Vy, psidot).

BRYSON'S RULE:  Q_ii = 1 / (largest acceptable error in state i)^2

Each cost term is then dimensionless and equals exactly 1.0 when that state sits
precisely at its tolerance, so the six terms are genuinely commensurable. v1's
Q_pose = (10, 10, 5, 1, 1, 0.5) added Q_1*(metres)^2 to Q_4*(m/s)^2 and called the
sum a number -- the ratio 10:1 compares a quantity to its derivative and means
nothing until a reference scale is named.

POSITION / HEADING -- taken directly from tune_controller.jl's TOL:

    tol_pos  = TOL.pos_max  = 1e-1 m     ->  Q_x = Q_y = 100
    tol_head = TOL.head_max = 1e-1 rad   ->  Q_psi     = 100

Use the *_max entries, not *_final. Bryson's rule asks for the largest acceptable
error, which is what a `max` spec states. TOL.pos_final (1e-2) is a STEADY-STATE
spec and is deliberately NOT represented here -- see the §9 caveat on this MPC
having no integral action.

VELOCITY / YAW RATE -- these have NO tolerance in the pose-mode objective.

    CRITICAL, and easy to get wrong: TOL.vel_rms and TOL.yawrate_rms are consumed
    ONLY by tune_controller.jl's :velocity branch. The :pose branch scores
    (final_pos, max_pos, final_head, max_head) and nothing else. Since every tier
    trajectory runs :pose, the velocity tolerances are IRRELEVANT to the score and
    must NOT be used as Bryson denominators. Doing so gives Q_vel/Q_pos ~ 1e4 and
    turns the MPC into a velocity tracker that barely corrects position.

    Instead derive the velocity tolerance from the position tolerance and a stated
    closed-loop time constant -- a velocity error eps_v sustained for tau produces
    a position error eps_v*tau, so the velocity error consistent with the position
    tolerance is:

        tol_v_i = tol_pos / tau_cl_i          (tol_head / tau_cl_3 for yaw rate)
        Q_(3+i) = 1 / tol_v_i^2

    tau_cl DEFAULTS TO THE PID v2 lam_inner = (0.15, 0.11, 0.12) s. That is a
    deliberate design choice, not an inherited constant: it makes all three v2
    controllers state the SAME closed-loop bandwidth, which is the substance of the
    fair-comparison argument in §2. It must be REPORTED, not buried.

EXPECTED OUTPUT at the defaults:

    Q_pose = (100, 100, 100, 2.25, 1.21, 1.44)

    position : velocity weight ratio ~ 44 : 1 -- the SAME SENSE as v1's 10:1, just
    with a derived magnitude and a stated reason. v1's ratio was under-derived, not
    inverted.

THE TOLERANCE CANCELS FROM THE SHAPE. Substituting tol_v = tol_pos/tau_cl into the
Bryson formula collapses to an identity worth implementing as the §10 check:

    Q_vel,i = Q_pos * tau_cl_i^2

    verify:  100*0.15^2 = 2.25    100*0.11^2 = 1.21    100*0.12^2 = 1.44

So the position:velocity ratio is 1/tau_cl^2 and does NOT depend on the tolerance at
all. This cleanly SEPARATES the two design decisions in this function, which are not
equally consequential:

    tau_cl                     -> the SHAPE of Q (position vs velocity weighting).
                                  Getting it wrong gives the wrong closed-loop
                                  damping/bandwidth. This is the consequential choice.

    *_max vs *_final in TOL    -> the SCALE of Q relative to the independently
                                  derived R, i.e. tracking-vs-effort aggression.
                                  Swapping pos_max (1e-1) for pos_final (1e-2)
                                  multiplies all six entries by 100 UNIFORMLY and
                                  leaves the shape untouched.

If a second residual tuning parameter is ever wanted beyond S_scale, that uniform
Q-scale multiplier is the honest place to put it -- a single interpretable
aggression scalar. Do NOT re-open a 6-dimensional Q search; that reintroduces
exactly the unconstrained-weight problem this brief removes.
"""
function bryson_Q_pose(tol_pos::Real, tol_head::Real, tau_cl::SVector{3,Float64}) end


"""
    bryson_R(lim, motor) -> SVector{4,Float64}

Bryson-normalized per-wheel weight on the voltage DEVIATION from U_eq.

    R_jj = 1 / (largest acceptable deviation from U_eq)^2

The denominator is NOT V_max. Once R penalizes `U - U_eq`, the quantity being
bounded is the FEEDBACK authority -- the voltage the controller spends beyond what
the trajectory already requires. The natural physical scale is the per-wheel
voltage deviation that commands the platform's full single-axis acceleration
capability:

    for each axis i:  W_i    = capability_wrench(lim)[i]        (6.59, 6.43, 56.7 N*m)
                      tau_j  = (Amix * W_i_as_a_wrench)[j]      pure-axis allocation
                      dV_i   = max_j |tau_j| / (G*eta*Kt/Ra)

    dTaudV = G*eta*Kt/Ra = 25.84*0.9*0.0335/2.0 = 0.3895 N*m/V

    axis x   : 0.25*6.59            = 1.648 N*m  ->  4.23 V
    axis y   : 0.25*6.43            = 1.608 N*m  ->  4.13 V
    axis psi : 0.25*lever*56.7      = 1.841 N*m  ->  4.73 V     (lever = R/(l+h) = 0.1299)

    dV_cap = max over axes = 4.73 V   ->   R_jj = 1/4.73^2 = 0.0447

The three axes land within 15% of each other, which is not a coincidence: the
1:1:8.6 capability ratio and the mixer allocation are two views of the same
geometry. Take the max so R never over-penalizes the axis with the largest
voltage demand.

ALL FOUR ENTRIES ARE IDENTICAL and must stay that way. The four wheels carry
identical motors and radii, and the O-config Amix gives each an equal-magnitude
share. A per-wheel R would break a symmetry the plant genuinely has: the optimizer
would be handed 3 extra parameters whose true optimum is "all equal", and any
deviation it reported would be noise-fitting.
"""
function bryson_R(lim::PhysicalLimits, motor) end


"""
    u_eq_horizon(ref, t, dt, Np, params, motor, lim) -> Vector{Float64}

Stacked feedforward voltage [U_eq,1; U_eq,2; ...; U_eq,Np], length 4*Np.

THE POINT: v1's R penalizes ‖U‖^2, the ABSOLUTE voltage. Holding a constant
velocity requires a nonzero voltage (back-EMF + drag), and v1 penalizes it. The QP
therefore trades "track the reference" against "don't spend the voltage the
reference requires", and settles at nonzero tracking error. With no integral action
anywhere in this MPC, that droop is permanent. Re-centering on U_eq puts the bottom
of the R penalty exactly where the trajectory already needs to be, so holding the
reference costs R nothing.

PER HORIZON STEP k (tk = t + (k-1)*dt):

  1. Equivalent wrench at the REFERENCE operating point:

         W_eq,k = (Mx_eq, My_eq, Mpsi_eq) evaluated with the reference's own
                  acceleration/velocity/heading and ZERO error terms

     Use the SAME expressions as ControllerV2Mod's asmc_wrench! (Mx_eq/My_eq/
     M_psi_eq), with Ax_eq -> Ax_ref, Ay_eq -> Ay_ref, alpha_eq -> alpha_ref and
     Vx/Vy/psi_dot taken from the reference rather than xhat. This is the ASMC's
     equivalent control and the PID-CT's M_eq_cmd -- one object, three consumers.

  2. Wheel torques -- EXACT, no pseudo-inverse:

         tau_eq,k = Amix * W_eq,k          (4x3 times 3x1)

     Amix maps wrench -> wheel torques in the FORWARD direction, which is exact and
     unambiguous. Do NOT use pinv here.

  3. Per-wheel motor inversion:

         V_eq,k,j = tau_eq,k,j / (G*eta*Kt/Ra)  +  Kb*G*omega_ref,k,j

     omega_ref,k,j is the reference wheel speed, obtained from the reference body
     velocity through the SAME kinematic map the plant uses (do not re-derive the
     sign pattern by hand -- reuse the existing map and verify with the §10
     round-trip test).

WHY THE ASMC ROUTE, NOT pinv(B_vel): solving B*U_eq = x_ref,k+1 - A*x_ref,k for the
minimum-norm U_eq also works and also lands null-space-free, but it derives the
feedforward from the LINEARIZED A,B and so inherits exactly the linearization error
the feedforward exists to cancel. The Amix route is nonlinear and exact.

NULL SPACE -- why this route needs no extra care: B_vel is 3x4, so it has a
one-dimensional null space. Solving Amix'*n = 0 for the O-config columns gives

    n = [1, 1, -1, -1]

a voltage pattern producing EXACTLY ZERO body wrench -- pure internal tension that
burns current, heats the motors, and scrubs the rollers, and that the tracking cost
‖X - X_ref‖^2_Q is completely blind to because it moves the platform not at all.
Since tau_eq = Amix*W lands in range(Amix) by construction, and n is orthogonal to
that range, the feedforward carries zero internal tension automatically.

This also settles what R is FOR after re-centering: it is the only term in the cost
that sees the null direction at all (the tracking cost cannot, and S penalizes only
its rate, not its level). So R must NOT be set to zero -- it should be re-centered
and then allowed to take the physically meaningful value bryson_R derives.

DOCUMENTED APPROXIMATION: step 3 inverts the quasi-static linear motor map and
ignores PlantMod's Coulomb term tau_f = 0.01 N*m. Against a typical tau_eq of
~1.6 N*m that is a 0.6% residual on a FEEDFORWARD term whose error the feedback
absorbs -- bounded and second-order, in the same spirit as the ASMC brief's
bare-vs-augmented mass note. Do not iterate to invert it exactly.

COST: Np evaluations of the equivalent-wrench expressions per tick. At Np=30 /
100 Hz this is the second-largest per-tick cost after the QP itself; if profiling
shows it dominating, cache per (ref, t) since the reference is deterministic.
"""
function u_eq_horizon(ref, t::Real, dt::Real, Np::Int, params, motor, lim::PhysicalLimits) end


"""
    terminal_cost(A_end, Bm, Q, R, cache) -> Matrix{Float64}

Infinite-horizon cost-to-go P, from the discrete algebraic Riccati equation

    P = A'PA - A'PB(R + B'PB)^-1 B'PA + Q

P is NOT a tuning parameter. It is the total cost still owed from state x_Np if the
loop continued optimally forever, and it is fully DERIVED from (A, B, Q, R) -- all
of which already exist.

WHY IT MATTERS: finite-horizon MPC with no terminal cost has NO STABILITY GUARANTEE,
even with a perfect model and no disturbances. Shortening Np can destabilize a loop
a longer Np stabilizes, because greedy-over-Np-steps and optimal-over-all-time can
disagree about the SIGN of the right action (the parallel-parking case: the spot is
behind you and the myopic move is to drive forward). Concretely here, Np_pose = 15
at 100 Hz is 0.15 s -- about 15 cm of path preview at 1 m/s, and under one open-loop
time constant on x (tau_open_x = 0.256 s). A curvature change 30 cm ahead is
invisible until it is 0.15 s away, at which point the demand arrives all at once and
the voltage box binds. That is precisely the look-ahead capability MPC was supposed
to have over PID.

With the terminal cost the truncated sum EQUALS the infinite sum (exactly, in the
unconstrained case), so Np reverts to being a constraint-and-preview parameter only.

WHICH A: use As[Np] -- the transition matrix at the END of the horizon, which is
where the terminal cost applies. Not As[1], not a nominal frozen A.

WIRING: the condensed cost already builds Qd = blkdiag(Q, Q, ..., Q). The terminal
cost is added by replacing the LAST n-by-n block with Q + P. The existing
MPCControllerV2 method already does exactly this with mpc.P_terminal; this function
supplies the value that field previously defaulted to zeros.

CACHING: a 6x6 ared is O(10) microseconds, tolerable at 100 Hz but not free. Cache
on the operating point and recompute only when it moves past a threshold (a
relative change in A_end above ~1e-3 is a reasonable trigger). MEASURE the hit rate
and the wall-clock, do not assume.

FAILURE HANDLING: ared can fail if (A_end, Bm) is not stabilizable at some operating
point. On failure, fall back to the last successful P, or to zeros(6,6) on the first
tick, and COUNT the failures in the diagnostic log. Never let it throw -- a solver
failure inside a 100 Hz control loop must degrade, not crash. A nonzero failure
count is a reportable finding, not something to silence.

DOCUMENTED APPROXIMATION: the DARE models (A, B, Q, R) but NOT the rate penalty S.
The terminal cost therefore describes an infinite-horizon problem without the ΔU
term. This is standard practice and is defensible here because S is a
transient-shaping penalty whose steady-state contribution is zero (holding a
trajectory needs ΔU ~ 0) -- but it IS an approximation and belongs in the paper's
method section, not hidden.

DETECTABILITY: Q from bryson_Q_pose is diagonal and strictly positive on all six
states, so (A, sqrt(Q)) is detectable trivially and no extra check is needed. If a
future variant zeroes the velocity weights, detectability must be re-argued (it
still holds, via the position rows integrating velocity, but it stops being free).
"""
function terminal_cost(A_end::Matrix{Float64}, Bm::Matrix{Float64},
                       Q::SVector{6,Float64}, R::SVector{4,Float64}, cache) end


"""
    _build_Su!(Sx, Su, As, Bm, n, m, Np) -> Nothing

Condensed LTV prediction matrices, built by COLUMN RECURSION.

    X = Sx*x0 + Su*U      (states eliminated; U is the only unknown)

MATHEMATICALLY IDENTICAL to the v1 nested-loop construction. This is purely an
efficiency fix, and §10 requires proving the identity numerically rather than
asserting it.

WHY: v1 recomputes each transition product Phi(k,j) from scratch inside a doubly
nested loop, costing O(Np^3) small matrix multiplies per tick. Np_pose is going
from 15 to 30, an ~8x increase on a 100 Hz hot path. The recursion exploits
Phi(k,j) = As[k]*Phi(k-1,j), reducing this to O(Np^2).

Also note the decision-variable count: at Np=30, m=4 gives 120 variables and a
120x120 H. The QP itself grows superlinearly. OSQP's warm start (already wired via
bus.mpc_warm) is doing real work at this size -- do not disable it.

Preallocate Sx and Su on the controller instance rather than allocating per tick.
"""
function _build_Su!(Sx::Matrix{Float64}, Su::Matrix{Float64},
                    As::Vector{Matrix{Float64}}, Bm::Matrix{Float64},
                    n::Int, m::Int, Np::Int) end
```

---

## 7. Data Flow

### 7.1 Per-tick sequence in `mpc_wrench!` (`:pose` mode)

Changes from v1 are marked **[NEW]**; everything unmarked is unchanged.

1. Read `t`, `dt`, measured `omega`, and `x0 = (X, Y, psi, Vx, Vy, psidot)` from `xhat`.
   `x0` is the estimated state — **this, not the horizon, is what makes MPC a feedback
   controller.** The horizon buys constraint-awareness and preview; the per-tick `x0`
   refresh buys feedback. If the plan were applied open-loop for all `Np` steps, an
   unmodeled LuGre slip event of even 0.5 m/s² would accumulate 0.05 m/s of error before
   the controller noticed, against a target on the order of 0.01 m/s.
2. Build `Bvel`, `Amix` via `_mpc_body_matrices` (unchanged).
3. Build `As[1..Np]` by re-linearizing at the reference operating point per step
   (unchanged, `use_ltv = true`).
4. **[NEW]** `_build_Su!` fills preallocated `Sx`, `Su` by column recursion.
5. Build `xref` from the reference preview (unchanged).
6. Build `D` and `p_prev` (unchanged).
7. **[NEW]** `U_eq_stack = u_eq_horizon(...)`, length `4*Np`.
8. **[NEW]** `P = terminal_cost(As[Np], Bm, Q_pose, R, cache)`; the last `6×6` block of
   `Qd` becomes `Q + P`.
9. Assemble the cost. Two changes to `f`, none to `H`:
   - `H = Su'*Qd*Su + Rd + D'*Sd*D` — **unchanged** (`R` re-centering is a linear shift,
     it does not touch the quadratic term)
   - `f` gains the term `−Rd*U_eq_stack` **[NEW]**, alongside the existing
     `Su'*Qd*(Sx*x0 − xref)` and `−D'*Sd*p_prev`
10. Constraints (voltage box from `omega`, slew from `p_prev`) — unchanged.
11. Solve, warm-started; on failure fall back to `bus.mpc_last_u` — unchanged.
12. Map `u0` back through `PlantMod.motor_torque` and `pinv(Amix)` — unchanged.

### 7.2 Constants: derived, specified, searched, dead

| quantity | count | status | value / source |
|---|---|---|---|
| `Q_pose` (x, y, psi) | 3 | **derived** | `1/TOL.pos_max²`, `1/TOL.head_max²` → `100, 100, 100` |
| `Q_pose` (Vx, Vy, psidot) | 3 | **derived** | `1/(tol_pos/tau_cl)²` → `2.25, 1.21, 1.44` |
| `R` | 4 | **derived** | `1/dV_cap²` → `0.0447` ×4 |
| `P_terminal` | 6×6 | **derived** | `ared(As[Np], Bm, Q_pose, R)` |
| `tau_cl` | 3 | **specified** | `(0.15, 0.11, 0.12)` s — PID v2 `lam_inner`; a reported design choice |
| `Np_pose` | 1 | **diagnostic sweep** | default **30**, swept downward (§7.3) |
| `Np`, `Q` (velocity mode) | 4 | **dead** | never read — every tier trajectory runs `:pose` |
| `rate_hz`, `use_ltv` | 2 | **pinned** | `100.0`, `true` |
| **`S_scale`** | **1** | **SEARCHED** | log, `[0.25, 25]`, default `2.5` |

**Dead-parameter handling.** `MPCControllerV2` keeps `Np` and `Q` for schema
compatibility, but the `:velocity` branch of `mpc_wrench!` must `error()` in v2 —
matching the `PIDControllerV2`/`pid_wrench!` precedent, which already errors on any mode
other than `:pose`. Silently carrying a dead 3-parameter block into a search space is how
this class of bug survives.

**Why `S` is the one thing that must be searched.** `Q` answers "how well must I track?"
and `R` answers "how much authority may I spend?" — both are trajectory-spec questions
with tolerance-table answers. `S` answers "how much voltage jitter is acceptable?", which
is about actuator wear, current ripple, and interaction with estimator noise. Nothing in
the trajectory spec pins it. It is also the only parameter whose optimum depends on the
noise realization, which is exactly what earns it the 5-seed averaging.

**Why the bracket is `[0.25, 25]`.** From the hard slew limit,
`max ΔV per tick = dV_max·dt = 200 × 0.01 = 2 V`. Bryson again:
`S_jj = 1/(acceptable ΔV)²`, so `ΔV = 2 V` (at the limit) gives `S = 0.25`, and
`ΔV = 0.2 V` (very smooth) gives `S = 25`. Two decades, physically bounded at both ends.
**v1's `[1e-3, 1]` does not transfer** — its `Q` carried a different normalization, so
the numbers are not comparable.

### 7.3 The `Np_pose` diagnostic sweep

`Np` is settled by a **convergence check you read**, not a cost you minimize.

**RUN THIS BEFORE THE `S_scale` SEARCH, NOT AFTER.** The two are coupled — `S` affects
where `Np` converges — but the cost asymmetry and the direction of the coupling together
settle the order.

*Cost:* the `Np` sweep is ~30 runs (5 values × 6 trajectories, deterministic criteria, one
seed). The `S` grid is ~300 (10 points × 6 trajectories × 5 seeds). Running the expensive
pass at `Np = 30` and only then discovering 15 suffices pays ~4× on the QP (120 vs 60
decision variables, plus the `O(Np²)` `Su` build and `Np` feedforward evaluations) across
the entire search.

*Coupling direction:* a lower `S` means a weaker rate penalty, so the QP moves voltage more
aggressively — which produces **more** constraint activity at the horizon end and a plan
**more** sensitive to truncation. Both acceptance criteria below fail longer at low `S`, in
the same direction. So run the sweep at **`S = 0.25`, the bracket floor** — the most
demanding member of the family. The `Np*` it returns is then conservative for every `S` the
later search could land on, and no re-derivation is needed.

Sweep **downward from 30**: `Np_pose ∈ {30, 25, 20, 15, 10}`, at the derived weights and
`S = 0.25`, on the tier trajectory set. Accept the smallest `Np` where **both** hold:

- **Plan convergence:** `‖U₁(Np) − U₁(30)‖∞` stays below a stated threshold across the
  trajectory (a threshold of ~1% of `V_max` is a reasonable starting point; state
  whichever is used).
- **Terminal-region validity:** the voltage box and slew constraints are **inactive over
  the last ~3 horizon steps**. This is not optional polish — the DARE assumes
  *unconstrained* LQR from step `Np` onward. If constraints still bind at the end of the
  horizon, that assumption is false and `P` is optimistic.

Do **not** put `Np_pose` in the optimizer. It is an integer; handing a continuous
optimizer an integer and rounding creates plateaus — which is a discrete version of
exactly the flat-direction problem this brief exists to remove.

**After the `S` search completes**, re-check both criteria once at `Np*` with the tuned
`S`. By the monotonicity argument above this should pass by construction; it is cheap
insurance, not a second sweep. **If it fails, the monotonicity assumption is wrong and
that is itself a reportable finding** — step `Np` up one grid point and say so, rather
than quietly widening the sweep.

**What invalidates the sweep:** the derived weights are inputs to it. `Q_pose`, `R`, and
`P` follow from `TOL`, the capability wrench, and the DARE, so they are settled the moment
the code is correct — there is nothing to tune first. But if `tau_cl` is ever revised,
`Q` changes, therefore `P` changes, and the sweep must be re-run.

Report the sweep as a horizon-convergence study.

---

## 8. Implementation Sequence

1. **`bryson_Q_pose`, `bryson_R`** — leaves, no dependencies. Unit-check the expected
   values in §7.2 before anything consumes them.
2. **`_build_Su!`** — leaf. Verify bit-equivalence against the v1 nested-loop
   construction on random `As`/`Bm` **before** wiring it in. If it is not identical,
   every downstream comparison is contaminated.
3. **`u_eq_horizon`** — depends on `PhysicalLimits` and `Amix`. Validate standalone with
   the §10 round-trip test on a steady reference; a sign error in the wheel-speed
   kinematics is the most likely failure and the round-trip catches it.
4. **`terminal_cost`** — depends on `MatrixEquations`. Validate `P` is symmetric PSD and
   satisfies the DARE residual before wiring.
5. **`MPCControllerV2` field extension** — depends on 1, 3, 4. `@kwdef` evaluates defaults
   in declaration order, so `lim` and `tau_cl` must be declared **before** the derived
   `Q_pose`/`R`/`S` (the same ordering constraint `ASMCControllerV2` already documents).
6. **`mpc_wrench!` method extension** — depends on all of the above. Land the four
   changes **one at a time**, running the §10 regression gate after each, so any
   behavioral change is attributable to a single cause.
7. **`_tmp/mpc_v2_horizon_sweep.jl`** — depends on 6. Run the `Np_pose` sweep at
   `S = 0.25` and fix `Np*` **before** building the search space (§7.3). The derived
   weights are already final at this point; there is nothing to tune first.
8. **`MPC_SPACE_V2` + grid backend** in `tune_controller_v2.jl` — depends on 5 and 7.
   The `S` grid runs at the `Np*` chosen in step 7, which is where the ~4× QP saving is
   realized. Finish with the single confirmation re-check of the §7.3 criteria at the
   tuned `S`.

---

## 9. Considerations

**No integral action.** This MPC has none, in any version. `U_eq` removes the *modeled*
steady-state bias but cannot reject an *unmodeled* constant disturbance — persistent
friction asymmetry, a load offset, a systematic estimator bias. Consequently
`TOL.pos_final = 1e-2` is not guaranteed by any weight choice, and raising `Q` is the
wrong lever for it (it buys aggression, not steady accuracy). The right fixes are
disturbance-observer augmentation or a `ΔU`-formulation with an integrating state — both
**out of scope** here (§11). If `final_pos` dominates the tuning score, report that as a
structural finding about the controller, not as a tuning failure.

**Solver-noise sensitivity.** With the gauge degeneracy dissolved, the remaining sensitivity
to `1e-6*I` and OSQP's absolute tolerances is small but nonzero. Keep the regularizer and
the settings block **exactly as they are**; changing them silently changes the meaning of
every tuned result. If a future run alters them, re-run the whole comparison.

**Compute budget.** `Np_pose = 30` at `m = 4` gives 120 decision variables and a 120×120
`H`, roughly 4× the v1 QP work, plus `Np` equivalent-wrench evaluations for `U_eq` and a
cached DARE. Measure wall-clock per tick and the real-time factor of the full sim before
committing to 30 — the sweep exists partly to buy that back.

**LTV vs the DARE.** `A` varies along the horizon and with `psi`, while the DARE assumes a
fixed `A`. Using `As[Np]` is the principled choice (the terminal cost applies at the
horizon end). There is a plausible argument that with `Q_x = Q_y` and `Q_Vx ≈ Q_Vy` the
solution is near-rotation-invariant and one solve would serve all `psi` — **note that
`bryson_Q_pose` gives `Q_Vx = 2.25` vs `Q_Vy = 1.21`, so x/y isotropy does NOT hold** and
that shortcut is unavailable. Rely on the operating-point cache instead.

**Estimator coupling.** `x0` comes from the frozen ESKF, and `S` in particular trades
against estimator noise. Keep the estimator identical across the ASMC/PID/MPC comparison
runs (the `compare_controllers_eskf.jl` frozen-estimator design) or the `S` optimum is not
comparable across controllers.

**Optimizer choice for a 1-D search.** dxNES is designed for coupled, ill-conditioned,
covariance-adapting problems. For a single log-scaled parameter it is the wrong tool. Use
a **deterministic log grid** (~10 points × 5 seeds = 50 evaluations), optionally followed
by golden-section refinement. This is reproducible, needs no restart policy, and makes the
`S` sensitivity curve directly publishable — which a stochastic search does not.

**Reporting.** `tau_cl`, the choice of `TOL.*_max` over `*_final`, the DARE's omission of
`S`, and the `tau_f` inversion residual are all design decisions, not derivations. Each
belongs in the paper's method section — but they do **not** carry equal weight, and the
write-up should say so. `tau_cl` sets the shape of `Q` and is shared with the PID v2
bandwidth claim, so it is load-bearing for the comparison. The `*_max`/`*_final` choice is
a uniform scale multiplier on `Q` against `R` (§6 identity) — a single aggression scalar,
defensible either way, and cheap to report a sensitivity for.

---

## 10. Success Criteria

**Regression gate (run first, and after every step of §8.6):**

- [ ] With `use_u_eq = false`, `use_terminal = false`, and `Q_pose`/`R`/`S` forced to the
      v1 defaults, `mpc_wrench!` on `MPCControllerV2` reproduces v1 **bit-identically**
      over a full trajectory. Without this, nothing below is interpretable.
- [ ] `_build_Su!` reproduces the v1 nested-loop `Sx`/`Su` to machine precision on random
      `As`/`Bm` at `Np ∈ {5, 15, 30}`, and is measurably faster at `Np = 30`.

**Derivation checks:**

- [ ] `bryson_Q_pose` returns `(100, 100, 100, 2.25, 1.21, 1.44)` at the defaults.
- [ ] **Identity holds:** `Q_vel,i == Q_pos · tau_cl,i²` to machine precision, and
      re-running with `tol_pos = TOL.pos_final` scales all six entries by exactly 100
      while leaving the position:velocity ratio unchanged. This is the check that
      confirms `tau_cl` owns the *shape* and the tolerance owns only the *scale*.
- [ ] `bryson_R` returns `0.0447 ± 0.001` on all four entries, and the three per-axis
      voltages land at `4.23 / 4.13 / 4.73 V`.
- [ ] **`U_eq` round-trip:** feeding `V_eq` through `PlantMod.motor_torque` at the
      reference `omega` and back through `pinv(Amix)` reproduces `W_eq` to within the
      documented `tau_f` residual (< 1%). **This is the test that catches a sign error in
      the wheel-speed kinematics** — do not skip it.
- [ ] **`U_eq` null-space check:** `dot(tau_eq, [1,1,−1,−1])` is zero to machine precision
      at every horizon step, confirming the feedforward carries no internal tension.
- [ ] `terminal_cost` returns a symmetric PSD `P` with DARE residual below `1e-8`, and
      `P ⪰ diagm(Q_pose)` (the cost-to-go is never less than one step of running cost).

**Behavioral checks:**

- [ ] **Droop test — the falsifiable check that the `R` bias is actually gone.** On a
      constant-velocity segment, measure steady-state tracking error at `R = 0.0447` and
      at `R = 10×` that. With `use_u_eq = false` the error must move with `R`; with
      `use_u_eq = true` it must be insensitive.
- [ ] **Degeneracy check.** Scale `Q_pose`, `R`, and `S` all by 10 and confirm the
      trajectory now **changes**. If it does not, something is still gauge-free and the
      Bryson wiring is incomplete.
- [ ] **Terminal-cost effect is visible.** At a deliberately short `Np_pose = 10`, the run
      with `use_terminal = true` scores better than `false`. If the two are
      indistinguishable, `P` is not actually reaching the cost — check the `Qd` last-block
      wiring.
- [ ] **Preview is real.** On a trajectory with a sharp curvature change, `U₁` begins
      responding before the change enters the tracking error — the capability that
      justifies MPC over PID at all.
- [ ] `terminal_cost` DARE-failure count and cache hit rate logged; failures are zero or
      explained.
- [ ] Per-tick wall-clock and full-sim real-time factor recorded at `Np_pose = 30`.

**Tuning outcome:**

- [ ] The `Np_pose` sweep (run **first**, at `S = 0.25`) reports the smallest `Np`
      meeting both §7.3 criteria, with the constraint-activity fraction over the last 3
      steps tabulated per `Np`.
- [ ] The `S_scale` grid, run at `Np*`, produces a **single-minimum, smooth** score curve
      over `[0.25, 25]`. A ragged or flat curve means the residual search is picking up
      solver noise rather than control performance — investigate before accepting a value.
- [ ] Confirmation re-check: both §7.3 criteria still hold at `Np*` with the tuned `S`.
      A failure here falsifies the monotonicity argument and must be reported, not
      absorbed by widening the sweep.
- [ ] Final MPC v2 score beats both the v1 dxNES result (67 evaluations) and the v1
      coarse-search result on the same trajectory set and seeds. **If it does not, that is
      a reportable finding about MPC's structural fit to this plant, not a reason to
      re-open the search space.**

---

## 11. Out of Scope

- Integral action, disturbance-observer augmentation, or a `ΔU`-formulation with an
  integrating state — the right fix for `pos_final`, deferred deliberately (§9)
- Soft constraints / slack variables in the QP
- Nonlinear MPC, multiple shooting, or any move off the condensed linear formulation
- Changing `rate_hz`, `use_ltv`, the `1e-6*I` regularizer, or the OSQP settings block
- Per-wheel `R` or `S` asymmetry (§6, `bryson_R`)
- `:velocity` mode — errors out in v2
- Friction diversity (`mu` sweep) — deferred to the next iteration, consistent with the
  ASMC v2 and PID v2 briefs
- Re-measuring `TOL` — taken as given from `tune_controller.jl`
