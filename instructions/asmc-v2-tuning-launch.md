# ASMC v2 — Tuning Launch Brief

> **Generated:** 2026-08-06
> **Stack:** Julia 1.x, `hybrid_ctrl_v2/`, existing project. No new packages.
> **Scope:** Launch configuration for the ASMC v2 5-seed tuning run.
> **Prerequisite briefs:** `asmc-v2-physical-gain-bounds.md` (the derivation this run tunes on top of)

---

## 1. Overview

The previous ASMC 5-seed run is **void** and must not be used as a baseline. Its scores
were obtained with the switching gain `K` **exceeding its derived friction-circle
ceiling by up to 11.7×** — the physical bound the whole v2 derivation exists to impose
was not enforced. This brief specifies the re-launch after that defect and two others
were fixed.

**What changed in the code since that run** (all in `ControllerV2Mod`, `hybrid_ctrl/` untouched):

1. **`K_max` is now actually enforced.** The growth gate reads `K_max_sched` (the
   per-tick physical ceiling) instead of `K_max_eff = max(K, K_max_sched)`. Under the
   old form, once `K` passed the ceiling the ceiling became `K` itself and the gate
   evaluated `0.5 − 0.5·tanh(0.02·K)` — 0.40 at `K = 10`, never closing. The lazy clamp
   was introduced to avoid a discontinuous drop when the *schedule* tightens; that
   intent is preserved by keeping `K_max_eff` in the **decay** terms, while **growth** is
   throttled against the real ceiling.
2. **`_smooth_bound_v2` width is fractional.** Was `tanh(K − 0.98·K_max)` — a knee at
   `0.98·K_max` but a width of ~1 *absolute* unit of `K`, so the gate did not close until
   ~1.4× the ceiling on x (`K_max = 6.59`) versus ~1.04× on ψ (`56.7`). Now
   `tanh(50·(K/K_max − 0.98))`: same fractional behaviour on every axis, essentially
   closed 2% above the ceiling.
3. **`start_at_floor` defaults to `true`** — `K` starts at `0.95·K_floor` rather than
   zero. **Measured to be inert** (identical scores to 3–4 significant figures); retained
   because starting at the measured typical switching demand is the more defensible
   initial condition, not because it fixes anything. Do not claim it as an improvement.

4. **`eps`/`eps_psi` are now DERIVED, not inherited** (`eps_from_noise`). Sized as
   `max(0.5·tol, k·σ_s)` with `σ_s = sqrt(σ_v² + (λ·σ_pos)²)` from a new `SurfaceNoise`
   field. Clean runs collapse to the resolution floor `(5e-4, 5e-3)`; the noisy branch
   is noise-sized. Rationale: the pinned `0.0175` was 17.5× the estimator objective's
   `v_tol = 1e-3`, so the ASMC could not respond to accuracy the estimator was being
   tuned to deliver, while the PID — which has no deadband — could. **Measured effect on
   clean runs is small** (1.7–3.5%; see §7), so this is a fairness/derivation fix, not a
   performance one.
6. **gamma is factored out of the whole gain-update bracket, fixed, and removed from
   the search; `tau_ceiling` takes its place.**

   ```
   dK = gamma * [ s*tanh(s/eps)*bound - c_tilde*(K/K_max)^3 - sigma_tilde*(K-K0)*exp(...) ]
   ```

   Previously growth scaled with gamma but the decay terms did not, so the equilibrium
   moved with the tuning knob: `K_eq ~ (gamma/c)^(1/3)`. gamma was doing two jobs --
   setting how fast K adapts AND where it settles. With `c_tilde` and `sigma_tilde`
   divided by `gamma_ref`, gamma cancels from the equilibrium and becomes a pure rate
   multiplier:

   ```
   K_eq = K_max * (s*tanh(s/eps) / c_tilde)^(1/3)      <- disturbance only
   ```

   **Measured consequence:** score now varies **0.5-1.2% across a 64x gamma sweep**
   (`octagon_stress` 0.7075 -> 0.7042 for gamma = 25 -> 1600). gamma is inert as a
   *score* parameter, so keeping it in a 6-D search wasted half the dimensions. It is now
   **fixed at derived values** (SS2) and `tau_ceiling` -- which is what actually sets the
   level -- is searched instead.

7. **Commanded-wrench logging** added to both controllers (`W_log`, and `Msw_log` on the
   ASMC), gated by the existing `log_K`/`log_diag` flags.

**Verified after the fixes:** `max(K)/K_max_base ≤ 0.888` across a γ sweep spanning
25–1600 on three stress trajectories, against 11.735 before.

---

## 2. What is tuned and what is derived

**Tuned — `ASMC_SPACE_V2`, FOUR dimensions:**

| parameter | role | box |
|---|---|---|
| `lam_x_max` | surface slope, x — sliding-phase position-error decay rate | log [0.3, 10] |
| `lam_y_max` | surface slope, y | log [0.3, 10] |
| `lam_psi_max` | surface slope, ψ | log [1, 20] |
| `tau_ceiling` | cubic-barrier constant — **sets where K settles** | log [0.1, 100] |

`tau_ceiling` is the ceiling-to-floor decay time of the cubic term at `γ = gamma_ref`:
`c̃ = (K_max_base − K_floor)/(gamma_ref·tau_ceiling)`. Since `K_eq ∝ tau_ceiling^(1/3)`,
it is the switching-authority **level** knob — larger means more authority and slower
relaxation. **The cube root makes it deliberately weak:** 10× in `tau_ceiling` moves
`K_eq` only 2.15×, which is why the box spans 200× rather than the narrow range the old
pinned `tau_ceiling = 0.5` might suggest.

**Derived — do not add to the search:** `K_max_base` (= `capability_wrench`), `K_floor`
(measured p50 of `|Msat − Meq|`), `decay_sigma`, `cubic_coeff`, `K_x0/y0/psi0`, and the
per-tick `kmax_schedule` ceiling.

**Derived from the noise model — not searched, not pinned:** `eps`, `eps_psi` via
`eps_from_noise(...)`, a max of **three** terms:

```
eps_i = max( eps_floor_i ,  0.5*tol_i ,  k*sigma_s,i )
```

Clean: `(5e-4, 5e-3)` — the resolution floor binds. The chatter floors are
implemented but **DISABLED (0.0) for this run**; the eps/chatter trade is deferred.
Noisy at the oracle's injection: `(0.0892, 0.0877)`. **The noise model is the
questionable input** — `OracleEstimator(:noisy)` uses `σ_pos = 0.020 m` and
`σ_vel = 0.010 m/s`, the latter 10× the estimator objective's own `v_tol`. Since
`λ·σ_pos` contributes 96% of `σ_s`, the noisy band is set almost entirely by a position
noise figure that has not been reconciled with the ESKF's achieved error. **Populate
`SurfaceNoise` from the ESKF's measured error before the noisy stage**, not from the
oracle.

**γ — FIXED at derived values, PER AXIS.** Because `c̃/K_max` is axis-independent
(0.0611 on x, 0.0612 on ψ — both carry `alloc_ratio`), equal γ would give equal
adaptation time constants. That is *not* wanted: the axes have different dynamics, so γ
is scaled so `τ_K` tracks each axis's own open-loop constant.

```
tau_K,i = 1 / ( 3*gamma_i * (c_tilde/K_max) * (K_eq/K_max)^2 )   =>  gamma_i ~ 1/tau_open,i

              tau_open      gamma   (tau_K = tau_open/2 at K_eq/K_max = 0.5)
x              0.256 s        170
y              0.114 s        380
psi            0.155 s        280
```

`τ_K = tau_open/2` is a **design choice**: `τ_K = tau_open` would lag a changing
disturbance by roughly a quarter cycle, and much faster buys little while costing chatter
(measured 0.248 at γ=25 → 0.328 at γ=400, ~30%). Since `lambda_chatter = 0` by default
the optimizer would never see that cost — which is exactly why this is specified rather
than searched. Bound check: `max(K)/K_max = 0.813` at γ = 400, first discrete-time
overshoot only at γ = 1600 (1.014), so a 4× margin.

CAVEAT: `τ_K ∝ 1/(K_eq/K_max)²`, and the measured ratio varies by trajectory (0.22 on
`spin_creep_stress_yaw`, 0.47–0.52 on `octagon_stress`, 0.81 on `coupled_vomega_stress`).
These γ are calibrated at a representative 0.5, not universally. Report that.

**DEFERRED — the `eps`/chatter trade.** `eps` is the direct chatter knob (switching
slew ~ `(K/eps)*sdot`), and widening it was measured to cut chatter **33–45% at no
tracking cost** (`octagon_stress` at `tau_ceiling = 0.5`: chatter 0.286 → 0.158,
tracking 0.6446 → 0.6401). That change is **not applied here** — per direction, this run
uses the resolution floor and the trade is revisited later. To reproduce the widened
configuration set `eps_floor_xy = 0.005`, `eps_floor_psi = 0.02`; the mechanism and its
measured effect are documented in `eps_from_noise`.

**Specified — pinned, reported, not searched:** `gamma = (170, 380, 280)`,
`gamma_ref = 250`, `k_eps = 1.75`, `eps_floor_xy = 0.005`, `eps_floor_psi = 0.02`,
`tol_v = 1e-3`, `tol_rate = 1e-2`, `tau_relax = 2.0`, `decay_k = 0.25`,
`v_max_axis = (0.63, 0.63, 3.8)`, `rate_hz = 1000`.

**Chatter is NOT penalized this run** (`lambda_chatter = 0`) and NOT handled via `eps`
either — both levers are deferred. Chatter is **logged and reported**: measured
0.24–0.66 V/ms at the launch configuration, i.e. **30–83% of the `0.8 V/ms` slew
ceiling**, rising monotonically with `tau_ceiling`. Expect the tuned result to sit near
the top of that range. If it becomes a criterion later, note
that `w = 24/0.8 = 30` makes `lambda_chatter*(chatter/V_MAX)` read as *fraction of slew
capability consumed*, and that applying it to the ASMC alone would recreate exactly the
kind of cross-controller asymmetry this brief exists to remove.

**TRAP — `gamma_ref` must stay a single fixed scalar, distinct from `gamma_*`.** Setting
`gamma_ref_i = gamma_i` per axis makes `γ·c̃ = (K_max − K_floor)/tau_ceiling`, so γ
cancels out of the decay term and the old growth-scales-but-decay-doesn't asymmetry
returns. At `gamma_ref = 250` the per-axis decay times come out as
`tau_ceiling·gamma_ref/γ_i` — faster axes relaxing faster, which is the intent.

**Reaching vs sliding — the organizing split.** `λ` governs the sliding phase (on-surface
error decay, `ė = −λe`); the `K` schedule governs the reaching phase. `λ` is the ASMC's
analogue of the PID's *position*-loop rate `Kp_pos = 1/(N·λ_inner)`, **not** of
`λ_inner` itself. Correct comparison at PID-FB's tuned gains:

```
              PID Kp_pos      ASMC λ (prior run, VOID)
x                1.91            0.55 – 1.46
y                2.50            0.69 – 1.71
psi              5.00           16.8  – 19.8
```

The ASMC is not bound by the cascade's `N = 4` separation factor — it is single-loop, so
it may legitimately run anywhere from `Kp_pos` up toward the PID's inner-loop rate
(7.6 on x). Its real ceilings are the `λ|e| ≤ v_max` actuation cap and chatter.

---

## 3. Boxes — launch as-is, but report the rails

Two boxes railed in the prior run, **but that run had the unbounded-`K` defect**, so the
rails are not trustworthy evidence either way. **Do not block the launch on resolving
them** — instead let this run tell you whether they still bind, and treat a rail as a
result to act on rather than a value to accept:

- **`tau_ceiling` (new, log [0.1, 100]).** Swept before launch, and it is a **strong,
  live parameter** — much better use of a dimension than γ was:

  Swept at the LAUNCH configuration (resolution-floor `eps`):

  ```
  tau_ceiling            0.5      5.0     20.0    100.0     chatter @100
  octagon_stress       0.6446   0.4557   0.1445   0.1095       0.553
  spin_creep_yaw       0.0577   0.0489   0.0450   0.1550       0.661   <- turns over by 100
  coupled_vomega       32.499   12.997   3.9485   1.2762       0.530
  ```

  The box was widened from 20 to 100 because two of three trajectories still improved at
  the old top. At 100, `octagon_stress` returns are flattening (1.32x for a 5x increase)
  and `spin_creep_stress_yaw` has **turned over** (0.0450 -> 0.1550), so 100 brackets an
  interior optimum for at least one trajectory. **Still watch for a top rail** across the
  full 12-trajectory tier.

  Chatter rises monotonically with `tau_ceiling` (0.24 -> 0.66 V/ms, 30 -> 83% of the
  slew ceiling) and is unpriced this run — so a top rail would also be a
  maximum-chatter result. Record it.
- **`lam_psi_max` upper bound (currently 20).** Prior run railed at 16.8–19.8. Re-scan
  under the fixed clamp before widening; the rail may have been the unbounded `K`
  compensating for a surface that was too tight.

**Do not raise the `gamma` cap to buy score.** Under the old clamp that bought score by
exceeding the physical bound. Any re-scan showing improvement above 100 must be checked
against `max(K)/K_max_sched` before it is accepted.

---

## 4. Budget — the reason the prior run scattered

The prior 5 seeds spread **1.36 – 3.89 in score** (2.9×) with no parameter determined
better than 5–26× across seeds. That is not multimodality; it is under-convergence:

```
                      PRIOR RUN (void)      THIS RUN
dimensions                  6                  4
CMA population              9                  8
phase-2 evals            53 - 61              ~60
rule of thumb        600 - 1200 evals     400 - 800 evals
budget used              ~310                ~310
shortfall                2 - 4x             1.3 - 2.6x
```

Dropping gamma from the search is the single largest improvement to the convergence odds
here: it removes three dimensions that measurably move the score by ~1%.

4 of 5 seeds stopped on `cap`, not convergence. The one that stopped on `plateau`
(seed 3, 53 evals) reached 1.45 while a capped seed reached 1.36 — so the plateau
detector fired early too.

Contrast PID v2: **3 dimensions**, same ~310 budget, 5 seeds agreeing to **0.27%**.
Dimension is the whole story.

**DECISION: launch the 4-D descent at the existing budget.** This is a
**decision run, not a production run** — its job is to determine whether 6-D is viable
here, and the seed spread answers that regardless of whether the run converges. Raising
the budget to the ~600 evaluations the rule of thumb wants would roughly double the wall
clock (12 trajectories/eval × ~13.7 s/sim under 5-seed concurrency ≈ 2.7 min/eval) and
would not change what the run tells you.

**Decision criteria, applied after this run:**

| observation | conclusion | action |
|---|---|---|
| seed spread ≲ 10% in score, parameters agreeing within ~2× | 6-D is viable at this budget | keep the space, proceed to noisy |
| seeds scatter like the void run (2.9× score, 5–26× parameters) | 4-D still under-converges | drop to the three `λ` alone, fixing `tau_ceiling` at whatever the scatter centres on; or coordinate-descent per axis |
| any parameter rails on ≥3 of 5 seeds | box truncates the optimum | widen that box and re-run before drawing conclusions — the ψ-floor precedent moved the PID's clean score 42% |

If further reduction is needed, drop `tau_ceiling` before touching `λ` — it is the
weakest knob (cube-root sensitivity) and the only one with no prior evidence behind it.

Reference: PID v2 at **3 dimensions** and the same ~310 budget gave 5 seeds agreeing to
**0.27%**. That is the target behaviour; anything far from it means the space is too
large for the budget.

---

## 5. Pre-launch gates

Run all four; do not launch on a failure.

- [x] **`K` stays bounded — PASSED.** Across the full `tau_ceiling` box
      (0.5 / 5 / 20 / 100) on `coupled_vomega_stress`, `octagon_stress`,
      `spin_creep_stress_yaw`, at the launch `eps`: **`max(K)/K_max_base ≤ 0.886`** on
      every axis and trajectory.

      **Use `K_max_base` as the denominator, NOT `K_max_sched`.** The scheduled ceiling
      is floored at `0.05·K_max_base` for the degenerate case (`asmc_wrench!`), so
      `K/K_max_sched` is bounded by `1/0.05 = 20` *by construction* and routinely reads
      2–16 without anything being wrong. That happens when K — legitimately grown while
      the schedule was permissive — sits above a momentarily collapsed schedule. It is
      the designed behaviour: the growth gate evaluates
      `_smooth_bound_v2(K, K_max_sched) = 0` there, so growth is fully stopped and the
      decay terms walk K back down, without the discontinuous drop a hard clamp would
      cause. A `K/K_max_sched` gate would fail on correct behaviour.
- [ ] **The gate closes at the same fraction on every axis.** `_smooth_bound_v2` at
      `K = K_max` returns ~0.119 for `K_max = 6.59` and for `K_max = 56.7` alike.
- [ ] **Reproducibility.** Identical `(seed, trajset, gains)` gives a bit-identical
      score. `Profiles.resolve_profile` must be called with an explicit `rng`.
- [ ] **`bus.K` initial condition is as configured** — `0.95·K_floor` with
      `start_at_floor = true`, zero with `false`.

## 6. During-run diagnostics to log

`ASMCControllerV2.log_K = true` already records `K_log` and `K_max_sched_log`.

- [ ] `max(K)/K_max_sched` per axis per trajectory — **any value > 1.02 invalidates that
      evaluation**, because it means the physical bound was exceeded.
- [ ] Fraction of ticks with `K` within 2% of `K_max_sched` (authority-saturated).
- [ ] `max_pos` and the **time at which it occurs** as a fraction of `T`. In the prior
      run this landed at 33–100% of `T`, never at startup — confirming `max_pos` measures
      in-run tracking, not a cold-start transient.
- [ ] **`|M_sw|/K` per tick — the single most informative diagnostic.** It is the tanh
      saturation factor: 1.0 means `|s| ≫ ε` and the controller is running **bang-bang,
      never reaching the sliding surface**. Pre-tuning measurements gave a median of
      **1.000 on `coupled_vomega_stress`** and 0.77 on `octagon_stress`, at every `ε`
      from 0.0175 down to 1e-4. If tuning does not bring this well below 1, the ASMC is
      not operating as a sliding-mode controller and no result should be reported as one.
- [ ] `chatter` alongside `tracking`. Higher γ raises `K` toward the ceiling and holds it
      there (`s·tanh(s/ε) ≥ 0`, so the base adaptation only ever *increases* `K`; it
      decays only through the σ-leak at `tau_relax = 2 s`). Chatter is the cost that
      should eventually bound γ from above.

## 7. What is established, and what is not

**Established — correctness properties, independent of tuning:**

- Feedforward equivalence: PID-CT's `M_eq_cmd` matches the ASMC's `M_eq` at zero
  tracking error to 1.4e-14 across all 12 `train_full` references.
- `K` was escaping its derived ceiling by up to 11.7×; now bounded at ≤ 0.888.
- Voltage saturates (31–40% of ticks for the PID); **current never does (0.0%)**, so
  `i_max = 12.8 A` is slack and voltage is the binding actuator limit.
- Every tier reference is feasible: peak `|V_y − h·ψ̇| = 0.694` against the 0.805
  threshold.
- `|M_sw|/K` median is 1.000 on `coupled_vomega_stress` and 0.77 on `octagon_stress` —
  the ASMC is not reaching its sliding surface.

**NOT established — do not carry these into the run or the paper:**

- **Any ASMC-vs-PID performance comparison.** All comparisons so far were at *untuned*
  gains, and the PID's defaults sit within 10–15% of its tuned optimum on x and y while
  the ASMC's are ~6× off (changing γ alone moved `coupled_vomega_stress` from 116.8 to
  19.8). The comparison is meaningless until both are tuned.
- **"The ASMC is authority-limited."** Retracted. On `octagon_stress` it uses 1.0%
  voltage saturation against the PID's 31.3%, slips 2.5× less, and has spare switching
  gain — nothing is limiting it. Low actuator usage is evidence of *unused headroom*,
  which is a tuning statement.
- **"`ε` floors the ASMC's accuracy."** Retracted. Sweeping `ε` 175× (0.0175 → 1e-4) on
  clean runs changed tracking by 1.7–3.5%, because the tanh is already saturated at
  every value. The `ε/λ` floor is an upper bound that applies only *inside* the boundary
  layer, which this controller never enters.
- **`start_at_floor` as an improvement.** Measured inert — identical to 3–4 significant
  figures. `max_pos` occurs at 33–100% of `T`, never at startup.

**PID reference values, for context only** (ψ floor 0.01, 5 seeds, tuned):

```
            clean            noisy
PID-CT   0.578 – 0.601    0.954 – 1.162
PID-FB   0.945 – 0.961    1.474 – 1.536
```

Those PID runs **predate the `vcmd_limits` restructure and the acceleration-convention
fix**, and the gate change alone moved `octagon_stress` from 257 to 0.21 — so they need
re-running before any cross-controller number is quoted.

**Known-hard trajectory:** `spiral_orbit_stress` scores high for every controller
(PID-FB 70.2, PID-CT 16.6, gate-independent). Consider whether it belongs in the
training tier, since it will dominate the objective mean.

## 8. Out of scope

- Re-deriving the `SurfaceNoise` inputs from the ESKF's achieved error. The derivation
  is implemented; feeding it realistic numbers instead of the oracle's injection is a
  prerequisite for the **noisy** stage only, not for this clean run.
- Noise (`--noise noisy`) — clean first, per the staged plan.
- Any change to `kmax_schedule`, `capability_wrench`, `K_floor`, or the decay derivation.
- The PID and MPC tuning runs, which have their own briefs.
