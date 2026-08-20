# The v3 controller/estimator campaign — what was run, and why

This document explains the v3 simulation campaign **from scratch**: the question it
answers, the system it simulates, the protocol it follows, and how to read its results.
It is the narrative companion to [`RESULTS_v3.md`](RESULTS_v3.md), which is the living
document of record for the numbers. Read this file for the *intent*; read RESULTS_v3.md
for the *evidence*.

---

## 1. The question

The project (Mecanum PINN digital twin, KUKA youBot four-Mecanum platform, IMECE 2026)
needs a defensible answer to:

> **Which control law should the omnidirectional platform fly — and how much of its
> performance survives real state estimation?**

Three candidate laws are compared, chosen so the comparison is a **two-step ablation a
reviewer cannot fault**:

| controller | structure | what it isolates |
|---|---|---|
| **PID-FB** | cascaded IMC-reparameterized PID; 18 gains collapse to 3 tuned `lam_inner` per axis, everything else derived from the plant model (pole cancellation); pure feedback | the industrial baseline |
| **PID-CT** | the same cascade **plus computed-torque feedforward**, structurally matched to ASMC's equivalent control | FB→CT measures *the value of the model* |
| **ASMC v2** | adaptive-gain supertwisting sliding mode with physically derived gain bounds (friction-circle ceiling recomputed per tick) | CT→ASMC measures *the value of adaptation* |

All three are tuned by the same black-box optimizer against the same scalar objective,
on the same trajectory tiers, and then evaluated through the same frozen state
estimator. Anything that makes one controller's objective differ from another's is
treated as a bug, not a detail.

MPC was part of the original plan and is **deferred out of this iteration** (user
decision, 2026-08-19): it diverges on 4 velocity-demanding trajectories, and its
`use_ltv=false`-is-better anomaly (backwards from theory) remains unresolved. The
three-way comparison is quotable without it.

## 2. The system under test

**Plant.** The 30-state (quasi-static) voltage-input plant in
`hybrid_ctrl/plant.jl`: slip–spin–LuGre contact physics (per-wheel bristle states,
roller spin), driven through a **DC-motor model with back-EMF** — wheel torque
`τ = G·η·(K_t·i − τ_f)`, `i = (V − K_b·G·ω)/R_a`, bus voltage `|V| ≤ 24 V` with a
200 V/s slew clamp in the mixer. This is the "newer plant": actuator authority falls
linearly with wheel speed and vanishes at the no-load speed 27.73 rad/s, which is the
hard feasibility wall the trajectory tiers are screened against (§3).

**Sensors** (`:realistic` suite, `sensors_v2.jl`): IMU and wheel encoders at 1000 Hz,
optical flow at 100 Hz with 2% dropout, 100 Hz pose fixes with per-run biases and 0.5%
outliers — industrial white noise plus deliberately uncalibrated consumer-grade biases.
The estimator never sees truth; the controllers in the deployable configuration never
see truth either.

**Estimator.** `ESKFEstimatorV3` — a 13-state error-state Kalman filter with a
yaw-acceleration random-walk state (a type-2 tracker, added to kill a structural
yaw-rate lag). It is **tuned once and frozen** before any controller comparison
(*estimator-first methodology*, §5), so that a control-law difference can never be an
observer difference in disguise.

**Feedback regimes — three, not two.** This distinction runs through the whole
campaign and getting it wrong inverts conclusions:

1. **clean oracle** — perfect state feedback. An upper bound on what any controller
   can do.
2. **noisy oracle** — true state **plus raw, unfiltered sensor noise** written directly
   into the controller's state estimate. There is *no estimator* in this path; it is a
   pessimistic "what if we never filter" bound, not a deployment model.
3. **ESKF** — the deployable path: `:realistic` sensors through the frozen filter.

The ESKF beats the noisy oracle *by construction* (a tuned filter beats raw noise), so
"ESKF vs noisy oracle" is **not** the cost of state estimation — that cost is ESKF vs
*clean* oracle.

## 3. Feasibility first: the v3 trajectory tiers

The campaign's predecessor tier (`train12`) contained trajectories the hardware cannot
fly, and the tuner was spending most of its budget on them: `spiral_orbit_stress`
alone carried 72–81% of the training tracking term while sitting at 85.4% of the
motor's no-load speed — a trajectory no control law recovers. The v3 tiers
(`train14_v3`, `test_v3`, in `controller_tuning/trajsets.jl`, whose header is the
authoritative change log) screen every candidate reference against **two walls**:

1. **Motor / back-EMF (hard):** max per-wheel speed `(Vx ± Vy ± (l+h)·ψ̇)/R` against
   the no-load speed `V_max/(K_b·G) = 27.73 rad/s`. Past ~85% of no-load there is no
   torque authority to arrest error growth, for any gain law. v3 targets ≤70% for
   margin. (This wall is derived in
   `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope_BackEMF.tex`.)
2. **Friction circle (soft):** per-wheel contact utilisation. Brief excursions are
   momentary slip the LuGre plant absorbs — measured to be harmless — so this wall
   informs but does not veto.

Two entries were replaced and two added; every reference is evaluated over its own
full `T_total` (9.5–85.3 s) after an earlier audit truncated long references at 12 s
and picked a combo that was 98.5% of no-load over its true duration. The controller
comparison is only meaningful where the hardware can actually go; that is the entire
point of the tier rebuild.

## 4. The objective — one scalar, three terms

Every tuning and eval run minimizes

```
score = tracking_v3 + 0.05·(ce/V_MAX) + 0.36·(chatter/CHATTER_REF)
```

- **tracking_v3** — six tolerance-normalised terms over two channels (position and
  heading × terminal / peak / typical), divided by a per-trajectory scale `k_traj`
  (the path's own extent, rotation-invariant). The v2 metric scored **four samples out
  of 12k–60k** (terminal and argmax only) and was blind to mid-path oscillation; the
  time-normalised integral ("typical") term is the fix, with its tolerances
  *calibrated from measured achieved ratios*, not chosen. A term of 1.0 means
  "exactly at target".
- **ce** — mean summed wheel-voltage magnitude (energy proxy).
- **chatter** — mean per-tick total variation of the wheel-voltage command, priced
  against `CHATTER_REF = 0.8 V/ms`, the mixer's own slew clamp. This term exists
  because an entire arc of v2 tuning (three box-widening rounds) kept railing on a
  chatter lever the objective could not price — chatter was being normalised by
  `V_MAX` (a voltage dividing a slew rate, ~30–100× too weak), so any natural λ left
  it silently inert.

`LAMBDA_CHATTER_V3 = 0.36` is **balance-preserving, not re-derived**: it reproduces the
29.9% tracking share that the accepted v2 objective had at its converged point. The
objective code *refuses* `lambda_chatter > 1.44` under `--metric v3` so a carried-over
v2 value cannot pass silently, and the guard itself is unit-tested both ways.

**The estimator has its own, separate objective** (`estimator_tuning/objective_v2.jl`):
a *replay* score (estimator vs ground truth on logged trajectories, no controller in
the loop), same tolerance-normalised shape, with velocity error during slip priced
twice (`lam_slip = 1.0`) — the point of an estimator on this platform is that its
model mismatch *is* slip. It converged to ~71% a velocity-estimation objective, which
matches what the controllers' inner loops actually consume.

## 5. The protocol — five stages, in order

1. **Estimator tuning (first, then frozen).** 5 seeds over a 7-dim log-scaled gain
   space, replay objective, `:realistic` sensors, `train12` tier. All 5 seeds converge
   to the same estimator within 0.23% on a common noise realisation (the apparent 3.3%
   tuning-score spread is realisation variance, not quality); seed 4 adopted and
   **frozen for the whole campaign** — including for tiers it was not tuned on.
   Re-freezing per tier would reintroduce the confound the freeze exists to remove.
2. **Feasibility screening** of trajectory tiers (§3).
3. **Clean controller tuning** — 5 seeds each, cold start, oracle-clean feedback,
   `train14_v3`, convergence by plateau detection (not a fixed budget: the v1 tuner
   was caught terminating mid-descent, which invalidated its scatter claims).
4. **Noisy controller tuning** — 4 seeds each, **warm-started from the same run's
   clean optimum**, feedback = noisy oracle (unfiltered injected noise; ASMC's noise
   scales populated from the tuned ESKF's measured in-loop errors). Run for all three
   controllers or none, so a noisy-tuned ASMC is never quoted against clean-tuned PIDs.
5. **Evaluation under three feedback regimes** — the six converged configs
   (3 controllers × clean/noisy-tuned) on fresh sensor seeds 101–105: oracle-clean,
   oracle-noisy, and through the frozen ESKF (420 runs: 6 × 14 trajectories × 5 seeds,
   zero non-converged). The held-out **`test_v3` leg** (generalisation of the final
   estimator+controller stack to unseen trajectories) completes the campaign:
   288 oracle sims + 240 ESKF runs — see §6 and RESULTS_v3.md §9.

Throughout: 5 independent OS processes (the pipeline has global mutable state —
threading is unsafe), BLAS pinned to 1 thread for determinism, one process per seed,
`stop_reason` recorded per run. Outputs under `runs_*/` are never hand-edited; an
existing `best_config.json` is a converged seed.

## 6. Headline results

The full tables and their caveats live in RESULTS_v3.md; the structural findings are:

- **PID-CT wins every regime** — clean oracle (0.14340 vs ASMC 0.15318), noisy oracle,
  and frozen-ESKF closed loop (0.34127; paired 69/70 and 70/70 against ASMC and
  PID-FB). The ordering PID-CT < PID-FB < ASMC has no seed overlap.
- **Noisy tuning does not transfer to deployment.** It helps against the noisy oracle
  it was tuned on (ASMC 5/5, CT 5/5 paired) but the advantage vanishes through the real
  ESKF, and it *damaged* ASMC's heading ~2× everywhere. Deploy clean-tuned configs.
- **Real state estimation costs ~+0.20 of score** against clean oracle, essentially
  identical across all six configs (42.9–45.1% of the clean→unfiltered gap recovered)
  — the estimator's contribution is controller-independent, which is exactly what makes
  a frozen observer a fair basis for comparison.
- **The estimator error budget is a three-stage story**: sensor injection (identical
  across configs by construction) → ESKF survival (74–79% of injected velocity error
  removed, flat pose removal at ~67.6% on every trajectory) → controller tracking
  (where essentially *all* inter-controller spread lives). The estimator is common
  mode; only the tracking stage discriminates control laws.
- **CT's one weakness is heading under sustained yaw**: best heading on 9 of 14
  trajectories (~1.1 mrad), collapsing to 29.8/37.6 mrad on exactly the two
  sustained-yaw stress trajectories — the same two where its yaw integral clamp fires
  on 17–20% of ticks (0.0% elsewhere). Three independent measurements agree.
- **On 7 of 14 trajectories, tracking error is at or below the estimator's noise
  floor** (~3 mm) — further controller improvement there is *unmeasurable* with this
  sensor suite, so the comparison is decided by the handful of trajectories where
  tracking rises clear of it.
- The ESKF's **heading channel is its best** (6.9× error removal) once an 11 ms
  startup transient on non-zero-initial-heading references is accounted for; its
  genuinely weak channel is **yaw rate** (~52% removal — the gyro is its only source).

**Generalisation (`test_v3`, RESULTS_v3.md §9):**

- **PID-CT generalises as the winner — unambiguously.** On the 7 genuinely held-out
  trajectories (3 on profiles never seen in tuning), clean-tuned PID-CT beats ASMC and
  PID-FB **35/35 paired** each, and is first on every individual held-out trajectory.
  This is the campaign's most robust result.
- **The ASMC-vs-PID-FB ordering does not generalise.** Their training-tier gap was
  carried by a single trajectory (`coupled_vomega_stress`, 66% of ASMC's gap); excluding
  it they are statistically indistinguishable (23/35, 8/35). The defensible claim is
  **"ASMC and PID-FB are equivalent, and PID-CT beats both."**
- **The tier's anchor caveat:** `coupled_vomega_anchor` is combo-identical to the
  training tier's `coupled_vomega_stress` — a deliberate cross-tier consistency check
  that passes bit-exactly (±0.000000) — and must be excluded from every generalisation
  claim. It is also the worst-tracked entry in either tier, so leaving it in would
  dominate any mean.
- **The estimator is tier-invariant:** every error-budget channel lands within ~3
  percentage points of its training value (the pose fix is an absolute measurement; the
  observer was never refit). On all 7 held-out trajectories every controller tracks to
  1.6–4.2 mm / 1.0–2.5 mrad — the collapses are entirely on the anchor, i.e. on a
  *training* trajectory, consistent with CT's yaw-clamp finding being a sustained-yaw
  property, not a generalisation failure.

## 7. How to read any number from this campaign

1. **Never mix tuning-stage scores with eval scores** — they run on different noise
   realisations, and their ranges overlap.
2. **Pair, don't average, across realisations** — realisation variance (~6–7% eval,
   ~16% tuning) swamps controller differences (~2–4%). Every ordering claim is a paired
   5/5 or 4/4, never a difference of marginal means.
3. **Best-seed selection under noise is confounded** — seed 4 is simultaneously the
   best seed and the easiest realisation for all three controllers; applied
   identically, so cross-controller comparison holds, but absolute improvements are
   flattered.
4. **Absolute values, never % degradation** — a better clean tracker reads a worse %
   for the same absolute result.
5. **Three feedback regimes, not interchangeable** — §2.

## 8. Reproduction map

| piece | where |
|---|---|
| plant (untouchable v1) | `hybrid_ctrl/plant.jl`, `scheduler.jl`, `mixer.jl` |
| v2 controllers + search spaces | `hybrid_ctrl_v2/controllers_v2.jl`, `tune_controller_v2.jl` |
| objective + trajectory tiers | `hybrid_ctrl_v2/controller_tuning/stage_objective.jl`, `trajsets.jl` |
| tuning drivers | `run_stage_asmc_v3.bat`, `run_stage_pid_v3.bat` (`<clean\|noisy>`, noisy warm-starts) |
| estimator tuning | `hybrid_ctrl_v2/estimator_tuning/` (objective_v2, param_space_v4, replay runner) |
| frozen estimator config | `runs_estimator_v4_mu0p5_train12/seed4/best_config.json` |
| converged controller configs | `runs_asmc_v3/`, `runs_pid_v3/` |
| closed-loop ESKF eval | `eval_v3_eskf.jl` (+ `.bat`), data `runs_eskf_v3_train14/` |
| diagnostics | `diag_v3_pid_imax_binding.jl`, `diag_v3_error_budget.jl` |
| provenance/verification | `metric_v3_{calib,verify}.jl`, `scalers_v3.jl`, `lambda_v3_verify.jl`, `v3_verify_final.jl` |
| feasibility theory | `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope_BackEMF.tex` |
| serialised results | `results_v3/*.jls` |

## 9. Deliberately not done

- **MPC** — deferred (§1).
- ~~**`test_v3` generalisation**~~ **DONE** (§6, RESULTS_v3.md §9): PID-CT generalises
  35/35; ASMC/PID-FB tie. Caveat: the tier yields 7 held-out trajectories, not 8
  (anchor is combo-identical to a training entry).
- **CT yaw integral budget sensitivity** — the clamp is measured to *bind*; that it
  *harms* is not established.
- **ESKF yaw-rate re-weighting** — interacts with the CT heading finding; the obvious
  next experiment.
- **`psi_hat` initialisation from the first pose fix** — removes the 11 ms startup
  transient that makes `est_heading` incomparable across mixed-`psi(0)` tiers.
