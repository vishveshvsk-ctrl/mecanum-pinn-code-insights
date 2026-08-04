# Estimator v2 — IAE Adaptive-Q Slip Channel (innovation-based online noise adaptation)

> **Generated:** 2026-08-03
> **Stack:** Julia 1.x, StaticArrays, LinearAlgebra, existing `hybrid_ctrl_v2` modules
> **Scope:** Estimator architecture — additive adaptive process-noise scheme, replay
> evaluation only
> **Companion briefs:** `sensors-suite-consolidation-and-physical-noise.md`,
> `estimator-v2-slip-observer-channel.md`, `estimator-v2-ensemble-replay-tuning.md`
> **Origin:** 2026-08-03 architecture discussion — the GM slip model's weak point is that
> τ_slip/σ_slip are one-trajectory point estimates (spin_creep, `estimators_v2.jl:127-143`)
> used as global constants. Instead of replacing the model (the slip-observer brief does
> that), this brief keeps the KF machinery fully stochastic and fixes the *statistics*:
> adapt the slip process noise online from the filter's own innovation sequence.

---

## 0. Preservation constraint — read this first

Identical to the sibling briefs. **Do not edit any of these (FROZEN):**

```
hybrid_ctrl_v2/scheduler_v2.jl
hybrid_ctrl_v2/estimators_v2.jl                     ESKFEstimatorV2
hybrid_ctrl_v2/sensors_v2.jl
hybrid_ctrl_v2/estimator_tuning/harness_v2.jl
hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl
hybrid_ctrl_v2/estimator_tuning/objective_v2.jl
hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl
hybrid_ctrl_v2/estimator_tuning/run_estimator_replay.jl
hybrid_ctrl_v2/runs_estimator_v2_replay/**          read-only reference
hybrid_ctrl/**                                       all v1
```

New files only; extend v1 generics by multiple dispatch (pattern documented at
`estimators_v2.jl:5-17`). The existing replay pipeline must remain runnable and produce
identical results — §8 gates this.

**Relationship to the slip-observer brief:** the two are ORTHOGONAL and must stay
composable — the slip-observer channel adds a measurement; IAE changes Q adaptation.
Neither brief's files may import or depend on the other's. A combined variant is a later
decision pending both compare results (§10).

---

## 1. Motivation and hypothesis

`ESKFEstimatorV2` derives slip process noise from physical constants:
`q_slip = 2·σ_slip²/τ_slip·dt` with measured `τ_slip = 0.06 s`, `σ_slip = 0.014 m/s`
(`derive_process_noise`, `estimators_v2.jl:71-77`). The derivation is principled, but the
*inputs* are a single-trajectory measurement — the struct docstring itself flags this as
"single-trajectory point estimate, likely trajectory-dependent." Across the manifest
(μ = 0.3…0.8, spin_creep through long_circle) the true slip intensity varies by an order
of magnitude, so a globally-frozen q_slip is mis-specified most of the time: too large in
grip (slip states wander, stealing observability from velocity), too small in sustained
slip (filter overconfident — the exact overconfidence → rejected-fix → free-run failure
mode the pivot handoff diagnosed for the IMM line).

The filter already *knows* when its slip model is wrong: the wheel+gyro innovation
ν = e′S⁻¹e (computed at `estimators_v2.jl:306`) is χ²₃-distributed under consistency.
Sustained ν above its expectation means the model under-weights reality (Q too small);
sustained ν below means over-weighting (Q too large).

**Hypothesis:** scaling q_slip online by a factor γ driven by the running innovation
statistic (innovation-based adaptive estimation, IAE — the Sage–Husa family) recovers
most of the slip-regime mismatch *without* any new sensor, observer state, or parametric
commitment — improving in-slip velocity RMSE and slip RMSE while leaving grip-phase
metrics within noise. This is the smallest possible delta on the existing estimator: no
new measurement channel, no new files beyond the estimator + harness + driver, and the
covariance semantics the NIS gates rely on stay fully intact.

**Why IAE and not re-tuning τ/σ on more traces:** broadening the measurement set (the
struct docstring's own suggestion) still yields *constants*; the slip intensity varies
*within* trajectories (grip → slip transitions). Adaptation handles both between- and
within-trajectory variation. The fixed-constant improvement is a worthwhile cheap
follow-up either way but is NOT this brief.

---

## 2. Architecture pattern

**NIS-supervised multiplicative Q adaptation on the slip block only.**

Everything in the v2 tick is unchanged except how `Q[10,10]`/`Q[11,11]` are computed.
One new scalar state `γ` (the slip-Q scale) is maintained by the estimator:

```
init:            γ = 1
each tick:
  prediction & covariance propagation with  Q[10,10] = Q[11,11] = pn.q_slip · γ · slip_q_scale
  wheel+gyro update (unchanged), producing NIS ν (already computed, estimators_v2.jl:306)
  after the update:
    ν̄  ← (1 − λ)·ν̄ + λ·ν        (EMA of the innovation statistic; λ = dt/τ_iae)
    γ  ← clip( γ · (ν̄/3)^κ ,  γ_min ,  γ_max )
```

- `E[ν] = 3` under consistency (3-D wheel+gyro measurement) — so `ν̄/3` is the
  measured-to-expected innovation power ratio; `κ` (the adaptation exponent) sets how
  aggressively γ responds. κ < 1 damps noise-driven jitter in γ.
- Adaptation is **multiplicative and slow**: `τ_iae = 0.5 s` default (≈10× τ_slip) so γ
  tracks slip *regimes*, not per-tick innovation noise. This timescale separation is the
  stability argument — γ must move slower than the filter's own error dynamics or the
  two chase each other.
- Clip bounds PINNED: `γ_min = 1e-2`, `γ_max = 1e4` (γ is bounded so the adaptation can
  never turn the slip channel off or blow up P; these are discipline bounds, not tuned
  physics — same doctrine as the pinned NIS thresholds).
- Only the slip block adapts. Velocity/yaw Q is derived from IMU white noise (physical);
  bias Qs are physical; pose Q stays tunable. Adapting anything else re-opens the
  unphysical-covariance failure mode the "derive, don't search" doctrine eliminated.

**Optional variant `:sage_husa`** (implement behind a flag, evaluate only if the EMA
scheme shows promise but insufficient response): windowed covariance matching on the
slip block over W = 200 ticks —

```
Q̂_ss = (1/W)·Σ_k ( K_s·e·e′·K_s′ + P⁺_ss − (F·P⁻·F′)_ss )
```

**Optional variant `:stf`** (strong-tracking / adaptive-fading filter — same
innovation-driven family, different lever): instead of scaling Q, inflate the prior
covariance by a fading factor λ_k ≥ 1 on the slip block before the update,
`P⁻ ← λ_k·P⁻` (slip rows/cols only), with λ_k computed from the innovation
orthogonality condition (the standard STF recursion, soft-clipped to [1, λ_max] with
λ_max pinned at 100). Equivalent in spirit to the γ scheme — where γ acts on Q
(persistent process-noise level), the fading factor acts on P directly (immediate
gain boost); STF typically reacts faster and decays automatically as innovations
whiten. Literature precedent for slip/state estimation under skid: Lv et al., terrain
vision-aided ICR estimation for skid-steering rovers, which applies STF to the robot
kinematic states. Same pinned-bound discipline as γ; select via `iae_kind`, never
simultaneous.

projected to its diagonal, symmetrized, floored at `pn.q_slip·γ_min`, capped at
`pn.q_slip·γ_max`, where K_s is the slip rows of the Kalman gain. Heavier (needs history
of K and P), noisier per-window, but a genuine estimator of Q rather than a controller
of ν̄. If both are implemented, the flag selects; do not run both simultaneously.

### Why not the alternatives

- **ν-triggered binary Q switching** (grip/slip two-level): a special case of the EMA
  scheme with κ → ∞; the smooth EMA avoids the chattering the IMM line already suffered
  from.
- **Adapting R instead of Q:** R is *derived from sensor physics* — an innovation
  mismatch in this architecture means the process model is wrong, not the sensor spec.
  Adapting R would contradict the brief §7.3 doctrine that R carries provenance.
- **Full Sage–Husa on all of Q:** re-opens black-box covariance tuning; rejected by the
  same doctrine.

---

## 3. File 1 — `hybrid_ctrl_v2/estimators_v2_iae.jl` (NEW, ~280 lines)

Module `EstimatorModV2IAE`. Header documents the include contract (after
`estimators_v2.jl`; extends `Main.EstimatorMod.estimator_update!` / `apply_pose_fix!`).

Export: `ESKFIAEEstimatorV2`, `init_eskf_iae_v2!`, `apply_flow_iae!`.

### 3.1 Struct

`Base.@kwdef mutable struct ESKFIAEEstimatorV2` — field-identical copy of
`ESKFEstimatorV2` (`estimators_v2.jl:121-182`, same defaults, same τ/σ provenance
comment copied) plus:

```julia
    # --- IAE adaptive slip-Q (brief §2) ---
    use_iae::Bool       = true     # ablation: false ⇒ identical to ESKFEstimatorV2
    iae_kind::Symbol    = :nis_ema # :nis_ema | :sage_husa (§2 optional variant)
    tau_iae::Float64    = 0.5      # s — adaptation timescale (EMA), tunable
    kappa_iae::Float64  = 0.5      # adaptation exponent, tunable (mild response)
    gamma_min::Float64  = 1e-2     # PINNED discipline bounds
    gamma_max::Float64  = 1e4

    # --- IAE internal state ---
    gamma_q::Float64  = 1.0        # current slip-Q scale
    nu_bar::Float64   = 3.0        # EMA of wheel-update NIS (init at E[ν]=3: neutral)
```

### 3.2 Functions

- `init_eskf_iae_v2!(est, params)` — mirror `init_eskf_v2!` (`estimators_v2.jl:187-214`)
  exactly, then reset `gamma_q = 1.0`, `nu_bar = 3.0`.
- `Main.EstimatorMod.estimator_update!(bus, y, est::ESKFIAEEstimatorV2, params, dt)` —
  duplicate the v2 tick body (`estimators_v2.jl:225-338`) with exactly two changes:
  1. Q assembly (v2 lines 285-286): `Q[10,10] = Q[11,11] = pn.q_slip · est.gamma_q ·
     slip_q_scale` — the ONLY change to prediction.
  2. after the wheel+gyro update's NIS computation (v2 line 306 computes `nu`; the
     adaptation uses the PRE-boost ν, i.e., the value before any `r_boost` inflation —
     adaptation must respond to the honest innovation, not the boosted one):
     `if est.use_iae` → the §2 EMA + γ update (and, if `iae_kind == :sage_husa`, the
     windowed estimator instead — history buffers as additional internal fields, sized
     W = 200, documented).
  When `use_iae == false` the method must execute exactly the v2 sequence, bit-for-bit
  (§8 regression gate).
- `apply_flow_iae!(bus, est::ESKFIAEEstimatorV2, m, z_flow, params)` and
  `Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFIAEEstimatorV2, fix, z_fix)` — bodies
  identical to v2's (`estimators_v2.jl:356-416`). No reanchor coupling (unlike the
  slip-observer brief — IAE has no open-loop integration state).

`bus.xhat`/`bus.d_hat` writes identical to v2.

---

## 4. File 2 — `hybrid_ctrl_v2/estimator_tuning/harness_v2_iae.jl` (NEW, ~110 lines)

Module `HarnessV2IAEMod`. Include-after: `harness_v2.jl` + `estimators_v2_iae.jl`.

- `build_estimator_v2_iae(est_cfg, suite)` — kwarg pass-through mirroring
  `build_estimator_v2` (`harness_v2.jl:52-60`) + `:use_iae, :iae_kind, :tau_iae,
  :kappa_iae, :gamma_min, :gamma_max` (+ the measured-constant overrides).
- `run_and_log_replay_v2_iae(est_cfg, traj_entry, suite; seed=42, rate_hz=1000.0,
  data_dir=...) -> EstimatorLogV2` — thin copy of `run_and_log_replay_v2`
  (`harness_v2.jl:226-305`), building via `build_estimator_v2_iae` and calling
  `apply_flow_iae!`. Reuse `Main.HarnessV2Mod._load_replay_data_v2`,
  `_interp_scalar_v2`, `EstimatorLogV2`, `REPLAY_CACHE_V2` unchanged (verify the module
  name at `harness_v2.jl`'s `module` line; do not guess). `est.wheel_H` field name kept
  (`harness_v2.jl:298` logging contract). No `EstimatorLogV2` change — γ is diagnostic
  only; if γ traces are wanted, print summary stats (final γ, time-in-γ>10) to the
  compare script's stdout, do not change the log struct.

---

## 5. File 3 — `hybrid_ctrl_v2/estimator_tuning/param_space_v2_iae.jl` (NEW, ~60 lines)

Module `ParamSpaceV2IAEMod`. Include-after: `param_space_v2.jl`.

- `eskf_iae_param_space_v2()`: the existing 10 dims reused verbatim from
  `ParamSpaceV2Mod.eskf_param_space_v2()` (call and append) plus 2 new dims:

  | dim       | len | scale | bounds       | why |
  |-----------|-----|-------|--------------|-----|
  | tau_iae   | 1   | log   | 0.05 .. 5.0  | adaptation timescale [s] — must stay slower than filter dynamics |
  | kappa_iae | 1   | linear| 0.1 .. 1.0   | adaptation exponent — response aggressiveness |

  12 dims total, `ParamSpace(:eskf_iae_v2, dims)`. γ bounds stay pinned.
- `apply_params_v2_iae!(theta, space)` — mirror `apply_params_v2!`
  (`param_space_v2.jl:86-111`); returns the 10 shared keys + `tau_iae, kappa_iae,
  use_iae=true, use_dhat=false`. Note `kappa_iae` is the first LINEAR-scale dim in the
  v2 spaces — copy the scale-handling from `apply_params_v2!` (it already branches on
  `:log` vs linear via `flat_scale`; just confirm the branch handles it).

---

## 6. File 4 — `hybrid_ctrl_v2/estimator_tuning/compare_iae_baseline.jl` + `.bat` (NEW)

First thing to run — fixed defaults, no tuning.

Per trajectory of the frozen 11-trajectory manifest (`replay_trajset.jl:46-60`), with
the seed's `runs_estimator_v2_replay/seed<S>/best_config.json` as the 10 shared params:

1. Regression gate (once, first trajectory): `ESKFIAEEstimatorV2` with `use_iae=false`
   vs `ESKFEstimatorV2`, same params/suite/seed; assert `max|Δv_hat| < 1e-12`,
   `max|Δpose_hat| < 1e-12`. Hard gate.
2. Baseline run (`Main.HarnessV2Mod.run_and_log_replay_v2`).
3. IAE run (`run_and_log_replay_v2_iae`, `iae_kind=:nis_ema`, defaults
   `tau_iae=0.5, kappa_iae=0.5`).
4. Score both with `ObjectiveV2Mod` terms (read `objective_v2.jl` for entry points) +
   slip RMSE per axis (true signed slip `Hw \ ω_true − v_true` from cached replay data;
   estimated = `d_hat_log[1:2, :]`). Also report γ summary per trajectory (final γ,
   fraction of ticks with γ > 10) — this is the diagnostic showing whether adaptation is
   actually engaging in slip regimes and staying near 1 in grip.

CLI: `--seed S` (default 1), `--out runs_estimator_v2_iae_replay/compare_seedS/`.
Output `compare_table.csv` + stdout table, manifest mean/median summary rows.
Companion `compare_iae_baseline.bat` per AGENTS.md §6 (copy the structure of an existing
`hybrid_ctrl_v2/run_stage_*.bat`; `cd /d` to the Windows tree root, `julia --project=. …`).

**Decision rule (in the script header):** hypothesis holds if in-slip velocity RMSE and
slip RMSE improve on high-slip entries (spin_creep, μ=0.3) while grip-phase metrics stay
within the ~5% eval-noise floor, AND the γ diagnostic confirms regime-tracking (γ rises
in slip, ≈1 in grip — otherwise the improvement is accidental). If γ never moves, the
innovation carries no usable slip-regime signal and the approach is dead — archive and
report, do not increase κ blindly.

---

## 7. File 5 — `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_iae.jl` + `.bat` (NEW)

Only if §6 is positive. Mirror `run_estimator_replay.jl` (read it; copy its CLI/optimizer
structure: `--seed`, `--p1-cap`, `--p2-cap`, `--smoke`, dxNES → BOBYQA) with:

- param space = `eskf_iae_param_space_v2()` (12 dims); objective = same terms as
  `make_replay_objective_v2` calling `run_and_log_replay_v2_iae` (thin
  `make_replay_objective_v2_iae` — reuse `ObjectiveV2Mod` helpers; do NOT edit
  `objective_v2.jl`).
- warm-start `--init-from runs_estimator_v2_replay/seedS/best_config.json`: 10 shared
  dims at the baseline optimum, 2 IAE dims at defaults.
- output `hybrid_ctrl_v2/runs_estimator_v2_iae_replay/seed<N>/` — new directory; never
  write into `runs_estimator_v2_replay/`.
- companion `run_estimator_replay_iae.bat`.

---

## 8. Verification gates

1. **Static:** include order in every header; dispatch signatures match the v1 generics;
   module names verified against actual `module` lines (read, don't guess).
2. **Regression:** §6 step 1 — `use_iae=false` bit-for-bit identical to
   `ESKFEstimatorV2` (< 1e-12). Hard gate.
3. **Frozen pipeline:** `run_estimator_replay.jl --smoke` after landing; identical
   results; `git diff` shows only new files.
4. **Hypothesis test + γ diagnostic:** §6 decision rule.
5. **Stability sanity:** across the whole manifest, assert `gamma_q` never hits the clip
   bounds for > 5% of any trajectory's ticks — persistent clipping means the adaptation
   is fighting the model, which is a finding, not a tuning opportunity.

---

## 9. Execution order

1. `estimators_v2_iae.jl`
2. `harness_v2_iae.jl`, `param_space_v2_iae.jl`
3. `compare_iae_baseline.jl` + `.bat` → run (gates 2, 4, 5)
4. `run_estimator_replay_iae.jl` + `.bat` → only if positive
5. Handoff `chat-handoff/estimator_v2_iae_handoff.md` (compare table, γ diagnostics,
   gate results); commit new files + handoff. Run outputs committed only after
   completion (AGENTS.md: git is transport + post-run archive).

## 10. Out of scope

- Combining IAE with the slip-observer channel (later decision, needs both compare
  results; the combined estimator would be a THIRD new file reusing both, never an edit
  to either).
- `:sage_husa` windowed variant beyond the flagged implementation (evaluate only if
  `:nis_ema` is promising-but-slow).
- Adapting R, pose Q, or any non-slip Q block (doctrine: R carries provenance; §2).
- Closed-loop integration; NN-based anything; GPU anything.
- Re-measuring τ_slip/σ_slip on more traces (separate cheap follow-up, not this brief).

---

## Key file:line references for the executor

| What | Where |
|---|---|
| ESKFEstimatorV2 struct + defaults + τ/σ provenance | `hybrid_ctrl_v2/estimators_v2.jl:121-182` |
| v2 tick body to duplicate (NIS ν at line 306) | `hybrid_ctrl_v2/estimators_v2.jl:225-338` |
| derive_process_noise (q_slip derivation) | `hybrid_ctrl_v2/estimators_v2.jl:51-77` |
| flow / pose-fix update patterns | `hybrid_ctrl_v2/estimators_v2.jl:356-416` |
| build_estimator_v2 | `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl:52-60` |
| replay loop to copy | `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl:226-305` |
| 10-dim space + decoder (scale branch) | `hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl:61-111` |
| 11-trajectory manifest | `hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl:46-60` |
| objective terms + in-slip mask | `hybrid_ctrl_v2/estimator_tuning/objective_v2.jl:33-145` |
| baseline results (read-only) | `hybrid_ctrl_v2/runs_estimator_v2_replay/seed1..5/best_config.json` |
