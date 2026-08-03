# Estimator v2 — Ensemble Replay Tuning (expanded trajectory set, multi-realization noise)

> **Generated:** 2026-08-02
> **Stack:** Julia 1.x, Distributed, Random, Arrow.jl, existing `hybrid_ctrl_v2` modules
> **Scope:** Estimator parameter tuning only — replay path, additive
> **Companion briefs:** `sensors-suite-consolidation-and-physical-noise.md`, `pcrlb-no-odometry-ellipse-bound.md`

---

## 0. Preservation constraint — read this first

The closed-loop estimator+controller simulation path is **provable capability and must not
be touched**. These files are **FROZEN**:

```
hybrid_ctrl_v2/scheduler_v2.jl                     run_hybrid_v2 — full ODE + callbacks
hybrid_ctrl_v2/estimators_v2.jl                    ESKFEstimatorV2
hybrid_ctrl_v2/sensors_v2.jl                       SensorModV2, build_suite
hybrid_ctrl_v2/estimator_tuning/harness_v2.jl      run_and_log_v2 AND run_and_log_replay_v2
hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl  eskf_param_space_v2, apply_params_v2!
hybrid_ctrl_v2/estimator_tuning/objective_v2.jl    make_replay_objective_v2
hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl  the 11-trajectory manifest
hybrid_ctrl_v2/estimator_tuning/run_estimator_replay.jl
hybrid_ctrl/**                                      all v1
```

Everything in this brief is a **new file that calls the existing API unchanged**. The key
enabling fact: `HarnessV2Mod.run_and_log_replay_v2(est_cfg, tr, suite; seed)` already
accepts a caller-supplied `suite` and `seed`, so an ensemble objective can invoke it *N*
times with *N* independently-seeded suites without a single edit. This is the same
never-edit-the-parent discipline `harness_v2.jl` itself follows relative to
`tuning/harness.jl`, and that `controllers_v2.jl` follows relative to
`hybrid_ctrl/controllers.jl`.

**The existing 11-trajectory single-realization pipeline must remain runnable and produce
identical results after this work lands.** §10 makes that a gate.

---

## 1. Overview

Estimator tuning already runs in **replay**: `objective_v2.jl` states outright *"no ODE
solve"* — candidates are scored by running `ESKFEstimatorV2` over sensor streams
synthesized from pre-existing Arrow trajectories in `data/`. That is the right
architecture and this brief does not change it.

What it changes is the **statistical design of the objective**, which currently has three
weaknesses:

1. **11 trajectories.** Selected by hand and limited by verification effort, not
   availability — the main sweep holds ~13,151 Arrow files. Trajectory count is the
   cheapest available lever on generalization, because in replay a trajectory costs only
   replay time; the simulation already happened.
2. **One noise realization per candidate.** `make_replay_objective_v2` takes a single
   `seed::Int=42`. The `seed1…seed5` directories under `runs_estimator_v2_replay/` are
   **five independent tuning runs**, not an average inside the objective. So the current
   objective estimates the loss under *one* noise draw, not its expectation over noise.
3. **No held-out noise.** With common random numbers (correctly) holding realizations
   fixed across candidates, nothing detects tuning to the *specific* bias walks in the
   chosen draws.

The new design: an expanded stratified trajectory set, *R* hash-derived noise realizations
averaged inside the objective, a disjoint held-out realization set for validation, and a
noise-free run kept **outside** the objective as a model-vs-noise diagnostic.

**Contract:** `theta::Vector{Float64}` (10 ESKF parameters from `eskf_param_space_v2`) →
`(score, per-trajectory metrics, per-realization spread)`.

---

## 2. Architecture Pattern

**Nested ensemble with common random numbers and a held-out noise split.**

```
outer : tuning seed  S           (5 runs — independent replications of the whole tuning)
  ├── training   realizations k = 1..3      derived as hash((S, modality, k))
  ├── validation realizations k = 101..103  disjoint index range, never scored
  └── clean      realization  k = 0         noise disabled, diagnostic only

inner : for each candidate theta
          for each trajectory (parallel, partitioned by trajectory)
            for each training realization k
              run_and_log_replay_v2  ->  metrics
          score = aggregate over (trajectory, k)
```

Justification for each layer:

- **Averaging realizations inside the objective** turns "best parameters for this noise
  draw" into "best parameters in expectation over noise" — the quantity actually wanted.
- **CRN (fixed realizations across candidates within a run)** is already in
  `objective_v2.jl` and must be preserved. Without it, candidate-to-candidate differences
  are dominated by noise-draw luck rather than parameter quality. This is the single
  largest variance-reduction device in simulation optimization and it is free.
- **Disjoint held-out realizations** are the only guard against overfitting to the noise
  rather than the trajectories.
- **Clean run outside the objective** — see §9, this one is a trap if scored.

---

## 3. Technology Constraints

- **Julia:** 1.x, existing `Project.toml`/`Manifest.toml`. **No new dependencies.**
- **Required (all already present):** `Distributed`, `Random`, `Arrow`, `StaticArrays`,
  `LinearAlgebra`, `Statistics`, `JSON3`/`CSV` for run artifacts
- **Device target:** CPU only.
- **Explicit exclusions:**
  - **No GPU.** Verified out of scope this iteration (§11). If revisited, the blocker is
    that a 12×12 covariance recursion over ~20,000 steps needs fp64, and this machine's
    RTX 3060 runs fp64 at 1/32 rate (~0.4 vs ~13 TFLOPS). A GPU port would first require
    converting `ESKFEstimatorV2` to **square-root (UD or Cholesky-factored) form** so it
    is numerically sound in fp32. That is a filter rewrite, not a port.
  - **No edits to any file in §0.**
  - **No recursive search of `../data/`** — see §9, this is a hard rule.

---

## 4. Component Breakdown

All new files live in `hybrid_ctrl_v2/estimator_tuning/`.

### `NoiseRealizationsMod` / `noise_realizations.jl`
- **Type:** `module` (pure functions, no state)
- **Responsibility:** Derive independent, reproducible per-modality RNG seeds for a given
  (tuning seed, realization index) pair.
- **Inputs:** `seed::Int`, `k::Int` (realization index), `modality::Symbol`
- **Outputs:** `UInt64` seed, and a `NamedTuple` of per-modality seeds
- **Depends on:** nothing

### `ReplayTrajSetExtMod` / `replay_trajset_ext.jl`
- **Type:** `module` (plain data + one verification function)
- **Responsibility:** The expanded stratified trajectory manifest, and an **O(1)-per-entry**
  existence check that never walks `data/`.
- **Inputs:** `run_dir::AbstractString`, `data_dir::AbstractString`
- **Outputs:** `Vector{NamedTuple}` in the *identical* schema as `replay_trajset()`
  (`name`, `combo_idx`, `mu`, `config_dir`, `ref_type`, `run_mode`, `role`) plus a new
  `strata::NamedTuple` field carrying the stratification coordinates
- **Depends on:** `DataStore.expected_output` (read-only use)

### `EnsembleObjectiveMod` / `objective_ensemble_v2.jl`
- **Type:** `module`
- **Responsibility:** Multi-realization replay objective — the drop-in replacement for
  `make_replay_objective_v2` that averages over realizations.
- **Inputs:** `space`, `trajs`, `seed`, `realizations::Vector{Int}`, scoring kwargs
- **Outputs:** closure `theta::Vector{Float64} -> NamedTuple` with a `.score` field
  (required by `StageOptimizerMod.optimize_staged`) plus per-realization detail
- **Depends on:** `NoiseRealizationsMod`, `HarnessV2Mod.run_and_log_replay_v2`
  (**unmodified**), `EstimatorObjectiveV2Mod.estimator_objective_abs_v2` (**unmodified**),
  `ReplayParallelMod`

### `ReplayParallelMod` / `replay_parallel.jl`
- **Type:** `module`
- **Responsibility:** Distribute `(trajectory × realization)` replays across workers,
  **partitioned by trajectory** so each Arrow file is loaded once per worker.
- **Inputs:** `trajs`, `realizations`, `est_cfg`, `seed`, `nworkers`
- **Outputs:** `Vector{EstimatorLogV2}` (or the `Inf`-sentinel on failure), in a
  deterministic order independent of completion order
- **Depends on:** `NoiseRealizationsMod`, `HarnessV2Mod.run_and_log_replay_v2`

### `ValidationReportMod` / `validate_realizations.jl`
- **Type:** `module`
- **Responsibility:** Post-tuning evaluation of a fixed `theta*` on the held-out
  realizations and on the clean run; emits the noise-generalization and
  model-vs-noise-error reports.
- **Inputs:** `theta_star::Vector{Float64}`, `space`, `trajs`, `seed`
- **Outputs:** a report `NamedTuple` + CSV/JSON artifacts
- **Depends on:** `EnsembleObjectiveMod`, `NoiseRealizationsMod`

### `run_estimator_replay_ext.jl`
- **Type:** `script` (entry point)
- **Responsibility:** CLI driver — parse args, build the expanded set, run the staged
  optimizer against `EnsembleObjectiveMod`, then `ValidationReportMod`.
- **Depends on:** everything above

### `_tmp/estimator_seed_spread.jl`
- **Type:** `script` (analysis, throwaway)
- **Responsibility:** **The gate.** Read the five existing
  `runs_estimator_v2_replay/seed*/` results and report whether the optima cluster.
- **Depends on:** nothing (reads run artifacts)

### `_tmp/replay_profile.jl`
- **Type:** `script` (analysis, throwaway)
- **Responsibility:** Wall-clock of one replay, and of one full candidate evaluation at
  the current and proposed sizes.

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl_v2/
│   └── estimator_tuning/
│       ├── noise_realizations.jl          # NEW  hash-derived realization seeds
│       ├── replay_trajset_ext.jl          # NEW  expanded stratified manifest
│       ├── objective_ensemble_v2.jl       # NEW  multi-realization objective
│       ├── replay_parallel.jl             # NEW  trajectory-partitioned Distributed driver
│       ├── validate_realizations.jl       # NEW  held-out + clean diagnostic
│       ├── run_estimator_replay_ext.jl    # NEW  entry point
│       │
│       ├── harness_v2.jl                  # FROZEN
│       ├── objective_v2.jl                # FROZEN
│       ├── param_space_v2.jl              # FROZEN
│       ├── replay_trajset.jl              # FROZEN
│       ├── run_estimator_replay.jl        # FROZEN
│       └── warm_refine.jl                 # FROZEN
├── instructions/
│   └── estimator-v2-ensemble-replay-tuning.md      # this file
└── _tmp/
    ├── estimator_seed_spread.jl           # NEW  the §7.1 gate
    └── replay_profile.jl                  # NEW  wall-clock measurement
```

Run artifacts go to a **new** directory `hybrid_ctrl_v2/runs_estimator_v2_ensemble/`, so
`runs_estimator_v2_replay/` stays untouched as the reference result.

---

## 6. Key Interfaces

Signatures and docstrings only; bodies are stubs.

```julia
"""
    realization_seeds(seed, k) -> NamedTuple

Per-modality RNG seeds for realization `k` under tuning seed `seed`.

DERIVE BY HASH, NEVER BY ARITHMETIC OFFSET. This is the single most important
correctness requirement in this file.

SensorModV2 uses MersenneTwister throughout (sensors_v2.jl lines 50/116/196/271).
MT19937 initializes its internal state through a simple linear recurrence on the
seed, so NEARBY SEEDS PRODUCE CORRELATED STREAMS -- a documented weakness, not a
theoretical one. Two consequences if `seed+k` were used instead:

  - the R realizations would be less independent than assumed, so the ensemble
    would measure less noise diversity than it reports -- a silent underestimate
    of the objective's variance
  - with tuning seeds spaced by 1, realization sets OVERLAP outright:
    seed 42 -> {42,43,44} and seed 43 -> {43,44,45} share two of three draws,
    making the five "independent" tuning runs partially the same experiment

Use `hash((seed, modality, k))` per modality. Hash avalanche makes adjacent inputs
produce unrelated states. This COMPOSES WITH, rather than replaces, the existing
per-modality derivation already in `SensorModV2.build_suite` (sensors_v2.jl:377,
which derives seed_imu/seed_enc and notes that draw order affects reproducibility).
`Future.randjump` is the rigorous alternative (provably non-overlapping streams) but
is unnecessary given a hash costs one line.

INDEX RANGES -- disjoint by construction, so training and validation noise can never
collide and both are reproducible from the single scalar `seed`:

    k = 0            CLEAN. Noise disabled entirely. Diagnostic only -- see
                     `clean_suite` and the §9 warning about scoring it.
    k = 1, 2, 3      TRAINING. Averaged inside the objective.
    k = 101,102,103  VALIDATION. Held out; never enters any score used for
                     optimization.

Modalities must cover every RNG-bearing component of the suite: :imu, :encoder,
:flow, :pose_fix. Confirm this list against `SensorModV2.build_suite` before
implementing -- a modality that silently keeps a fixed seed across realizations
would make the ensemble narrower than reported.
"""
function realization_seeds(seed::Int, k::Int) end


"""
    build_suite_for_realization(sensor_kind, seed, k; flow, fix_tier) -> suite

Thin wrapper over the FROZEN `SensorModV2.build_suite`, supplying the realization
seeds from `realization_seeds`. Exists so no caller ever constructs a suite with a
raw integer seed and accidentally bypasses the hash derivation.

For k == 0 (clean), noise amplitudes are zeroed rather than the seed changed --
sigma_acc/sigma_gyro/sigma_omega/sigma_pos/sigma_psi to 0, bias random walks to 0,
dropout and outlier fractions to 0. Setting a seed cannot produce a noise-free
stream; the amplitudes must actually be turned off.
"""
function build_suite_for_realization(sensor_kind::Symbol, seed::Int, k::Int;
                                     flow::Bool=true, fix_tier::Symbol=:docking) end


"""
    replay_trajset_ext(run_dir, data_dir; target_n) -> Vector{NamedTuple}

Expanded stratified trajectory manifest. Returns the SAME schema as
`ReplayTrajSetMod.replay_trajset` plus a `strata` field, so it is a drop-in
substitute everywhere the 11-entry list is currently accepted.

TARGET SIZE ~25-30, up from 11. Rationale: adding trajectories reduces BIAS
(regime coverage); adding noise realizations reduces VARIANCE, and only as
1/sqrt(R). Going from 1 to 5 realizations costs 5x for a 2.2x standard-error
reduction; going from 11 to ~28 trajectories costs 2.5x and buys genuine
generalization. Trajectories are the better spend, and in replay they are cheap
because the simulation already happened.

STRATIFY, DO NOT JUST ADD. Adjacent combos of the same profile are near-duplicates
and buy almost nothing. Select to span the axes that actually stress the estimator:

    slip fraction     wheel odometry lying -- the entire reason for IMU fusion
    |omega_z|         gyro bias observability, centripetal terms
    lateral loading   V_y near the friction-circle edge
    mu                keep the existing 0.3 / 0.5 / 0.8 spread

RETAIN ALL 11 EXISTING ENTRIES unchanged as a subset, so results remain comparable
to `runs_estimator_v2_replay/`. Add ~15-19 new ones filling gaps in the strata grid.
Keep the 2 ellipse PosRef (:pose) entries and add 2-4 more pose entries: pose drift
feeds the control loop and pose-mode replay exercises a different measurement path.

STRATIFICATION SOURCE: prefer the existing analysis CSVs (`diagnostics_combined.csv`,
and whatever `roller_slip_fraction.py` emits) over recomputing anything. Confirm the
column names before relying on them; do not assume a schema.

EXISTENCE VERIFICATION -- HARD CONSTRAINT, read carefully:

    NEVER recursively search ../data/. It holds ~275 GB across ~23,000 files and a
    recursive find/glob there hangs the session. This is a standing project rule
    (CLAUDE.md), not a performance suggestion.

    Verify each candidate in O(1): construct the exact path via
    `DataStore.expected_output` from the filename contract
    (<profile>_c<combo:%03d>_mu_<mu:%g>_case<fc>_<fm>_chi_<chi:%.3f>.arrow) and
    `isfile` it. One stat call per candidate, no directory walk.

    Drop any entry that fails, log which and why. An entry that cannot be replayed
    is not a failure to work around -- it is simply not in the set.
"""
function replay_trajset_ext(run_dir::AbstractString, data_dir::AbstractString;
                            target_n::Int=28) end


"""
    make_ensemble_objective(space, trajs, seed, realizations; kwargs...) -> Function

Multi-realization replay objective. Drop-in for
`EstimatorObjectiveV2Mod.make_replay_objective_v2` -- returns a closure over
`theta` exposing a `.score` field, as `StageOptimizerMod.optimize_staged` requires.

Per candidate: for every (trajectory, realization) pair, build a suite via
`build_suite_for_realization` and replay through the FROZEN
`HarnessV2Mod.run_and_log_replay_v2`. Score each log with the FROZEN
`estimator_objective_abs_v2`, then aggregate.

COMMON RANDOM NUMBERS -- PRESERVE THIS. `realizations` and `seed` are fixed for the
lifetime of the closure, so every candidate sees the IDENTICAL noise draws and
differences between candidates are attributable to parameters alone. This property
already holds in `objective_v2.jl` ("same seed every call so noise realizations are
held fixed across candidates") and must not be lost in the generalization to R > 1.
Never draw a fresh seed inside the closure.

AGGREGATION: mean over (trajectory, realization) for the score. ALSO return the
per-realization spread -- if realization variance dominates trajectory variance,
R = 3 is too few and §10 will catch it. Reporting the spread is what makes that
diagnosable rather than invisible.

FAILURE HANDLING: preserve the existing convention exactly -- ANY replay failure for
ANY (trajectory, realization) returns the Inf-score sentinel for the whole candidate.
A candidate that cannot replay cleanly is worse than any candidate that can.

DEFAULT realizations = [1, 2, 3]. Three, not five: returns are 1/sqrt(R), and the
budget is better spent on trajectories. Three rather than two because the BERNOULLI
channels, not the Gaussian ones, set the floor -- pose-fix runs outlier_frac ~ 0.01
at 100 Hz, so a 20 s trajectory sees ~20 outlier events with realization-to-
realization spread of roughly sqrt(20)/20 ~ 22%. Discrete rare events carry more
relative variance than the white noise and hit the filter harder; two draws cannot
reveal whether they are driving the objective.
"""
function make_ensemble_objective(space, trajs::Vector,
                                 seed::Int, realizations::Vector{Int};
                                 sensor_kind::Symbol=:default, flow::Bool=true,
                                 fix_tier::Symbol=:docking) end


"""
    replay_batch(trajs, realizations, est_cfg, seed; nworkers) -> Vector

Distribute (trajectory x realization) replays across workers.

PARTITION BY TRAJECTORY, NOT BY PAIR. Each Julia worker is a separate process with
its own memory. Assign a worker one trajectory and let it run ALL realizations for
that trajectory, so the Arrow file is read once per worker instead of once per
(trajectory, realization) pair -- an R-fold reduction in I/O and resident memory.

The binding constraint on `nworkers` is the WINDOWS COMMIT LIMIT (~39.4 GB RAM +
16 GB pagefile, with a large baseline), not core count. Each worker re-imports the
module tree and holds its own Arrow buffers. Start conservative (4), measure
committed bytes, and raise only if headroom is demonstrated. An OOM here manifests
as a worker dying mid-run, which the Inf-sentinel will silently convert into a
"bad candidate" -- so DISTINGUISH worker death from replay failure in the logs, or
a memory problem will be misread as an optimization result.

Return order must be DETERMINISTIC and independent of completion order, or the
objective becomes irreproducible across runs at identical seeds.
"""
function replay_batch(trajs::Vector, realizations::Vector{Int},
                      est_cfg::NamedTuple, seed::Int; nworkers::Int=4) end


"""
    validation_report(theta_star, space, trajs, seed) -> NamedTuple

Post-tuning evaluation. Runs NOTHING that feeds back into optimization.

TWO SEPARATE REPORTS:

1. NOISE GENERALIZATION -- score theta_star on the held-out realizations
   (k = 101,102,103) and compare against the training score (k = 1,2,3). A large
   gap means the tuning fitted the specific bias walks in the training draws rather
   than the noise process. This is the ONLY check for that failure, because CRN --
   correctly -- makes it invisible during optimization.

   The risk is real and concentrated: now that sensors_v2.jl:78 correctly uses
   `gyro_bias_rw*sqrt(dt)*randn()` (v1's `sqrt(t+0.001)` made accumulated bias std
   grow LINEARLY in T instead of sqrt(T)), the bias walk is the dominant error
   source over a 20 s run and is governed by a handful of slow random variables.
   Few draws, few effective degrees of freedom, easy to fit.

2. MODEL vs NOISE ERROR -- score theta_star on the clean realization (k = 0).
   This separates process-model error from measurement-noise error: if the filter
   tracks poorly even on noise-free measurements, the fault is Q and the process
   model, not R. Report it; do not fold it into any score.

Emit both to `runs_estimator_v2_ensemble/<seed>/validation.json` alongside the
per-realization spread from training.
"""
function validation_report(theta_star::Vector{Float64}, space, trajs::Vector, seed::Int) end
```

---

## 7. Data Flow

### 7.1 The gate — run this before writing any of the above

The realization count is an **empirical question you already have the data to answer**.
`runs_estimator_v2_replay/seed1…seed5` are five independent tunings under five different
noise draws.

`_tmp/estimator_seed_spread.jl` reads them and reports, per parameter, the spread of the
optimum across the five seeds (normalized by each parameter's search-box width, since the
space is log-scaled).

- **Optima cluster tightly** → objective noise is small relative to the landscape.
  Averaging buys little. **Ship `realizations = [1]`** and put the entire budget into
  trajectories. The rest of this brief still applies; only `R` changes.
- **Optima scatter** → the objective is noise-dominated and averaging is necessary rather
  than optional. Proceed with `R = 3`, and if the spread is extreme reconsider `R = 5`
  against a reduced trajectory count.

This costs one analysis pass and decides a 3× difference in the whole tuning budget.
**Do not skip it.** Record the result in the run artifacts either way — it is the
justification for `R` in the paper.

### 7.2 Per-candidate evaluation

1. `theta` arrives from `StageOptimizerMod.optimize_staged`.
2. `apply_params_v2!(theta, space)` (**frozen**) → `est_cfg` NamedTuple, 10 ESKF parameters.
3. `replay_batch` fans out over `(trajectory × realization)`, partitioned by trajectory.
4. Each worker, per trajectory: load the Arrow file once, then for each `k` build a suite
   via `build_suite_for_realization` and call the **frozen** `run_and_log_replay_v2`.
5. Each returned `EstimatorLogV2` is scored by the **frozen**
   `estimator_objective_abs_v2`.
6. Aggregate: mean over all pairs → `.score`; also compute the per-realization and
   per-trajectory spreads.
7. Any failure anywhere → `Inf` sentinel for the candidate (existing convention).

### 7.3 Budget

```
current   : 11 trajectories × 1 realization  =  11 replays / candidate
proposed  : 28 trajectories × 3 realizations =  84 replays / candidate     (7.6×)

validation:  28 × 3 held-out                 =  84 replays   ONCE per tuning run
clean     :  28 × 1                          =  28 replays   ONCE per tuning run
```

With a 10-parameter space, CMA-ES-family population is `λ = 4 + ⌊3 ln 10⌋ = 10`, so a
generation costs ~840 replays. **Measure one replay's wall-clock first**
(`_tmp/replay_profile.jl`) — that single number sets the entire feasibility picture and
nobody has it yet.

---

## 8. Implementation Sequence

1. **`_tmp/estimator_seed_spread.jl`** — the §7.1 gate. Decides `R` before any design
   commitment. Depends on nothing.
2. **`_tmp/replay_profile.jl`** — per-replay wall-clock at the current 11-trajectory size.
   Sets the budget. Depends on nothing.
3. **`noise_realizations.jl`** — leaf. Validate independence (§10) **before** anything
   consumes it; a correlated seed scheme silently invalidates every result downstream.
4. **`replay_trajset_ext.jl`** — depends on `DataStore.expected_output`. Validate that all
   11 original entries survive verification unchanged, then that the new entries exist.
5. **`replay_parallel.jl`** — depends on 3. Test with `nworkers = 1` first and confirm it
   matches serial replay exactly, then scale up while watching committed bytes.
6. **`objective_ensemble_v2.jl`** — depends on 3, 5. **Regression gate first:** with
   `trajs = replay_trajset()` and `realizations = [1]`, it must reproduce
   `make_replay_objective_v2` exactly (§10).
7. **`validate_realizations.jl`** — depends on 6.
8. **`run_estimator_replay_ext.jl`** — depends on all. Entry point last.

---

## 9. Considerations

**The clean run must stay out of the objective.** This is the one place the intuitive
choice is wrong. With zero measurement noise the optimal `R` (measurement covariance) goes
to zero — trust the measurements completely. Averaging a clean case into the score
therefore biases `R` **downward**, which is precisely wrong for deployment. Keep it as a
held-out diagnostic where it is genuinely valuable: it is the only clean way to separate
model error from noise error.

**Never search `../data/` recursively.** ~275 GB, ~23,000 files; a recursive glob hangs.
All existence checks are `expected_output` + `isfile`, one stat per candidate. Standing
project rule.

**Distinguish worker death from replay failure.** The `Inf` sentinel makes both look like
"bad candidate." If commit-limit pressure kills a worker, the optimizer will read it as a
parameter verdict and steer away from a region for no reason. Log the two separately.

**What the five tuning seeds now measure.** If each tuning seed draws different
realizations, the five runs differ in **both** the objective and the optimizer path — the
more honest end-to-end reproducibility number, but discrepancies can no longer be
attributed to one or the other. Given CRN inside the objective, optimizer stochasticity is
the smaller effect and noise generalization is the actual question, so this is the right
trade. **Log the realization indices used per run** so the attribution stays recoverable
if the spread turns out large.

**Determinism.** Identical `(seed, realizations, trajs, nworkers)` must give a
bit-identical score. Worker count must not change results — if it does, the return
ordering or a shared RNG is leaking.

**Do not let this become a GPU project.** §3 records why it is out of scope and what would
have to change first (square-root filtering for fp32 viability). Revisit only if the
profile from step 2 shows the CPU budget is genuinely prohibitive after parallelism.

---

## 10. Success Criteria

**Preservation gates (run first — nothing below matters if these fail):**

- [ ] Every file listed in §0 is **byte-identical** to its pre-brief state.
- [ ] `run_estimator_replay.jl` (frozen entry point) runs and reproduces a previously
      recorded `runs_estimator_v2_replay/` score to machine precision.
- [ ] `make_ensemble_objective(space, replay_trajset(), seed=42, realizations=[1])`
      returns **exactly** the same score as `make_replay_objective_v2(space,
      replay_trajset(); seed=42)` on the same `theta`. This is the equivalence proof that
      the generalization to `R > 1` did not change the `R = 1` case.

**Noise-derivation checks:**

- [ ] `realization_seeds(42, 1)`, `(42, 2)`, `(42, 3)` produce streams whose pairwise
      sample correlation over ~10⁴ draws is within sampling error of zero, **per modality**.
- [ ] `realization_seeds(42, k)` and `realization_seeds(43, k)` share **no** seed value
      for any `k` or modality — the check that catches an accidental offset scheme.
- [ ] Training indices `{1,2,3}` and validation `{101,102,103}` produce disjoint seed sets.
- [ ] `k = 0` produces a genuinely noise-free stream: measured sensor output equals ground
      truth to machine precision (confirms amplitudes were zeroed, not merely re-seeded).
- [ ] Every RNG-bearing modality in `build_suite` is covered — enumerate them and assert
      the count matches.

**Trajectory-set checks:**

- [ ] All 11 original entries appear unchanged in `replay_trajset_ext` output.
- [ ] Every returned entry passes `isfile` on its `expected_output` path.
- [ ] **No recursive directory listing of `data/` occurs anywhere** — verify by inspection
      and by wall-clock (verification of ~30 entries should take milliseconds).
- [ ] The strata table is reported: min/median/max of slip fraction, `|omega_z|`, lateral
      loading, and the `mu` histogram, showing the new set spans wider than the original 11.

**Statistical-design checks:**

- [ ] CRN holds: evaluating two different `theta` twice each gives bit-identical scores
      per `theta`.
- [ ] Per-realization spread is reported alongside the score. **If realization variance
      exceeds trajectory variance, `R = 3` is too few — report it rather than absorbing it.**
- [ ] Validation (held-out realizations) score is within a stated tolerance of the training
      score. A large gap is a **finding about noise-overfitting**, to be reported, not
      tuned away.
- [ ] Clean-run score is reported separately and never enters any optimized score.

**Operational checks:**

- [ ] `nworkers = 1` and `nworkers = N` give identical scores.
- [ ] Peak committed bytes recorded at the chosen `nworkers`; headroom against the commit
      limit stated explicitly.
- [ ] Worker death and replay failure appear as distinct log events.
- [ ] Per-replay and per-candidate wall-clock recorded at both the old (11×1) and new
      (28×3) sizes.

---

## 11. Out of Scope

- **Any modification to the files in §0** — in particular the closed-loop
  `run_and_log_v2` / `run_hybrid_v2` path, which is the provable estimator+controller
  simulation capability this brief must not disturb
- **GPU acceleration** — see §3; requires a square-root ESKF rewrite first
- **Changes to the ESKF itself, the sensor models, or the 10-parameter search space** —
  this brief changes only the *statistical design of the objective*
- **Changes to the scoring function** `estimator_objective_abs_v2`
- **Closed-loop (estimator-in-the-loop) re-validation of the tuned parameters** — a
  necessary follow-up, but it runs through the frozen path and is a separate task
- **Friction diversity beyond the existing μ ∈ {0.3, 0.5, 0.8} spread**
- **Controller tuning** — verified separately as not GPU-viable and unaffected by this work
