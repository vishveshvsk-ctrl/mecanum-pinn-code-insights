# Estimator (KF / SMO) Tuning Harness — Mecanum Hybrid Observer

> **Generated:** 2026-07-22
> **Stack:** Julia 1.10+ (matches `code_insights/Project.toml` / `Manifest.toml`), OrdinaryDiffEq.jl, DiffEqCallbacks.jl, StaticArrays.jl, DataFrames/Arrow, plus the existing `hybrid_ctrl/` modules
> **Scope:** Tuning / evaluation harness (no controller redesign; no PINN/ML)
> **Downstream target:** A coding model implements `tune_estimator.jl` + supporting modules from this spec. **No code is written here.**

---

## 1. Overview

Build a **reusable block-tuning harness** whose first instantiation finds good hyperparameters for the two state estimators (`KalmanEstimator`, `SMOEstimator`) that feed the hybrid controllers. The estimator is tuned to **minimize state-estimation error against simulation ground truth** — with extra weight on **slip intervals**, where the wheel-odometry map is corrupted and the estimators differ most — over a small, **curated, frozen trajectory subset that spans both velocity-tracked (VelRef) and position-tracked (PosRef) modes**. The output is a frozen, validated estimator config per estimator, plus the subset manifest and objective definitions that **all later controller tuning reuses unchanged** (same subset, same parallel harness, same result store — only the parameter space and objective swap).

**Why PosRef is in scope:** the estimator must work under *both* tracking modes. Under VelRef the dead-reckoned pose is eval-only (drift is cosmetic); under **PosRef the pose estimate feeds back into the position controller**, so pose-estimation drift becomes load-bearing. Tuning/validating the observer only on VelRef would leave its pose behavior untested exactly where it matters.

**System-level contract:**
- **Input:** a choice of estimator (`:kalman` | `:smo`), a parameter search space, a frozen curated trajectory subset (VelRef + PosRef), a fixed nominal controller per mode, a compute budget, and a parallelism cap.
- **Output:** the best hyperparameter vector per estimator (as a loadable config), a ranked results table, per-trajectory diagnostics, and a validation-set score on a held-out trajectory — persisted as JSON/Arrow + plots.

**Why estimators first:** the estimated state is the common front-end every control law consumes; per the project's fairness contract, the estimator must be a **frozen, controlled variable** before any control-law comparison. This harness produces that frozen block and establishes the tuning methodology controllers will inherit.

---

## 2. Architecture Pattern

**Generic "tune-a-block-against-an-objective-over-a-fixed-subset" harness, with three pluggable slots (parameter space, apply-map, objective) over a shared spine (subset selector → parallel run-and-log → result store → optimizer).**

Justification: the only things that differ between estimator tuning and later controller tuning are (a) *what* is being varied, (b) *how* a parameter vector maps onto a config, and (c) *what* objective scores a run. Everything else — trajectory subset, closed-loop simulation harness, per-tick logging, parallel evaluation, result persistence, optimizer — is identical. Encoding that split makes "any controller tuning will follow this" literally true: swap the three slots, keep the spine.

---

## 3. Technology Constraints

- **Language:** Julia 1.10+ (run from `code_insights/`; activate the project env).
- **Simulation:** reuse `SchedulerMod.run_hybrid` and the `hybrid_ctrl/` stack (plant, sensors, estimators, controllers, mixer, scheduler) — do **not** reimplement the plant.
- **Numerics:** stiff solver already configured in `run_hybrid` (`FBDF`, `reltol=1e-9`, `dtmax≤1e-3`, tight bristle tolerances); the harness must not weaken these.
- **Static data:** `StaticArrays.jl` for all 3-/4-vectors (as elsewhere in the stack).
- **I/O:** JSON for configs/metrics/manifests; Arrow/DataFrames for per-tick logs; the project's plotting stack (`Plots.jl`) for diagnostics.
- **Parallelism:** `Distributed`/`pmap` or a task pool; **cap is a single machine knob** (default low — the stiff closed-loop sim is heavy; respect the project's commit-limit ceiling, do not assume >2–4 concurrent heavy runs).
- **Optimizer backend:** pluggable; the default (coarse Latin-hypercube/random → local refine) must not add a heavy dependency. A Bayesian/CMA-ES backend (e.g., `BlackBoxOptim.jl`) is optional behind the same interface.
- **Explicit exclusions:** no controller-gain search in this deliverable; no PyTorch/PINN; no hardware/real-time. (PosRef **is** in scope — see below.)

### Prerequisite dependencies (companion work, not this brief)

This harness **assumes** three upstream changes exist (track them as blockers):
1. **Estimator rewrite** (`hybrid_ctrl/estimators.jl`): fix the `KalmanEstimator` `Rn` dimension bug (3×3, not 4×4); add **accelerometer fusion** (accel-driven prediction + accel-bias states) to both estimators; add **slip-adaptive measurement weighting** (KF: inflate wheel-velocity `R` on slip detection; SMO: sliding gate + ZUPT re-anchor). Each new knob these introduce becomes a **tunable parameter** in §4's `ParamSpace`.
2. **Per-tick estimator logging** (`hybrid_ctrl/scheduler.jl`): the current `log_run` does **not** record `x̂` over time — it only logs true/ref/error and *final* bus values. The harness needs `x̂(t)` (and dead-reckoned pose `X̂o,Ŷo,ψ̂`) synchronized with true state. Add an estimator-probe log (see `run_and_log`, §4).
3. **Pose-mode control path** (for closed-loop PosRef entries): `asmc_wrench!`'s `:pose` mode is currently a stub (`error("… :pose mode not implemented")`), so a PosRef trajectory cannot yet be closed-loop pose-tracked. Options, in preference order: (a) wire a working pose-tracking nominal controller (the `run_one.jl` `asmc_torques` degree-2 position path is the reference); or (b) **interim fallback** — run the PosRef geometry *velocity-tracked* (drive its induced velocity profile) so the estimator still sees that trajectory, and defer the pose-feedback validation until (a) lands. The brief must make the chosen mode explicit per PosRef entry in the manifest.

---

## 4. Component Breakdown

### `TuningSubset` / `build_tuning_subset`
- **Type:** immutable `struct` + builder `function`
- **Responsibility:** assemble the frozen, **curated** tuning trajectory set — a small hand-picked list chosen to span the excitation modes that stress the estimator differently, covering **both VelRef and PosRef** — and serialize it to a manifest shared across all future tuning. (Not a random draw — a fixed declared list, so it is stable and defensible.)
- **Inputs:** `run_dir::AbstractString` (a `trajectory_files_run_*` dir), `entries::Vector{NamedTuple}` (the declared curated list), optional `include_optional::Bool`.
- **Outputs:** `TuningSubset` = ordered list of `(name, profile_toml, ref_type, mu, config_dir, run_mode)` entries + a content hash. `ref_type ∈ {:velref,:posref}`; `run_mode ∈ {:velocity,:pose}` (pose only where the pose-mode controller is available, else velocity per the §3.3 fallback).
- **Curated core (must-include, 5 trajectories):**
  | # | trajectory | ref_type | why it's in |
  |---|---|---|---|
  | 1 | `octagon` | VelRef | multi-directional translational start-cruise-stop legs → accel/decel transients |
  | 2 | `coupled_vomega` | VelRef | coupled translation+yaw → the Mecanum cross-coupling the estimator must track |
  | 3 | `spiral_orbit` | VelRef | sustained curved orbit, time-varying speed & curvature |
  | 4 | `ellipse` | **PosRef** | the PosRef representative — validates the observer where dead-reckoned pose feeds control |
  | 5 | `multisine_75percent_cap` @ **μ=0.3** | VelRef | broadband + high-amplitude + high-slip probe (worst-case slip) |
- **Recommended optional (flag `include_optional`):** `spin_creep` (VelRef) — the χ-gated high-yaw-rate regime, where spin→translation slip coupling is strongest. Add unless a smaller set is explicitly required.
- **μ assignment (span grip→slip):** default the geometric core (1–4) to **μ=0.5** (nominal) and the broadband probe (5) to **μ=0.3** (max slip); additionally set **`octagon` to μ=0.3** so at least one geometric trajectory stresses slip. Rationale: the slip-adaptive gate/R must see *both* gripping and slipping conditions to tune its threshold. μ is a documented knob in the manifest — widening to a μ grid is a later coverage step, not a re-pick.
- **Depends on:** `Profiles` (registry/`ref_type` introspection: `is_velref`/builder family), the `trajectory_files_run_*/profiles/` listing.
- **Contract:** written once to `runs_estimator/subset_manifest.json`; **frozen**; controller tuning loads the *same* manifest. The manifest records each entry's `ref_type` and `run_mode` so downstream tuning inherits the identical mode assignment.

### `ParamSpace` / `kf_param_space` / `smo_param_space`
- **Type:** `struct` describing named search dimensions (bounds, scale linear/log, whether diagonal-vector) + two constructors.
- **Responsibility:** declare the tunable hyperparameters and their bounds for each estimator.
- **KF dimensions (after the accel-fusion rewrite):** `Qn` (3 diag, velocity process noise), `Rn` (3 diag, measurement noise — the 3×3 fix), accel-bias process noise (2), initial `P` scale, **slip-adaptive** params (wheel-`R` inflation factor, slip-detection threshold), `rate_hz` (fixed or coarse).
- **SMO dimensions:** `L` (3, sliding gain), `K` (3, disturbance/integral gain), `δ` (boundary layer), **slip-gate** threshold(s), **ZUPT** re-anchor threshold(s), accel-bias params.
- **Outputs:** a `ParamSpace` an optimizer can sample and `apply_params!` can decode.
- **Depends on:** the estimator structs' exposed fields (prerequisite rewrite).

### `apply_params!`
- **Type:** `function` (block-specific slot #2)
- **Responsibility:** decode a raw parameter vector from the optimizer into a concrete estimator config object (`KalmanEstimator` or `SMOEstimator` kwargs), enforcing bounds/positivity.
- **Inputs:** `theta::Vector{Float64}`, `space::ParamSpace`.
- **Outputs:** an estimator config ready to hand to `run_hybrid`.

### `run_and_log`  (shared spine)
- **Type:** `function`
- **Responsibility:** run one closed-loop simulation for a `(estimator_config, trajectory)` pair with a **fixed nominal controller**, capturing the synchronized per-tick true state, estimated state, and slip indicator. Handles **both tracking modes**: the entry's `run_mode` selects `cfg.tracking = :velocity | :pose` and the corresponding nominal controller (per §3.3, `:pose` requires the pose-mode control path; else the entry falls back to `:velocity`).
- **Inputs:** `est_cfg`, `traj::NamedTuple` (one subset entry, carrying `ref_type`/`run_mode`), `nominal_ctrl::HybridConfig` (fixed per mode), `seed::Int`.
- **Outputs:** an `EstimatorLog` (below).
- **Required harness change:** install an estimator-probe callback (or extend the estimator callback) that, at each estimator tick, records `t`, true velocity `[Vx,Vy,ψ̇] = u[1:3]`, **true pose `[Xo,Yo,ψ] = u[17,18,4]`**, `bus.xhat[1:6]` (velocity **and** dead-reckoned pose), `bus.d_hat`, and the slip indicator. True state comes from `integrator.u`; estimate from `bus.xhat`.
- **Depends on:** `SchedulerMod.run_hybrid`, `slip_indicator`.

### `EstimatorLog`
- **Type:** `struct` (per-run record)
- **Responsibility:** hold aligned time series `t[:]`, `v_true[:,3]`, `v_hat[:,3]`, `pose_true[:,3]`, `pose_hat[:,3]`, `d_hat[:,3]`, `slip[:]`, plus the trajectory id, `ref_type`, `run_mode`, and seed.

### `slip_indicator`
- **Type:** `function` (ground-truth slip signal, evaluation-only)
- **Responsibility:** quantify, from true state, how badly the no-slip wheel map disagrees with true body velocity — this is exactly the corruption the estimator must survive, and the weighting signal for the objective.
- **Inputs:** true wheel speeds `ω_true = u[9:12]`, true body velocity `[Vx,Vy,ψ̇] = u[1:3]`, `params` (wheel Jacobian).
- **Outputs:** scalar slip magnitude per timestep `‖ Hω\ω_true − [Vx,Vy,ψ̇]_true ‖`.
- **Note:** ground-truth only; **never** exposed to the estimator. Also usable later to validate that the SMO `d̂` tracks real slip.

### `estimator_objective`  (block-specific slot #3)
- **Type:** `function`
- **Responsibility:** reduce one or more `EstimatorLog`s to a scalar (or small vector) score to minimize — **mode-aware**, so PosRef entries additionally penalize pose-estimation error.
- **Inputs:** `logs::Vector{EstimatorLog}`, weights `(λ_slip, λ_smooth, λ_pose)`.
- **Outputs:** `NamedTuple` of `overall_nrmse`, `inslip_nrmse`, `pose_drift`, `smoothness`, and a combined `score`.
- **Definition (prose):**
  - **Velocity term (all entries):** per trajectory, per channel, NRMSE = RMS(`v_hat − v_true`) / RMS(`v_true`); aggregate channels, then **average across trajectories** (so one long run doesn't dominate). Re-weight samples by `1 + λ_slip · normalized_slip(t)` so slip intervals dominate.
  - **Pose term (PosRef entries only):** the dead-reckoned pose `(X̂o,Ŷo,ψ̂)` feeds the position controller, so score its error vs true pose — as a **drift metric** (error growth rate over the run, or terminal drift normalized by path length), scaled by `λ_pose`. For VelRef entries pose is eval-only → `λ_pose=0`, pose reported but not scored.
  - **Smoothness penalty (all entries):** on `d/dt(v_hat)` high-frequency content (chatter injected into controllers is harmful), scaled by `λ_smooth`.
  - Report all sub-metrics; combine into one `score` for the optimizer. Keep `inslip_nrmse` and `pose_drift` as first-class reported metrics, not just blended.

### `Optimizer`  (shared spine, pluggable backend)
- **Type:** abstract interface + at least a `CoarseThenLocal` default (Latin-hypercube/random coarse pass → local refine), with an optional `BlackBox` backend behind the same `ask`/`tell` interface.
- **Responsibility:** propose parameter vectors and consume scores under a fixed evaluation budget.
- **Inputs:** `space::ParamSpace`, `budget::Int`, `parallelism::Int`.
- **Outputs:** ranked `(theta, score)` history + best.

### `parallel_evaluate`  (shared spine)
- **Type:** `function`
- **Responsibility:** evaluate a batch of proposed parameter vectors concurrently — each vector runs the whole subset — respecting the machine parallelism cap.
- **Inputs:** `thetas::Vector{Vector{Float64}}`, `subset::TuningSubset`, `space`, `objective`, `max_parallel::Int`.
- **Outputs:** `Vector` of per-θ objective results (failures → sentinel, logged, not fatal).

### `ResultStore`  (shared spine)
- **Type:** `function`s
- **Responsibility:** persist the subset manifest, per-θ metrics, the ranked table, the best config (loadable by `run_hybrid`), and diagnostic plots, under a run directory.
- **Outputs:** `runs_estimator/<est>/{subset_manifest.json, trials.arrow, best_config.json, diagnostics/*.png}`.

### `tune_estimator.jl`  (entry point)
- **Type:** CLI `script`
- **Responsibility:** parse args, build/load the subset, select estimator + param space + objective, run the optimizer via the parallel harness, persist results, and run a **held-out validation** on a trajectory *not* in the tuning subset.
- **CLI args:** `--estimator {kalman|smo|both}`, `--run-dir`, `--seed`, `--budget`, `--max-parallel`, `--out`, `--subset-manifest` (optional; else generate + freeze).

---

## 5. File & Directory Structure

```
code_insights/
├── tune_estimator.jl                 # CLI entry (§4 entry point)
├── hybrid_ctrl/
│   ├── estimators.jl                 # PREREQUISITE: accel fusion + Rn fix + adaptive/gate params
│   └── scheduler.jl                  # PREREQUISITE: add per-tick x̂/d̂/slip probe logging
├── tuning/                           # NEW — the reusable harness
│   ├── subset.jl                     # TuningSubset + build_tuning_subset (§4)
│   ├── param_space.jl                # ParamSpace + kf/smo spaces + apply_params! (§4)
│   ├── harness.jl                    # run_and_log, EstimatorLog, slip_indicator (§4, shared spine)
│   ├── objectives.jl                 # estimator_objective (block-specific slot)
│   ├── optimizer.jl                  # Optimizer interface + CoarseThenLocal (+ optional backend)
│   ├── executor.jl                   # parallel_evaluate (§4)
│   └── results.jl                    # ResultStore (§4)
└── runs_estimator/                   # OUTPUT
    ├── subset_manifest.json          # frozen; shared with future controller tuning
    ├── kalman/{trials.arrow, best_config.json, diagnostics/*.png}
    └── smo/{trials.arrow, best_config.json, diagnostics/*.png}
```

The `tuning/` spine (`subset.jl`, `harness.jl`, `optimizer.jl`, `executor.jl`, `results.jl`) is **estimator-agnostic**; a future `tune_controller.jl` reuses it verbatim, supplying only a controller `param_space` + `apply_params!` + `objective`.

---

## 6. Key Interfaces

Signatures + docstrings only. Bodies are always `# ...`.

```julia
"""
    build_tuning_subset(run_dir, entries; include_optional=false) -> TuningSubset

Assemble the frozen CURATED tuning set from a fixed declared list spanning
translational (octagon), coupled V–ω (coupled_vomega), curved-orbit VelRef
(spiral_orbit), curved-orbit PosRef (ellipse), and broadband high-slip
(multisine_75 @ μ=0.3); optional spin_creep for the χ-gated spin regime.
Each entry carries ref_type (:velref|:posref) and run_mode (:velocity|:pose).
Serializes a manifest with a content hash. Not random — stable and defensible.
  run_dir          :: AbstractString        a trajectory_files_run_* directory
  entries          :: Vector{NamedTuple}    the curated list (name, mu, ref_type, run_mode)
  include_optional :: Bool                  add spin_creep
Returns TuningSubset (ordered entries + hash).
"""
function build_tuning_subset(run_dir, entries; include_optional=false)
    # ...
end

"""
    kf_param_space() -> ParamSpace   /   smo_param_space() -> ParamSpace

Declare tunable hyperparameters + bounds for each estimator (post accel-fusion
rewrite). KF: Qn(3), Rn(3), accel-bias noise(2), P0 scale, slip-R-inflation,
slip-threshold. SMO: L(3), K(3), δ, slip-gate, ZUPT-threshold, accel-bias.
"""
function kf_param_space() # ... end
function smo_param_space() # ... end

"""
    apply_params!(theta, space) -> est_cfg

Decode a bounded parameter vector into a concrete estimator config
(KalmanEstimator|SMOEstimator kwargs), enforcing positivity/bounds.
"""
function apply_params!(theta, space)
    # ...
end

"""
    slip_indicator(u, params) -> Float64

Ground-truth slip magnitude ‖ Hω\\ω_true − [Vx,Vy,ψ̇]_true ‖ at one timestep
(the no-slip-map corruption the estimator must survive). Evaluation-only;
never fed to the estimator.
"""
function slip_indicator(u, params)
    # ...
end

"""
    run_and_log(est_cfg, traj, nominal_ctrl; seed) -> EstimatorLog

Run one closed-loop sim (fixed nominal controller) via SchedulerMod.run_hybrid
with per-tick probe logging. Returns aligned t / v_true / v_hat / d_hat / slip.
"""
function run_and_log(est_cfg, traj, nominal_ctrl; seed)
    # ...
end

"""
    estimator_objective(logs; λ_slip, λ_smooth, λ_pose) -> NamedTuple

Reduce per-trajectory logs to (overall_nrmse, inslip_nrmse, pose_drift,
smoothness, score). Mode-aware: velocity NRMSE (slip-weighted, averaged across
trajectories) for all entries; a pose-drift penalty (λ_pose) for PosRef entries
where dead-reckoned pose feeds control; a d/dt(v_hat) chatter penalty (λ_smooth).
`score` is the scalar the optimizer minimizes.
"""
function estimator_objective(logs; λ_slip, λ_smooth, λ_pose)
    # ...
end

"""
    parallel_evaluate(thetas, subset, space, objective; max_parallel) -> Vector

Evaluate proposed parameter vectors concurrently (each runs the whole subset),
respecting the machine parallelism cap. Per-θ failures → logged sentinel.
"""
function parallel_evaluate(thetas, subset, space, objective; max_parallel)
    # ...
end
```

---

## 7. Data Flow

Primary tuning pass (one optimizer iteration):

1. `build_tuning_subset` loads the frozen curated manifest (5 core VelRef+PosRef trajectories, + optional `spin_creep`); each entry carries `ref_type`/`run_mode`.
2. `Optimizer.ask` proposes a batch of parameter vectors `θ` from the estimator `ParamSpace`.
3. For each `θ`: `apply_params!` → `est_cfg`. `parallel_evaluate` fans the batch across workers (cap-limited).
4. For each `(θ, trajectory)`: `run_and_log` runs `run_hybrid` with the **fixed nominal controller for that entry's `run_mode`** (`:velocity` or `:pose`) and the candidate estimator; the probe callback records `t, v_true(=u[1:3]), v_hat(=bus.xhat[1:3]), pose_true(=u[17,18,4]), pose_hat(=bus.xhat[4:6]), d_hat, slip_indicator(u,params)` at each estimator tick → `EstimatorLog`.
5. `estimator_objective` reduces the trajectory logs to sub-metrics + a scalar `score` (slip-weighted velocity NRMSE + PosRef pose-drift + chatter penalty).
6. `Optimizer.tell` consumes `(θ, score)`; loop until budget exhausted.
7. `ResultStore` writes trials, the best config (loadable by `HybridConfig`/`run_hybrid`), and diagnostics.
8. **Held-out validation:** re-score the best config on a trajectory *outside* the subset; report generalization.

**Evaluation/optimization chain:** closed-loop stiff ODE (plant) with discrete estimator/controller callbacks → per-tick state-error time series → slip-weighted NRMSE + chatter penalty → scalar objective → derivative-free optimizer. No gradients.

**Controlled-variable separation (enforce):** the estimator is tuned with a **single fixed nominal controller** and fixed sensor seed(s); the true state is used **only** by `slip_indicator` and `estimator_objective`, **never** by the estimator. Document the mild closed-loop coupling (estimator tuned under controller A, later reused under B/C — legitimate because the *same frozen* estimator is shared).

---

## 8. Implementation Sequence

1. **Prerequisites (blockers):** estimator rewrite (accel fusion + `Rn` fix + adaptive/gate params) and scheduler per-tick probe logging. Nothing below runs correctly without these.
2. **`slip_indicator` + `EstimatorLog`** — the ground-truth signal and the record type everything reduces over.
3. **`run_and_log`** — the closed-loop harness; validate it reproduces a known `run_hybrid` trajectory and that `v_hat` is captured and time-aligned with `v_true`.
4. **`build_tuning_subset`** — the frozen curated list; verify determinism (same list → same manifest hash), that the PosRef `ellipse` is **included** with its `run_mode`, and that the high-slip `multisine_75 @ μ=0.3` is present.
5. **`ParamSpace` + `apply_params!`** — the search dimensions and decode; unit-test round-trip decode within bounds.
6. **`estimator_objective`** — the scalar score; sanity-check that a deliberately detuned estimator scores worse and that in-slip weighting shifts the ranking.
7. **`Optimizer` + `parallel_evaluate`** — the search loop and concurrent evaluation; test at `max_parallel=1` first, then scale to the cap.
8. **`ResultStore` + `tune_estimator.jl`** — persistence, CLI, held-out validation.
9. **Run for `:kalman`, then `:smo`;** freeze both best configs for downstream controller tuning.

---

## 9. Numerical / Solver / Reproducibility Considerations

- **Ground truth is trusted, never fed:** true `[Vx,Vy,ψ̇]` from `sol.u`/`integrator.u` is objective-only. Any leakage into the estimator invalidates the whole exercise — assert the estimator's inputs are strictly `y_meas` + bus.
- **Time alignment:** the estimator runs at `f_est` (1 kHz) while the solver saves at `saveat_hz`. Log `v_hat` **at the estimator tick** (where `bus.xhat` updates), and sample `v_true` at the same instants — do not compare mismatched grids or interpolate silently.
- **Slip weighting is the point:** estimators are nearly identical in-grip; they diverge in-slip. Report `inslip_nrmse` separately and make it a first-class ranking metric, not just a blended scalar — otherwise the tuner optimizes the easy regime.
- **PosRef pose drift is load-bearing:** under `:pose` mode the dead-reckoned `(X̂o,Ŷo,ψ̂)` closes the loop, so its drift affects control (unlike VelRef, where pose is eval-only). Score pose as a **drift rate** (or path-length-normalized terminal drift), not raw RMS — pose is unobservable from IMU+encoder, so absolute drift is expected and only its *rate* is a fair, tunable target. Keep the velocity term dominant; pose is a secondary, mode-gated objective.
- **Mode fallback must be explicit:** if the pose-mode controller (§3.3) isn't wired yet, PosRef entries run velocity-tracked and `λ_pose` is inert for them — the manifest's `run_mode` records which, so results are never silently mislabeled as pose-validated.
- **Chatter penalty:** the SMO's switching term can inject high-frequency content into `x̂`; since `x̂` feeds controllers, penalize `d/dt(v_hat)` energy so a "low-NRMSE-but-chattery" estimator doesn't win.
- **Seed handling:** tune across a *small set* of sensor seeds (not one) to avoid overfitting a noise realization; log seeds; identical seeds ⇒ identical results. The subset manifest, seeds, and best config together must reproduce every number.
- **Solver settings are fixed:** do not let the search alter `reltol`/`dtmax`/bristle tolerances — those are physics-fidelity, not estimator hyperparameters.
- **Parallel safety:** each worker owns its own `ControllerBus`/estimator state; no shared mutable estimator across concurrent runs. Respect the machine's heavy-run cap (default low); a failed/infeasible run returns a logged sentinel and does not abort the batch.
- **Reusability guard:** keep `subset.jl`, `harness.jl`, `optimizer.jl`, `executor.jl`, `results.jl` free of any estimator-specific logic so `tune_controller.jl` can import them unchanged; estimator-specific code lives only in `param_space.jl`, `apply_params!`, and `objectives.jl`.

---

## 10. Success Criteria

- [ ] `build_tuning_subset` is deterministic (curated list → identical manifest hash), includes the PosRef `ellipse`, the named VelRef cores (`octagon`, `coupled_vomega`, `spiral_orbit`), and the high-slip `multisine_75 @ μ=0.3`; each entry's `ref_type`/`run_mode` is recorded.
- [ ] PosRef entries either close the loop in `:pose` mode (if the pose controller is wired) or run the documented velocity-tracked fallback — never silently mislabeled; `pose_drift` is reported for PosRef.
- [ ] `run_and_log` reproduces a reference `run_hybrid` trajectory and returns time-aligned `v_true`/`v_hat`.
- [ ] Ground-truth state is provably absent from the estimator's inputs (leakage assertion passes).
- [ ] The tuner runs end-to-end for `:kalman` and `:smo`, producing a frozen best config loadable by `HybridConfig`/`run_hybrid`.
- [ ] Best config beats the current defaults on both overall and **in-slip** NRMSE, and holds up on the **held-out** validation trajectory.
- [ ] Accel-fusion estimator materially lowers **in-slip** NRMSE versus a no-accel baseline (the design's whole premise).
- [ ] Deterministic reproduction from manifest + seeds; all sub-metrics logged.
- [ ] Harness spine is estimator-agnostic (demonstrated by a stub controller param-space/objective plugging in without touching the spine).

## 11. Out of Scope

- The estimator **structural rewrite** itself (accel fusion, `Rn` fix, gates) — prerequisite/companion brief.
- The **pose-mode control path** (`asmc_wrench!` `:pose`, or wiring `run_one.jl`'s position controller) — a §3.3 prerequisite consumed here, not built here.
- **Controller** gain tuning and the control-law comparison — later instantiation reusing this harness.
- μ/χ **sweep design** beyond the curated subset's coverage (broader robustness grids are a later evaluation concern).
- Hardware / real-time / PINN / GPU.
