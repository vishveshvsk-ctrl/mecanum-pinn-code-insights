# Controller Comparison on a Frozen ESKF Estimator

> **Generated:** 2026-07-24
> **Stack:** Julia (project at `code_insights/Project.toml`); reuses `hybrid_ctrl/` modules + `tune_controller.jl` helpers
> **Scope:** Evaluation / comparison harness (no tuning, no new controllers, no new estimator)

---

## 1. Overview

Compare the three finalized controllers (ASMC, PID, and — optionally — MPC) **closed-loop on a single frozen ESKF state estimator**, on a trajectory subset that **excludes ellipse**, run in **two variants** (with and without `coupled_vomega`). This is the culmination of the estimator-first methodology: the controllers were tuned against an `OracleEstimator` (true state + noise); here they are re-evaluated when fed a **real, imperfect** ESKF estimate, so the comparison reflects deployable performance.

**System-level contract.** Input: a frozen ESKF `best_config.json`, three controller `best_config.json` files, a trajectory-config dir, and a seed set. Output: per-run tidy CSV (per controller × trajectory × seed) + mean±std summary tables, produced **twice** — once for subset variant A (with `coupled_vomega`), once for variant B (without) — reporting true-state tracking error, control effort, chatter, and (as context) ESKF estimation error.

**Key architectural fact (the gap this fills):** `compare_estimators.jl` varies the *estimator* with a hard-coded default ASMC; `tune_controller.jl::run_controller` hard-wires `estimator=:oracle`. Neither can vary the *controller* against a *frozen real estimator*. This brief specifies a thin new driver that composes the two halves.

---

## 2. Architecture Pattern

**Thin evaluation driver over the existing closed-loop harness.** Reuse `load_frozen_estimator` + `_build_estimator` (frozen ESKF), `build_controller` + the `*_override` kwargs (varying controller), `run_hybrid` (closed loop), and `controller_metrics` (scoring). The only new code is the driver loop + a run helper that injects a **frozen estimator instead of the oracle**, plus subset selection and output. No physics, no controller, no estimator logic is rewritten.

Justification: every piece already exists and is validated; the task is composition. Rewriting any of it risks divergence from the tuned/frozen artifacts.

---

## 3. Technology Constraints

- **Language:** Julia. Run single-threaded (`julia -t 1`), `--project=. --startup-file=no`, with `LinearAlgebra.BLAS.set_num_threads(1)` at top (determinism + this machine OOMs on parallelism).
- **Reused modules (via `include` after `run_one.jl`):** `hybrid_ctrl/{config,bus,plant,sensors,estimators,controllers,fuzzy,mixer,scheduler}.jl`, plus `tuning/harness.jl` (for `_build_estimator`) and `tune_controller.jl` (main-guarded — `include` does NOT run `main()`; reuse `build_controller`, `controller_metrics`, `default_trajs_3`).
- **Determinism:** trajectory refs must be built with pinned `combo_idx` + fixed RNG (`Random.Xoshiro(0)`), exactly as `tune_controller.jl::run_controller` does — the global-RNG `pick_and_build` path is non-deterministic and must be avoided.
- **Frozen ESKF provenance caveat (from handoff):** the config at `runs_eskf_noellipse_v2/eskf_dxnes/best_config.json` was **hand-reconstructed** from `checkpoint_best.json` (the `tune_estimator.jl` run crashed in `save_trials` before `save_best_config`; SENTINEL bug, fixed for future runs). It is numerically equivalent and schema-correct, but `trials.arrow`, `diagnostics/`, `validation.json` are absent in that dir — **not needed here** (comparison only reads `best_config.json`).
- **Explicit exclusions:** do NOT retune anything; do NOT modify the frozen ESKF or the controller configs; do NOT use `OracleEstimator` for the main comparison (it is the baseline, not the test).

---

## 4. Component Breakdown

### `compare_controllers.jl` (new — the driver/entry point)
- **Type:** script with a main-guard (`if abspath(PROGRAM_FILE) == @__FILE__`).
- **Responsibility:** parse args, load the frozen ESKF config + the controller configs, build the two trajectory-subset variants, run the controller × trajectory × seed loop through the frozen ESKF, score, and write CSV + summary tables per variant.
- **Depends on:** all components below.

### `load_frozen_estimator(est_dir::String)` (REUSE from `compare_estimators.jl`)
- **Responsibility:** read `joinpath(est_dir, "best_config.json")`, return the estimator config **NamedTuple** (the `:eskf` branch fills all ESKF fields + `estimator=:eskf`, `use_dhat=false`).
- **Inputs:** `est_dir` (pass the parent `eskf_dxnes/` dir, NOT `runs_eskf_noellipse_v2/`).
- **Outputs:** config NamedTuple.
- **Note:** it does NOT build the struct — that is `_build_estimator`.

### `_build_estimator(est_cfg)` (REUSE from `tuning/harness.jl`)
- **Responsibility:** turn the config NamedTuple into a **fresh** `ESKFEstimator` struct (mutable, stateful: `x`, `P`, `initialized`).
- **Contract:** MUST be called **once per run** (per traj×seed) so estimator state never leaks between runs.

### `load_controller_kw(path::String, ctrl::Symbol)` (new tiny helper, or reuse the `experiment_*.jl` one-liner)
- **Responsibility:** read a controller `best_config.json`, return the `best_gains` as a kwargs NamedTuple (SVector-reconstructing list fields), consumable by `build_controller(ctrl, kw)`.
- **Inputs:** config path, controller symbol (`:asmc|:pid|:mpc`).
- **Outputs:** NamedTuple. (ASMC config already folds pinned K_max into `best_gains`, so no injection needed.)

### `run_controller_on_estimator(ctrl, kw, est_cfg, tr; seed)` (new — the core run helper)
- **Type:** function. **This is the one genuinely new run path.**
- **Responsibility:** mirror `tune_controller.jl::run_controller` BUT feed a **freshly-built frozen estimator** instead of the oracle. Build `HybridConfig` with `estimator=:eskf` (or the frozen kind), `use_pose_fix` per the estimator (ESKF exposes pose), the correct `fixed_weights`/`use_*` flags for `ctrl`; build the controller via `build_controller`; build the ref with pinned `combo_idx` + `Xoshiro(0)`; call `run_hybrid(cfgh, params, name; est=_build_estimator(est_cfg), ref=ref, asmc_override=/mpc_override=/pid_override=, chi=…, config_dir=…, profile_toml=…, return_bus=true, sensor_seed=seed)`.
- **Inputs:** `ctrl::Symbol`, `kw::NamedTuple`, `est_cfg` (frozen NamedTuple), `tr` (subset entry), `seed::Int`.
- **Outputs:** `(probe, ref, mode)` — `probe = ESTIMATOR_PROBE_LOG[objectid(bus)]` per-tick buffer holding both **true plant state** (`u[...]`) and **estimated** `xhat`.
- **Depends on:** `_build_estimator`, `build_controller`, `run_hybrid`, `Profiles.resolve_profile`/`build`.

### `controller_metrics(probe, ref, mode)` (REUSE from `tune_controller.jl`)
- **Responsibility:** score a run. Returns `(tracking, ce, chatter, ok, abs)`. `tracking` uses the **true** plant state (`probe[i].u[...]`) vs `ref` → measures real tracking given the imperfect estimate fed to the controller. `abs` = physical-unit sub-metrics.

### `estimator_error(probe)` (new — secondary/context metric)
- **Responsibility:** compute ESKF estimation error from the probe: NRMSE of `xhat` vs the true state (`u`) per channel (Vx, Vy, ψ̇, ψ, X, Y). Context only — shows how badly the frozen ESKF estimates under each controller's closed-loop trajectory (estimator quality is closed-loop-coupled, so it varies by controller).
- **Inputs:** `probe`. **Outputs:** NamedTuple of per-channel NRMSE.

### `build_subset_variants(run_dir)` (new)
- **Responsibility:** produce the two ellipse-excluded subsets from `default_trajs_3`'s entries.
  - **Variant A ("with coupled"):** `octagon(206)`, `spin_creep(178)`, `coupled_vomega(12)`, `spiral_orbit(37)` (all velref/velocity).
  - **Variant B ("no coupled"):** `octagon(206)`, `spin_creep(178)`, `spiral_orbit(37)`.
- **Rationale for the two variants:** `coupled_vomega` is the one trajectory that couples translation *and* yaw; isolating its presence tests whether controller ranking is sensitive to the coupled-dynamics case under a real estimator.

### `ResultWriter` (new — output)
- **Responsibility:** write, per variant, `<out>/<variant>/runs.csv` (tidy per-run rows: controller, trajectory, mode, seed, tracking, ce, chatter, est_nrmse_* , abs_*) and `<out>/<variant>/summary.csv` (mean±std grouped by `[controller, trajectory]`). Long-format CSV for figures (this session builds the figures).

---

## 5. File & Directory Structure

```
code_insights/
├── compare_controllers.jl                 # NEW — driver/entry point (this brief)
├── tune_controller.jl                     # REUSE — build_controller, controller_metrics, default_trajs_3 (main-guarded)
├── compare_estimators.jl                  # REUSE — load_frozen_estimator
├── tuning/harness.jl                      # REUSE — _build_estimator
├── run_one.jl + hybrid_ctrl/*.jl          # REUSE — plant, scheduler, estimators, controllers
├── runs_eskf_noellipse_v2/eskf_dxnes/
│   └── best_config.json                   # INPUT — frozen ESKF (pass this parent dir)
├── runs_controller_asmc_pin/asmc_FINAL_seed3.json    # INPUT — ASMC
├── runs_controller_pid_5seed/pid_FINAL_seed2.json    # INPUT — PID
├── runs_controller/mpc_clean/best_config.json        # INPUT — MPC (NOT finalized; see §11)
└── runs_controller_compare_eskf/          # OUTPUT
    ├── with_coupled/{runs.csv, summary.csv}
    └── no_coupled/{runs.csv, summary.csv}
```

---

## 6. Key Interfaces

Signatures + docstrings only; bodies are stubs for the downstream model to fill.

```julia
"""
    run_controller_on_estimator(ctrl, kw, est_cfg, tr; seed=42) -> (probe, ref, mode)

Closed-loop run of controller `ctrl` (gains `kw`) fed by a FRESH frozen estimator
built from `est_cfg`, on subset entry `tr`. Mirrors tune_controller.jl::run_controller
but injects `est=_build_estimator(est_cfg)` instead of an OracleEstimator, and sets
`estimator=:eskf` in the HybridConfig. Ref built with pinned combo_idx + Xoshiro(0).
Returns the per-tick probe (true `u` + estimated `xhat`), the ref, and the run mode.
"""
function run_controller_on_estimator(ctrl::Symbol, kw::NamedTuple, est_cfg,
                                     tr; seed::Int=42)
    error("stub")
end

"""
    estimator_error(probe) -> NamedTuple

Per-channel NRMSE of xhat vs true plant state over the run:
(vx, vy, psidot, psi, X, Y). Context metric — ESKF quality is closed-loop-coupled.
"""
function estimator_error(probe)
    error("stub")
end

"""
    build_subset_variants(run_dir) -> (with_coupled::Vector, no_coupled::Vector)

Two ellipse-excluded trajectory subsets from default_trajs_3's velref entries.
with_coupled = [octagon, spin_creep, coupled_vomega, spiral_orbit];
no_coupled   = [octagon, spin_creep, spiral_orbit].
"""
function build_subset_variants(run_dir::String)
    error("stub")
end
```

CLI (mirror `compare_estimators.jl` arg style):
```
julia -t 1 --project=. --startup-file=no compare_controllers.jl \
  --estimator-dir runs_eskf_noellipse_v2/eskf_dxnes \
  --controllers asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,pid:runs_controller_pid_5seed/pid_FINAL_seed2.json,mpc:runs_controller/mpc_clean/best_config.json \
  --run-dir trajectory_files_run_0p5_main \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --out runs_controller_compare_eskf
```

---

## 7. Data Flow

1. Load frozen ESKF config NamedTuple via `load_frozen_estimator(estimator_dir)` (once).
2. Load each controller's `kw` via `load_controller_kw` (once each).
3. Build the two subset variants via `build_subset_variants(run_dir)`.
4. For each **variant** → for each **controller** → for each **trajectory** `tr` → for each **seed**:
   a. `run_controller_on_estimator(ctrl, kw, est_cfg, tr; seed)` — builds a FRESH ESKF (`_build_estimator`), runs closed loop; the scheduler's sensor callback routes to `estimator_update!(bus, y, eskf, params, dt)` (NOT `oracle_feed!`), writing `bus.xhat=[Vx,Vy,ψ̇,ψ,X,Y]` that the controller consumes.
   b. `m = controller_metrics(probe, ref, mode)` → true-state tracking, ce, chatter.
   c. `e = estimator_error(probe)` → ESKF NRMSE (context).
   d. Append a tidy row.
5. Per variant: write `runs.csv` (all rows) + `summary.csv` (mean±std by [controller, trajectory]).

**Baseline cross-reference (no new computation):** the oracle-clean tracking numbers for ASMC/PID already exist (`runs_controller/RESULTS_controller_tuning.md`, `noise_eval_10seed.csv` scale=0). The frozen-ESKF tracking should be compared against those to quantify the oracle→real-estimator gap per controller.

---

## 8. Implementation Sequence

1. **`load_controller_kw` + config-load smoke** — load all three configs, `build_controller` each, confirm no error. (Unblocks everything.)
2. **`run_controller_on_estimator`** — the core; verify one ASMC run on `octagon` through the frozen ESKF returns a non-empty probe and finite `controller_metrics`. Confirm the sensor callback took the `estimator_update!` branch (not oracle) — e.g. assert `bus.xhat` differs from true `u` (real estimator has error).
3. **`estimator_error`** — compute from the same probe; sanity-check NRMSE is finite and > 0.
4. **`build_subset_variants`** — assert the two entry lists (4 vs 3, ellipse absent, coupled present/absent).
5. **Driver loop + `ResultWriter`** — full nested loop, both variants, write CSVs.
6. **Run** the 10-seed job (single-threaded, background) and eyeball `summary.csv`.

---

## 9. Julia / simulation-specific considerations

- **Fresh estimator per run:** `ESKFEstimator` is mutable and accumulates `x`/`P`/`initialized`. Build a NEW one inside the loop (via `_build_estimator(est_cfg)`) for every (traj, seed) — never reuse an instance across runs.
- **Determinism:** BLAS single-thread; build refs with pinned `combo_idx` + `Random.Xoshiro(0)`; the only intended randomness is the **sensor-noise seed** (`sensor_seed=seed` in `run_hybrid`), which the seed set varies. With the ESKF frozen and refs deterministic, seed variation isolates sensor-noise realization.
- **World-age:** if `_build_estimator`/`load_frozen_estimator` are pulled from modules loaded via `Base.require`, wrap late calls in `Base.invokelatest` (follow the pattern already in `compare_estimators.jl`).
- **Metric semantics:** `controller_metrics.tracking` is computed on the **true** plant state (`probe.u`) vs ref — this is deliberate (it measures real tracking under an imperfect estimate). `estimator_error` (xhat vs true) is separate context. Do NOT confuse the two.
- **Chatter caveat (carry forward):** `chatter` = TV of the slew-limited `v_cmd`; it under-discriminates under noise (the mixer slew limiter masks controller chatter). On the frozen ESKF (which injects estimate error, akin to noise) expect chatter to look similar across controllers; the discriminator is **tracking** and **est_error**. (A true controller-chatter metric would need the pre-slew wrench — out of scope unless requested.)
- **Pose vs velocity:** all subset entries are **velref/velocity** (ellipse, the only posref, is excluded), so `controller_metrics` takes its `:velocity` branch and the ESKF's pose channels (X,Y,ψ) are context, not tracked targets. Confirm `run_mode=:velocity` for every entry.

---

## 10. Success Criteria

- [ ] One-run smoke: ASMC on octagon through the frozen ESKF yields finite `tracking/ce/chatter` and a non-trivial `estimator_error` (ESKF ≠ oracle).
- [ ] Sensor callback confirmed on the `estimator_update!` (ESKF) branch, not `oracle_feed!`.
- [ ] Both subset variants built correctly (4-traj with coupled, 3-traj without; no ellipse).
- [ ] Fresh ESKF per run (no state leak — a repeated (traj,seed) run reproduces identical metrics).
- [ ] `runs.csv` (per-seed, long-format) + `summary.csv` (mean±std by controller×trajectory) written for both variants.
- [ ] Controller ranking reported for each variant; frozen-ESKF tracking cross-referenced against the oracle-clean baseline.

## 11. Out of Scope

- Any retuning of controllers or the estimator.
- **MPC finalization:** MPC has only a coarse/single-run config (`runs_controller/mpc_clean/best_config.json`, best_score≈67 vs ASMC 15 / PID 24, and it fails grossly on pose). Include it in the comparison if desired, but flag in the output that it is **not** finalized/multi-seeded like ASMC/PID, and its known pose-horizon/basin rewire is pending. Do not attempt the MPC rewire here.
- Ellipse / posref trajectories (excluded by request).
- Figure generation (done in the originating session).
- A true pre-slew controller-chatter metric (note as a possible follow-up).
```
