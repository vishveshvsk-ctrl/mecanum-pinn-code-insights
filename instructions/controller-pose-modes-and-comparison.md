# Controller Pose Modes (MPC + PID) + Three-Controller Comparison Protocol

> **Generated:** 2026-07-22
> **Stack:** Julia 1.10+ (matches `code_insights/Project.toml`), OSQP.jl, OrdinaryDiffEq.jl, DiffEqCallbacks.jl, StaticArrays.jl, DataFrames/Arrow — edits within `hybrid_ctrl/` + reuse of the estimator-tuner harness
> **Scope:** Control-law implementation (MPC/PID pose modes) + experiment protocol (velocity-transit / pose-docking comparison)
> **Downstream target:** A coding model implements the pose modes and the comparison driver from this spec. **No code is written here.**
> **Depends on (must land first):** `instructions/estimator-rewrite-posref-logging.md` (accel-fused estimators, ASMC `:pose`, `PoseFixModel`, per-tick logging) and `instructions/estimator-kf-smo-tuner.md` (the reusable tuning harness + a **frozen** estimator config).

---

## 1. Overview

Two deliverables so all three control laws can be compared under the project's realistic localization model. **(A) Pose modes:** give **MPC** and **PID** a `:pose` tracking mode (ASMC `:pose` already exists), so every controller can track a `PosRef` on the **estimated, exteroceptively-corrected** pose. MPC gets a **position-augmented** linear model + position-tracking cost; PID gets a **cascade** (outer position loop → velocity setpoint → existing inner velocity PID). **(B) Comparison protocol:** a driver that tunes each controller *fairly* on the **frozen** estimator (reusing the tuning harness), then evaluates all three across two operating regimes — **transit** (velocity tracking, intermittent fix) and **docking** (pose tracking, precise fix) — plus a **fix-dropout robustness** experiment, reporting a Pham-&-Han-comparable metric table.

**System-level contract:** after this work, `run_hybrid` can drive any controller in either mode; a single `run_comparison` call produces the ranked, effort-normalized metric tables (transit + docking + dropout) that back the paper's control-law comparison.

**Governing decisions (do not re-derive):** the estimator is a **frozen controlled variable** (tuned first, per the estimator-first plan); comparison fairness requires **equal control-effort normalization** + a **common excitation/subset/seed set**; pose is bounded only where the exteroceptive fix is reliable, so the **pose comparison lives at docking** (fix reliable) and the **velocity comparison lives in transit** (fix intermittent).

---

## 2. Architecture Pattern

**Mode-extension of the existing wrench controllers + a harness-reusing comparison driver.** MPC and PID keep their signatures and gain the `:pose` branch (mirroring how ASMC already carries `:velocity`/`:pose`); the mixer/plant/motor path is untouched. The comparison driver is a thin instantiation of the estimator-tuner spine (subset → run_and_log → objective → optimizer → results), swapping in controller parameter spaces and objectives — so "controller tuning follows the estimator harness" is realized literally.

---

## 3. Technology Constraints

- **Language:** Julia 1.10+; run from `code_insights/`.
- **Reuse (do not reinvent):**
  - `hybrid_ctrl/controllers.jl` — existing `MPCController`/`mpc_wrench!` (OSQP, position-*velocity* model, hard V/i/slew constraints) and `PIDController`/`pid_wrench!` (velocity-error PID); extend, don't replace.
  - `run_one.jl:asmc_torques` — the authoritative position sliding law already ported into `asmc_wrench!(:pose)` by the estimator-rewrite brief; ASMC pose is a dependency, not built here.
  - The **tuning harness** (`tuning/` from `estimator-kf-smo-tuner.md`): `TuningSubset`, `run_and_log`, `Optimizer`, `parallel_evaluate`, `ResultStore` — reused verbatim.
  - The **frozen estimator** config produced by the estimator tuner; the `PoseFixModel` (transit/docking tiers) from the estimator-rewrite brief.
  - `Profiles` (`PosRef`/`VelRef`, `global_to_local_frame`, `current_posref`).
- **QP:** OSQP.jl (already used by the velocity MPC), warm-started, hard constraints.
- **Explicit exclusions:** no estimator changes (frozen); no plant/mixer/motor changes; no new fuzzy design; no PINN/GPU/hardware.

---

## 4. Component Breakdown

### `mpc_wrench!` — `:pose` branch  (position-augmented MPC)
- **Type:** extend the existing `function` in `controllers.jl` + extend `MPCController`.
- **Responsibility:** finite-horizon QP tracking a `PosRef` on the **estimated pose+velocity**, in per-wheel voltage, with the same hard constraints as velocity mode.
- **Model (augmented, 6-state):** `z = [X, Y, ψ, Vx, Vy, ψ̇]`. Velocity block = the existing `A_v, B_v` (voltage→body-accel). Position block integrates body velocity through the **frozen** current heading `ψ̂`: `X⁺ = X + dt·(Vx·cosψ̂ − Vy·sinψ̂)`, `Y⁺ = Y + dt·(Vx·sinψ̂ + Vy·cosψ̂)`, `ψ⁺ = ψ + dt·ψ̇`. Assemble `A_aug (6×6)`, `B_aug (6×4) = [0; B_v]`.
- **Reference:** `PosRef` world pose `[xo, yo, psi]` (+ velocity feedforward `[Vxo,Vyo,om]`) previewed over the horizon.
- **Cost:** `Σ‖z − z_ref‖²_Q_pose + ‖U‖²_R + ‖ΔU‖²_S`, with `Q_pose` weighting **position ≫ velocity** (reference point: Pham & Han `Q=diag([10,10,5,1,1,0.5])`).
- **Constraints:** identical hard voltage/current/slew box (unchanged) — re-used from the velocity path.
- **State fed:** `z0 = [X̂o,Ŷo,ψ̂, V̂x,V̂y,ψ̂̇]` from `bus.xhat` (bounded by the exteroceptive fix in `:pose`).
- **Output:** first-step voltage → equivalent task-space wrench (same mapping as velocity mode) → mixer.
- **New `MPCController` fields:** `Q_pose::SVector{6}`, `Np_pose::Int` (may differ from velocity `Np`).
- **Depends on:** existing `_mpc_body_matrices`, OSQP path, `global_to_local_frame`.

### `pid_wrench!` — `:pose` branch  (cascade) + `pose_outer_loop`
- **Type:** extend the existing `function` + a helper + extend `PIDController`.
- **Responsibility:** pose tracking via a standard **cascade** — an outer position loop turns pose error into a body-velocity setpoint that the existing inner velocity PID tracks (matches deployed practice and reuses the vetted inner loop).
- **`pose_outer_loop(xhat, ref, gains) -> V_cmd`:** world position error `(X̂o,Ŷo) − (ref.xo,ref.yo)` rotated to body; `V_cmd = [Vxo,Vyo,om]_body_ff − Kp_pos·e_pos_body (− Kd_pos·ė_pos)`; heading error via smooth wrap. Outer loop is P/PD (no integral) to avoid double-integrator windup.
- **Inner:** the existing velocity PID tracks `V_cmd` (anti-windup on its integral).
- **New `PIDController` fields:** `Kp_pos::SVector{3}`, `Kd_pos::SVector{3}` (outer gains).
- **Depends on:** existing velocity `pid_wrench!`, `global_to_local_frame`.
- **Note:** a direct position-PID (single loop) is an acceptable simpler alternative if the cascade proves hard to tune — document whichever is used.

### `DockingProfile` / `build_docking` (new `PosRef` trajectory)
- **Type:** a `PosRef` builder registered in `Profiles.BUILDERS`.
- **Responsibility:** the docking regime's reference — approach a target pose from an offset and **hold** (settle), so pose-tracking metrics (settling, overshoot, steady pose RMSE) are meaningful and the precise docking fix tier applies.
- **Outputs:** `PosRef` (approach ramp → hold).
- **Depends on:** `Profiles` smoothstep helpers.

### Controller `ParamSpace` / `apply_params!` (harness slot)
- **Type:** `struct` + `function` per controller, mirroring the estimator tuner's slot.
- **Responsibility:** declare tunable gains + bounds per controller and decode a parameter vector into a config.
- **Spaces:** ASMC (`γ`, `ε`, `K_max`, `λ` ranges, `σ`, `K0`), MPC (`Q`/`Q_pose`, `R`, `S`, `Np`/`Np_pose`), PID (velocity `Kp,Ki,Kd,I_max` + pose `Kp_pos,Kd_pos`). Hard actuator limits stay in `MotorParams` (not tuned).
- **Depends on:** the extended controller structs.

### `controller_objective` (harness slot)
- **Type:** `function` (mode-aware, effort-normalized).
- **Responsibility:** score a controller run for tuning/ranking.
- **Definition:** **transit/velocity** → velocity NRMSE (slip-weighted) at a **normalized control effort**; **docking/pose** → pose RMSE + settling + overshoot at normalized effort. Effort = `CE = mean Σ|Uᵢ|` (Pham & Han eq. 50). Enforce **equal-effort**: either constrain candidates to a target CE band or report tracking *at matched CE* so no controller wins by spending more voltage.

### `MetricSet` / `compute_metrics`
- **Type:** `struct` + `function`
- **Responsibility:** the full comparison metric vector per run.
- **Metrics:** pose RMSE `√(mean[(x−x_ref)²+(y−y_ref)²+(ψ−ψ_ref)²])` (Pham & Han eq. 49); velocity NRMSE; control effort `CE`; max error `Emax`; **settling time**; **overshoot**; **disturbance-rejection factor**; in-slip velocity NRMSE; pose-drift rate; fix-dropout drift. True state from `sol.u`; estimates from the probe log.

### `ComparisonDriver` / `run_comparison`
- **Type:** `function` (top-level experiment) + CLI `compare_controllers.jl`
- **Responsibility:** run the full selection matrix on the **frozen estimator** and emit the ranked tables.
- **Selection matrix:** `{ASMC, MPC, PID}` × `{transit(:velocity, intermittent fix), docking(:pose, precise fix)}` × `{fix conditions: continuous | intermittent | dropout-N s}`, over the curated subset (transit) + docking profile, across the seed set.
- **Fairness:** frozen estimator; each controller pre-tuned via the harness to matched effort/bandwidth; identical seeds/subset; report mean ± std.
- **Outputs:** `runs_comparison/{transit_table.arrow, docking_table.arrow, dropout_table.arrow, best_gains.json, diagnostics/*.png}`.

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl/
│   ├── controllers.jl    # EXTEND: mpc_wrench! :pose (aug model + Q_pose),
│   │                     #         pid_wrench! :pose (cascade) + pose_outer_loop,
│   │                     #         MPCController/PIDController new fields
│   └── scheduler.jl      # (reuse) ref-dispatch + mode already threaded
├── profiles.jl           # ADD: build_docking (PosRef approach-and-hold) + registry entry
├── tuning/               # REUSE the estimator-tuner spine unchanged
│   ├── param_space.jl    # ADD: asmc/mpc/pid spaces + apply_params!
│   ├── objectives.jl     # ADD: controller_objective (effort-normalized, mode-aware)
│   └── metrics.jl        # NEW: MetricSet + compute_metrics (Pham-Han + velocity/slip)
├── compare_controllers.jl# NEW: run_comparison CLI (selection matrix → tables)
└── runs_comparison/      # OUTPUT: transit/docking/dropout tables + gains + plots
```

No estimator/plant/mixer/motor files change.

---

## 6. Key Interfaces

Signatures + docstrings only. Bodies are always `# ...`.

```julia
"""
    mpc_wrench!(bus, xhat, ref, params, motor, mpc; mode=:velocity) -> SVector{3}

`:velocity` unchanged. `:pose` builds the 6-state augmented model
z=[X,Y,ψ,Vx,Vy,ψ̇] (position integrates body velocity through frozen ψ̂), tracks a
PosRef world pose over the horizon with Q_pose (position ≫ velocity), same hard
V/i/slew constraints, on the estimated pose z0 from bus.xhat[1:6].
"""
function mpc_wrench!(bus, xhat, ref, params, motor, mpc; mode=:velocity)
    # ...
end

"""
    pose_outer_loop(xhat, ref, pid) -> SVector{3}

Cascade outer loop: world position error (X̂o,Ŷo,ψ̂)−(ref.xo,ref.yo,ref.psi)
→ body frame → body-velocity setpoint V_cmd = ff(Vxo,Vyo,om) − Kp_pos·e (−Kd_pos·ė).
"""
function pose_outer_loop(xhat, ref, pid)
    # ...
end

"""
    pid_wrench!(bus, xhat, ref, pid, dt; mode=:velocity) -> SVector{3}

`:velocity` unchanged. `:pose` runs pose_outer_loop → V_cmd, then the existing
inner velocity PID tracks V_cmd (anti-windup on the inner integral).
"""
function pid_wrench!(bus, xhat, ref, pid, dt; mode=:velocity)
    # ...
end

"""
    compute_metrics(log, ref, mode) -> MetricSet

Pose RMSE / velocity NRMSE / CE / Emax / settling / overshoot / disturbance-
rejection / in-slip NRMSE / pose-drift / dropout-drift from one run's aligned log.
"""
function compute_metrics(log, ref, mode)
    # ...
end

"""
    run_comparison(estimator_frozen, subset, docking, seeds; effort_target, out) -> tables

Selection matrix {ASMC,MPC,PID}×{transit,docking}×{fix conditions} on the FROZEN
estimator, each controller pre-tuned to matched effort via the tuning harness;
emits ranked transit/docking/dropout metric tables (mean±std).
"""
function run_comparison(estimator_frozen, subset, docking, seeds; effort_target, out)
    # ...
end
```

---

## 7. Data Flow

**Tuning (per controller, reusing the harness):** `Optimizer.ask` → `apply_params!` → `run_and_log` (frozen estimator, fixed mode) → `controller_objective` (effort-normalized) → `Optimizer.tell`; freeze best gains per controller per regime.

**Comparison (per matrix cell):**
1. Load frozen estimator + frozen controller gains.
2. `run_hybrid` with `cfg.tracking = :velocity` (transit) or `:pose` (docking), the matching `PoseFixModel` tier + fix condition, on the subset/docking trajectory.
3. Per control tick: the controller's `:velocity`/`:pose` branch reads `bus.xhat` (velocity, or exteroceptively-corrected pose) → wrench → mixer (unchanged).
4. `build_estimator_log` + `compute_metrics` → one `MetricSet`.
5. `run_comparison` aggregates across seeds → mean±std tables.

**Control law reads estimate, not truth (unchanged):** all pose modes consume `bus.xhat` pose (fix-corrected), never `sol.u`; truth enters only `compute_metrics`.

---

## 8. Implementation Sequence

1. **`build_docking` PosRef** — the docking reference must exist before pose modes can be exercised meaningfully.
2. **`pid_wrench!` `:pose` (cascade)** — simplest pose mode; validate it holds the docking pose with the precise fix.
3. **`mpc_wrench!` `:pose` (augmented model)** — extend the QP to 6 states + `Q_pose`; validate feasibility, warm-start stability, and that it tracks the docking approach within constraints.
4. **`MetricSet` / `compute_metrics`** — the shared scoring (Pham-Han + velocity/slip); unit-test settling/overshoot on a known step.
5. **Controller `ParamSpace` + `apply_params!` + `controller_objective`** — plug into the harness; verify a detuned controller scores worse and that effort-normalization changes rankings.
6. **Tune each controller** (ASMC/MPC/PID) per regime on the frozen estimator to matched effort; freeze gains.
7. **`run_comparison` + `compare_controllers.jl`** — selection matrix → transit/docking/dropout tables; verify the dropout experiment shows bounded-then-drift behavior per controller.

---

## 9. Numerical / Control / Fairness Considerations

- **Equal-effort is non-negotiable:** report tracking at matched `CE` (or constrain to an effort band). Without it the comparison just measures who spent more voltage. State the effort next to every tracking number.
- **Frozen estimator:** identical estimator config/instance across all controllers and cells — the estimator is a controlled variable, not co-tuned with controllers.
- **MPC pose linearization:** freezing `ψ̂` over the horizon is fine for short `Np`; if heading slews fast, shorten `Np_pose` or re-linearize per tick. Watch QP conditioning as the state grows 3→6 (scale/regularize as the velocity path already does).
- **Cascade windup:** keep the outer loop P/PD (no integral); only the inner velocity PID integrates, with anti-windup — otherwise stacked integrators wind up during the docking approach.
- **Pose observability gate:** pose modes are only meaningful where the fix is reliable (docking) or short-horizon (transit). Do **not** report a pose comparison in a fix-dropout segment as control-law quality — that segment is the *robustness* experiment (drift bound), not the tracking comparison.
- **Sawtooth from intermittent fixes:** the drift-then-snap of intermittent corrections is a disturbance to the controller; verify no controller destabilizes on a fix jump (the fix gating in the estimator bounds jump size).
- **Design-equivalent tuning (defensible story):** in addition to effort-matching, target common closed-loop specs (ζ≈0.707, PM≈60°, matched bandwidth) — this is how Pham & Han justify their gains, so matching it beats them on their own methodology.
- **Reproducibility:** frozen estimator + frozen gains + seed set + subset/docking manifest reproduce every table cell.
- **Parallelism:** reuse the harness executor cap (heavy stiff sims; low concurrency).

## 10. Success Criteria

- [ ] `pid_wrench!(:pose)` and `mpc_wrench!(:pose)` both close the loop on the docking profile using **estimated** pose; bounded, no divergence, within actuator limits.
- [ ] MPC pose QP stays feasible/warm-started at the 6-state size; falls back gracefully on infeasibility.
- [ ] `compute_metrics` reproduces settling/overshoot on a known step and matches hand-computed RMSE/CE on a fixture.
- [ ] Each controller is tuned to a **matched effort** (CE within band) on the frozen estimator; gains frozen and logged.
- [ ] `run_comparison` emits transit (velocity), docking (pose), and dropout tables with mean±std over seeds — all three controllers present in each.
- [ ] Docking pose comparison is **unconfounded** (fix reliable → pose bounded), directly comparable to Pham & Han's position-RMSE framing.
- [ ] Fix-dropout experiment shows per-controller drift growth during dropout and recovery on re-anchor.
- [ ] Reproducible: frozen estimator + gains + seeds → identical tables.

## 11. Out of Scope

- **Estimator changes** — frozen; any estimator work is the estimator-rewrite/tuner briefs.
- **Fuzzy supervisor redesign** — the existing blend is reused; controllers are compared solo and (optionally) blended, not re-architected here.
- **New actuator/plant/mixer** — untouched.
- **A learned/PINN controller**, GPU, hardware/real-time.
- **The exteroceptive `PoseFixModel` mechanism itself** — provided by the estimator-rewrite brief; consumed here.
