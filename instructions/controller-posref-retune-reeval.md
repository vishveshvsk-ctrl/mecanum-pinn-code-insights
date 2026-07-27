# Controller Pose-Tracking Retune & Re-eval (velref → posref adapter)

> **Generated:** 2026-07-24
> **Stack:** Julia (project at `code_insights/Project.toml`); reuses `profiles.jl`, `hybrid_ctrl/*.jl`, `tune_controller.jl`
> **Scope:** Reference adapter + harness edit + 5-seed retune + re-eval — **ASMC & PID only** (no MPC)

---

## 1. Overview

The 4 velref trajectories (octagon, spin_creep, coupled_vomega, spiral_orbit) were tracked in **velocity mode**: `pid_wrench!`/`asmc_wrench!` `:velocity` feeds back **velocity + yaw-rate only**, so (a) the **2-loop cascade** (`Kp_pos`/`Kd_pos` outer loop — the part we fixed) was never exercised on them, and (b) position `X,Y=∫V` and heading `ψ=∫ψ̇` **drift unboundedly** (bias integrates under noise). This work makes **all** trajectories pose-tracked: construct a `PosRef` from each `VelRef` by transforming body velocity to world frame through the reference heading and integrating to position; route all entries through the existing `:pose` control path (position + heading feedback, **velocity feedforward retained** — the controllers thereby "accept both" the velocity feedforward *and* the position reference via one complete `PosRef`); then **re-tune ASMC and PID (5-seed each)** on the all-pose subset and **re-eval** clean + noise with pose metrics.

**System contract.** Input: the existing 6-combo subset + the tuned/whitelisted configs. Output: bounded pose-tracking gains (ASMC + PID, 5-seed, logged) and a pose-metric clean+noise comparison, replacing the drifting velocity-only results. Nothing in the plant/estimator/mixer/controller *logic* changes — only a reference adapter, harness routing, subset definition, and metric mode.

## 2. Architecture Pattern

**Reference-adapter + unified pose harness.** A pure functional adapter lifts `VelRef → PosRef`; the *already-present* `:pose` control path in both controllers consumes it unchanged. Justification: both controllers already implement full pose modes (position feedback + velocity/accel feedforward); the only missing ingredient for the velref-designed trajectories is a position reference — a **data adapter, not new control logic**. This keeps the change small, reversible, and orthogonal to the tuned control laws.

## 3. Technology Constraints

- **Language:** Julia; run single-threaded (`julia -t 1`, `--project=. --startup-file=no`, `BLAS.set_num_threads(1)`).
- **Reuse (no rewrite):** `profiles.jl` (`VelRef`, `PosRef`, `global_to_local_frame`), `hybrid_ctrl/controllers.jl` (`asmc_wrench!`/`pid_wrench!` `:pose`, `pose_outer_loop`), `tune_controller.jl` (`run_controller`, `controller_metrics`, `default_trajs_3`, tuning driver, `--trajset`), `hybrid_ctrl/scheduler.jl` (`run_hybrid`).
- **Determinism:** refs built with pinned `combo_idx` + `Random.Xoshiro(0)` (existing pattern); adapter is deterministic given its `VelRef`.
- **Explicit exclusions:** **no MPC**; no changes to plant, estimator, mixer, or the controller wrench laws; no new control logic. Only: adapter + ref-routing + subset + metric-mode + retune/re-eval scripts.

> **⚠ PRESERVATION CONSTRAINT (hard requirement).** The velocity-mode v1 results are the fallback for initial submission and MUST NOT be overwritten (archived at `runs_controller_v1_velocity_ARCHIVE/`). Therefore: **(1)** all edits are **additive** — do NOT modify `default_trajs_3` or the `--trajset 3`/`:velocity` behavior of `run_controller`; the velref→posref routing must be **gated** on the new pose subset only, so `--trajset 3` still reproduces v1 exactly. **(2)** Do NOT edit `experiment_noise_eval.jl` in place — create a **new** `experiment_noise_eval_pose.jl`. **(3)** All pose outputs go to **new** dirs/files (`runs_controller_asmc_pose_5seed/`, `runs_controller_pid_pose_5seed/`, `noise_eval_pose_10seed.csv`); never write into `runs_controller_asmc_pin/`, `runs_controller_pid_5seed/`, `runs_controller/noise_eval_10seed.csv`, `runs_controller/RESULTS_controller_tuning.md`, or `runs_controller/viz/*`. Verify each target dir is free before writing.

## 4. Component Breakdown

### `velref_to_posref` (NEW — the core adapter)  — `profiles.jl`
- **Type:** pure `function`.
- **Responsibility:** build a complete `PosRef` from a `VelRef` by rotating body velocity to the world frame through the reference heading and integrating to position (with Coriolis-correct world acceleration).
- **Inputs:** `vr::VelRef`; optional `x0,y0` origin (default 0,0); optional integration grid step.
- **Outputs:** `PosRef` with all 9 getters populated:
  - `psi = vr.psi`, `om = vr.Wz`, `al = vr.al`
  - `Vxo(t) = vr.Vx·cosψ − vr.Vy·sinψ`, `Vyo(t) = vr.Vx·sinψ + vr.Vy·cosψ`
  - `xo(t) = x0 + ∫₀ᵗ Vxo`, `yo(t) = y0 + ∫₀ᵗ Vyo` (numerical cumulative integral → interpolated getter)
  - `Axo(t) = vr.Ax·cosψ − vr.Ay·sinψ − vr.Wz·Vyo`, `Ayo(t) = vr.Ax·sinψ + vr.Ay·cosψ + vr.Wz·Vxo` (world accel incl. Coriolis)
  - `tstops = vr.tstops`, `T_total = vr.T_total`
- **Depends on:** `VelRef`/`PosRef` structs; trig transform through `vr.psi`.

### `build_ref` routing (EDIT) — inside `run_controller`, `tune_controller.jl`
- **Responsibility:** for a subset entry flagged velref-derived-pose, build the `VelRef` then wrap with `velref_to_posref` → `PosRef` and run `:pose`; for native posref entries (ellipse), build the `PosRef` directly. All entries → `run_mode=:pose`.
- **Inputs:** subset entry `tr`, `run_dir`, rng.
- **Outputs:** `(PosRef, :pose)`.
- **Depends on:** `velref_to_posref`, `Profiles.resolve_profile`/`build`.

### `default_trajs_pose` (NEW subset) — `tune_controller.jl`
- **Type:** `function`.
- **Responsibility:** the all-pose 6-trajectory subset — same combos as `default_trajs_3` (octagon 206, spin_creep 178, coupled_vomega 12, spiral_orbit 37, ellipse_tangent 55, ellipse_crab 83) but **every entry `run_mode=:pose`**; the 4 former-velref entries carry an `adapt=true` flag routing them through `velref_to_posref`.
- **Wiring:** new `--trajset "pose"` (or "4") value in `main`'s trajset selector.

### `asmc_wrench!` / `pid_wrench!` `:pose` (REUSE — verify only)
- Both consume the full `PosRef`. **ASMC pose** reads `xo/yo/psi` (feedback) + `Vxo/Vyo/om` (velocity feedforward) + `Axo/Ayo/al` (accel feedforward, via `global_to_local_frame`). **PID pose** (`pose_outer_loop`) reads `xo/yo/psi` + `Vxo/Vyo/om`. The adapter MUST populate all fields ASMC needs — confirm no `nothing`/undefined getter.

### `controller_metrics` `:pose` (REUSE)
- Already computes `final_pos/max_pos/final_head/max_head` from `probe.u[17,18,4]` (true pose) vs `ref.xo/yo/psi`. Selected automatically because `run_mode=:pose`. This replaces the velocity metrics for the 4 former-velref trajectories.

### Retune driver + eval scripts (EDIT/REUSE)
- `tune_controller.jl` — add `--trajset "pose"` (additive branch); `run_controller` velref→posref routing **gated to pose entries only** (v1 `--trajset 3`/`:velocity` path untouched). Re-tune ASMC (5-seed) + PID (5-seed) into NEW dirs.
- `experiment_noise_eval_pose.jl` (**NEW file — copy of `experiment_noise_eval.jl`, do NOT edit the original**) — pose subset; emit **pose** metrics (`final_pos_cm`, `max_pos_cm`, `final_head_rad`, `max_head_rad`) + `ce` + `chatter` per row → `runs_controller/noise_eval_pose_10seed.csv`.

## 5. File & Directory Structure + Code-Change Log

```
code_insights/
├── profiles.jl                     # ADD  velref_to_posref (adapter, near PosRef)
├── tune_controller.jl              # ADD  default_trajs_pose + --trajset "pose"
│                                   # EDIT run_controller: route velref-pose entries through velref_to_posref
├── experiment_noise_eval_pose.jl   # NEW (copy of experiment_noise_eval.jl; original UNTOUCHED)
├── runs_controller_asmc_pose_5seed/{seed1..5}/   # OUTPUT (ASMC retune — new dir)
├── runs_controller_pid_pose_5seed/{seed1..5}/    # OUTPUT (PID retune — new dir)
├── runs_controller/noise_eval_pose_10seed.csv    # OUTPUT (re-eval — new file)
└── runs_controller_v1_velocity_ARCHIVE/          # DO NOT TOUCH — v1 fallback
```

| File | Symbol | Change | Note |
|---|---|---|---|
| `profiles.jl` | `velref_to_posref` | **ADD** | pure adapter, 9-getter PosRef |
| `tune_controller.jl` | `default_trajs_pose` | **ADD** | all-pose 6-traj subset |
| `tune_controller.jl` | `--trajset` selector | **EDIT (additive)** | add `"pose"` branch; `"3"` unchanged |
| `tune_controller.jl` | `run_controller` ref-build | **EDIT (gated)** | adapt velref→posref for POSE entries only; velocity path preserved |
| `experiment_noise_eval_pose.jl` | (whole script) | **ADD (copy)** | do NOT edit the velocity original |

## 6. Key Interfaces

Signatures + docstrings only; bodies are stubs.

```julia
"""
    velref_to_posref(vr::VelRef; x0=0.0, y0=0.0, dt=nothing) -> PosRef

Lift a body-velocity reference to a full global-frame position reference by
rotating (Vx,Vy) through the reference heading vr.psi and integrating to (xo,yo).
Fills all 9 PosRef getters:
  xo,yo   = x0,y0 + cumulative-integral of (Vxo,Vyo)      [world position]
  psi     = vr.psi                                         [heading]
  Vxo,Vyo = R(psi)*(vr.Vx, vr.Vy)                          [world velocity FF]
  om      = vr.Wz
  Axo,Ayo = R(psi)*(vr.Ax, vr.Ay) + Coriolis(vr.Wz x Vworld)   [world accel FF]
  al      = vr.al
Integration is numerical on a uniform grid (<= solver dt) with interpolated
getters; deterministic given vr. Origin (x0,y0) should match the plant's
initial pose so no constant offset is charged as tracking error.
"""
function velref_to_posref(vr, ; x0=0.0, y0=0.0, dt=nothing)
    error("stub")
end

"""
    default_trajs_pose(run_dir) -> Vector{NamedTuple}

All-pose 6-traj subset: octagon(206), spin_creep(178), coupled_vomega(12),
spiral_orbit(37) [velref profiles, adapt=true -> velref_to_posref] +
ellipse_tangent(55), ellipse_crab(83) [native posref]. Every entry run_mode=:pose.
"""
function default_trajs_pose(run_dir::String)
    error("stub")
end
```

CLI (retune + re-eval):
```
# retune (5 seeds each, single-threaded, separate dirs)
julia -t 1 --project=. --startup-file=no tune_controller.jl \
  --controller asmc --optimizer dxnes --budget 50 --trajset pose --noise clean \
  --seed <N> --out runs_controller_asmc_pose_5seed/seed<N>
julia -t 1 ... --controller pid --budget 150 --trajset pose --seed <N> \
  --out runs_controller_pid_pose_5seed/seed<N>
# re-eval (pose metrics, clean + 1x/2x/5x, 10 seeds)
julia -t 1 --project=. --startup-file=no experiment_noise_eval.jl   # edited to pose subset
```

## 7. Data Flow

1. Subset entry (octagon 206, velref profile) → build `VelRef` → `velref_to_posref` → `PosRef` (xo,yo,psi + Vxo,Vyo,om + Axo,Ayo,al). Ellipse entries → `PosRef` directly.
2. `run_hybrid` `:pose`: sensor callback writes `bus.xhat=[Vx,Vy,ψ̇,ψ,X,Y]`; the controller's `:pose` branch feeds back estimated pose `xhat[5,6,4]` vs `PosRef.xo/yo/psi` (PID via `pose_outer_loop` cascade → velocity setpoint → inner PID; ASMC via pose sliding surface), with world-frame velocity/accel feedforward rotated to body via `global_to_local_frame`.
3. `controller_metrics` `:pose`: **true** pose (`probe.u[17,18,4]`) vs `PosRef` → bounded `final_pos/max_pos/final_head/max_head` (no ramp).
4. **Tuning:** objective = pose tracking (tolerance-normalized) averaged over all 6 pose trajs; dxNES, 5 seeds; ASMC then PID.
5. **Re-eval:** clean + 1×/2×/5× × 10 noise seeds → pose metrics + ce + chatter → CSV for figures.

## 8. Implementation Sequence

1. **`velref_to_posref`** + round-trip unit check (a velref built from a known profile: verify `d/dt(xo) ≈ Vxo`, `xo` matches trapezoidal `∫Vxo`, all 9 getters finite over `[0,T_total]`). Unblocks everything.
2. **`run_controller` ref-build edit + `default_trajs_pose` + `--trajset "pose"`.**
3. **Smoke:** one ASMC pose run and one PID pose run on an adapted octagon → finite, **bounded** pos/heading error (no linear ramp) under **clean and noisy**; confirm PID's `pose_outer_loop` (Kp_pos/Kd_pos) is actually invoked (nonzero outer contribution).
4. **Retune ASMC 5-seed** (`--trajset pose`), then **PID 5-seed**. Log scores + gains; check score reproducibility.
5. Pick representative seeds (as before), **re-eval** clean+noise, regenerate figures with pose-metric columns.

## 9. Julia / simulation-specific considerations

- **Integration grid & smoothness:** integrate `Vxo/Vyo` on a uniform grid ≤ solver `dt`; the resulting `xo/yo` getters must be smooth and **Dual-safe** (the ODE/feedforward path may autodiff the ref) — match the pattern the closed-form `PosRef` builders use (e.g. `build_ellipse`).
- **Origin alignment:** set `(x0,y0)` (and `psi(0)`) to the plant's initial pose so the run doesn't start with a constant offset that `controller_metrics` would charge as error.
- **Heading:** use `vr.psi` (unwrapped) for the rotation; heading error uses `_wrap_angle` in controllers/metrics (already handled) — do not re-wrap inside the adapter.
- **Feedforward completeness (ASMC):** ASMC pose reads `Axo/Ayo/al`; the adapter MUST provide Coriolis-correct world accel — omitting it silently degrades ASMC's feedforward and biases the comparison against ASMC.
- **Determinism:** same pinned `combo_idx` + `Xoshiro(0)`; adapter deterministic; only the noise seed varies in eval.
- **Fresh mutable state per run:** controllers/estimator are mutable — rebuild per (traj, seed) as the existing harness does.
- **Feasibility re-check:** velref combos were feasibility-pinned in *velocity* terms; in pose mode the tracked path is identical, but confirm the pose loop keeps up (no new infeasibility). spin_creep pose = hold position + track a spinning heading (near-stationary translation, large heading sweep) — expect heading, not position, to dominate its error.
- **Metric-mode consistency:** since all entries are now `:pose`, the tuning objective and eval both use the pose branch uniformly — no mixed velocity/pose aggregation (which previously combined incommensurable units).

## 10. Success Criteria

- [ ] `velref_to_posref` round-trip: `d/dt(xo) ≈ Vxo`, `xo` matches `∫Vxo`, all 9 getters finite on `[0,T_total]`.
- [ ] Smoke: ASMC & PID pose runs on all 6 adapted trajs give **finite, bounded** pos/heading error (no ramp) under clean **and** noisy.
- [ ] PID 2-loop confirmed active on all 6 (nonzero `Kp_pos`/`Kd_pos` outer contribution — the previously-untested cascade).
- [ ] ASMC 5-seed + PID 5-seed retuned; scores reproducible; gains logged to `best_config.json`.
- [ ] Re-eval CSV (pose metrics + ce + chatter) for clean + 1×/2×/5× × 10 seeds, both controllers, ready for figures.

## 11. Out of Scope

- **MPC** — this brief is ASMC + PID only.
- **Frozen-ESKF controller comparison** (`instructions/controller-comparison-frozen-eskf.md`) — separate; note it **inherits this same velref→pose fix** and must adopt `velref_to_posref` + pose mode before it is run.
- Plant / estimator / mixer / controller-wrench-law changes.
- New figure styling — reuse the existing `runs_controller/viz/*.py` scripts with the pose-metric columns swapped in.
