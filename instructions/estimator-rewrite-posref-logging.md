# Estimator Rewrite + Pose-Mode Control + Probe Logging — Prerequisites for the Estimator Tuner

> **Generated:** 2026-07-22
> **Stack:** Julia 1.10+ (matches `code_insights/Project.toml`), OrdinaryDiffEq.jl, DiffEqCallbacks.jl, StaticArrays.jl, DataFrames/Arrow — edits within the existing `hybrid_ctrl/` modules
> **Scope:** Estimator/controller/logging implementation (the three blockers named in `instructions/estimator-kf-smo-tuner.md` §3)
> **Downstream target:** A coding model implements these three changes so the estimator tuner can run. **No code is written here.**

---

## 1. Overview

Implement the three prerequisites the estimator-tuner brief depends on: **(1)** rewrite both estimators (`KalmanEstimator`, `SMOEstimator`) in `hybrid_ctrl/estimators.jl` to fuse the **accelerometer** (slip-independent body-acceleration reference), fix the `Rn` dimension bug, add **accel-bias states** and **slip-adaptive measurement weighting** (KF: inflate wheel-velocity `R` on slip; SMO: gate the sliding correction + ZUPT re-anchor), and expose every new knob as a tunable kwarg; **(2)** wire the **pose-tracking control mode** (`asmc_wrench!` `:pose`), ported from the authoritative simulator's `asmc_torques` position law in `run_one.jl`, so PosRef trajectories (ellipse) close the loop on the **estimated** pose; **(3)** finalize the **per-tick estimator probe log** in `hybrid_ctrl/scheduler.jl` (a raw scaffold already exists) into a structured record the tuner's `run_and_log` consumes.

**System-level contract:** after this work, `run_hybrid` can run any curated-subset trajectory in either tracking mode, both estimators use IMU accel + encoders (not slip-corrupted odometry alone), and every run yields a time-aligned `(true state, estimated state, slip)` log — with all estimator hyperparameters settable from outside.

**Governing design decisions (from prior analysis; do not re-derive):** the accelerometer is the only slip-independent translational reference; the wheel-velocity pseudo-measurement is slip-corrupted and must be *down-weighted during slip*; pose is unobservable from proprioception, so under **PosRef the position is supplied as a noisy exteroceptive fix** (`PoseFixModel`, deployable LiDAR-AMCL/fiducial, two-tier for hospital micrologistics) that bounds pose drift, while VelRef stays proprioceptive-only; the estimated pose — not truth — feeds the controller. See the session summary (`session_summaries/SESSION_INFERENCES_smo_hybrid_architecture_tuning.md` §6) for the two-tier rationale.

---

## 2. Architecture Pattern

**In-place enhancement of existing `hybrid_ctrl/` blocks under the unchanged ZOH-bus / `PeriodicCallback` contract.** The estimators keep their `estimator_update!(bus, y, est, params, dt)` signature (the accel `y.a_x, y.a_y` is already delivered — currently ignored); the controller keeps its `asmc_wrench!(…; mode)` signature (only the `:pose` branch is filled in); the scheduler keeps its callback structure (only the probe logger is finalized). No new subsystem, no change to the plant or mixer — this is a targeted upgrade of three files so the tuner's prerequisites are met without disturbing the validated simulation path.

---

## 3. Technology Constraints

- **Language:** Julia 1.10+; run from `code_insights/` with the project env.
- **Reuse (authoritative sources, do not reinvent):**
  - **Accel-fusion reference:** the `KalmanEstimator`/`SMOEstimator` in `Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1_sonnet5.ipynb` already put IMU accel in the prediction (`v_pred = v_prev + dt·a_ff`, `a_ff=[a_x,a_y,0]`); the split-module versions regressed to constant-velocity/`f_x=0`. Port that accel term, then add the bias/gate/ZUPT/adaptive-R design below.
  - **Pose-control reference:** `run_one.jl:asmc_torques` (lines ~397–536) — the degree-2 position sliding law (global position error → body-frame rotation → sliding surface + equivalent control). Port its *structure*, substituting the **estimated** pose for the true pose it reads.
  - **PosRef interface:** `Profiles.PosRef` (`xo,yo,psi, Vxo,Vyo,om, Axo,Ayo,al`) via `current_posref()`/`active_ref()`/`is_velref()`; `global_to_local_frame` for desired-velocity rotation.
- **Static data:** `StaticArrays.jl` for all vectors/matrices (as in the current modules).
- **Measurement available now:** `SensorMod.simulate_measurement` returns `(θ, ω, a_x, a_y, g_z)` — accel and gyro are already on the bus (`bus.y_last`), so no sensor change is needed.
- **Explicit exclusions:** no plant/mixer/motor changes; no new estimator *type* (enhance the two existing structs); no PINN/ML; no importing a slip/friction model into the estimator (keep it model-free — see §9).

---

## 4. Component Breakdown

### `KalmanEstimator` (rewrite) / `estimator_update!(::KalmanEstimator)`
- **Type:** mutable `Base.@kwdef struct` + dispatched `function`
- **Responsibility:** accel-fused EKF that estimates body velocity + dead-reckoned pose, trusting the IMU for translation and the wheels only when gripping, and exposing its uncertainty.
- **State:** `x̂ = [V̂x, V̂y, ψ̂̇, ψ̂, X̂o, Ŷo]` (unchanged) + internal **accel-bias** estimate `[b̂x, b̂y]`.
- **Prediction (accel-driven):** velocity propagated by the body-frame kinematic reconstruction from measured proper acceleration and gyro (`V̇x = a_x + ψ̇·V̂y − b̂x`, `V̇y = a_y − ψ̇·V̂x − b̂y`); pose dead-reckoned through `R(ψ̂)`.
- **Update:** measurement `z = [wheel-velocity pseudo-measurement (Hω\y.ω), gyro yaw rate y.g_z]`; **3-D innovation** ⇒ `Rn` is **3×3** (fixes the current 4×4 bug).
- **Slip-adaptive weighting:** the wheel-channel entries of `R` are **inflated** when `slip_detect` (below) exceeds a threshold, so slip-corrupted wheel velocity is de-weighted; accel-bias is only corrected while gripping.
- **Outputs:** writes `bus.xhat`; keeps `P` (covariance) queryable as a **confidence** signal; sets `bus.d_hat = 0`.
- **Key constructor params (all tunable kwargs):** `Qn` (3), `Rn_base` (3), `bias_Qn` (2), `P0_scale`, `R_inflate` (slip-`R` multiplier), `slip_thresh`, `rate_hz`, `use_dhat`.
- **Depends on:** `_wheel_jacobian`, `slip_detect`, `SensorMod` measurement.

### `SMOEstimator` (rewrite) / `estimator_update!(::SMOEstimator)`
- **Type:** mutable `Base.@kwdef struct` + dispatched `function`
- **Responsibility:** accel-driven sliding-mode observer whose gated sliding term tracks the wheel velocity while gripping and coasts on the accelerometer during slip, with the equivalent-injection integral reinterpreted as the **slip estimate**.
- **Prediction (accel-driven):** same body-frame accel reconstruction as the KF (replaces the current `f_x = 0` placeholder), minus accel-bias.
- **Correction:** smoothed sliding term toward the wheel-velocity pseudo-measurement, **gated** by `1 − slip_gate(slip_detect)` so it is suppressed during slip; gyro yaw rate tracked directly.
- **`d̂` / slip:** the integral term (`ζ̇ = K·switch`) accumulates the wheel↔accel disagreement ⇒ `bus.d_hat` is the slip/lumped-disturbance estimate; fed forward when `use_dhat`.
- **ZUPT re-anchor:** when grip is detected (`slip_detect` low, wheel accel low), re-open the wheel correction to re-pin velocity and bias (bounds long-slip drift).
- **Outputs:** `bus.xhat`, `bus.d_hat`.
- **Key constructor params (tunable):** `L` (3), `K` (3), `δ`, `slip_gate_thresh`, `zupt_thresh`, `bias_gain` (2), `rate_hz`, `use_dhat`.
- **Depends on:** `_wheel_jacobian`, `slip_detect`.

### `slip_detect`
- **Type:** `function` (runtime, **measured-signals only**)
- **Responsibility:** produce the estimator's own slip indicator from measurements — the discrepancy between the wheel-derived velocity and the accel-integrated (predicted) velocity — driving KF `R`-inflation, the SMO gate, and ZUPT.
- **Inputs:** `y::NamedTuple`, `x̂_pred::SVector{3}` (accel-propagated velocity), `params`.
- **Outputs:** scalar slip magnitude (measured).
- **Critical distinction:** this is **not** the tuner's ground-truth `slip_indicator` (§ tuner brief), which uses `ω_true`/`v_true`. `slip_detect` may use **only** `y` and the estimate — never truth.

### `PoseFixModel` / `sample_pose_fix` (exteroceptive absolute-pose input — PosRef only)
- **Type:** `Base.@kwdef struct` (config) + `function`
- **Responsibility:** supply a **noisy absolute pose measurement** `(x, y, ψ)` at a low rate that bounds the otherwise-drifting dead-reckoned pose. Models a deployable indoor localization sensor (LiDAR-AMCL primary / fiducial secondary), **two-tier** for hospital micrologistics: a degraded/intermittent **transit** fix and a precise **docking** fix. Enabled **only in `:pose` runs** and behind `use_pose_fix`; VelRef runs stay proprioceptive-only.
- **How the "measurement" is formed (sim):** true pose `(u[17],u[18],u[4])` + Gaussian noise `(σ_pos,σ_pos,σ_ψ)` sampled at `fix_rate_hz`, with **dropout** (`dropout_frac` — no fix that tick) and rare **outliers** (`outlier_frac` — large jump); optional fixed `latency_ms`. Ground truth is used only to *generate* the fix, never exposed as truth.
- **Tier presets:** `:transit` (≈5–10 Hz, σ_pos≈0.05 m, σ_ψ≈1–2°, dropout 0.1–0.3, outlier ≈0.01) and `:docking` (≈20–30 Hz, σ_pos≈0.01–0.02 m, σ_ψ≈0.5–1°, reliable). Selectable per run/segment.
- **How it enters the estimator:** a **low-rate correction on the pose states** `(X̂o,Ŷo,ψ̂)` with `R_ext = diag(σ_pos², σ_pos², σ_ψ²)` — a standard KF measurement update (or SMO re-anchor) on the pose block only; velocity states untouched. Between fixes the estimator dead-reckons at 1 kHz as before. Apply innovation gating (χ²/threshold) so outlier jumps are rejected.
- **Key params (tunable/config):** `use_pose_fix`, `tier` (`:transit|:docking`), `fix_rate_hz`, `sigma_pos`, `sigma_psi`, `dropout_frac`, `outlier_frac`, `latency_ms`, `gate_thresh`, `seed`.
- **Depends on:** the estimator pose block; a dedicated low-rate `PeriodicCallback` (or a counter in the sensor callback) at `fix_rate_hz`.

### `asmc_wrench!` — `:pose` branch (new mode)
- **Type:** extend the existing `function` in `hybrid_ctrl/controllers.jl`
- **Responsibility:** pose-tracking task-space wrench, ported from `run_one.jl:asmc_torques`, consuming the **estimated** pose so PosRef closes the loop on `x̂` (drift is load-bearing).
- **Inputs:** `bus`, `xhat` (uses `[V̂x,V̂y,ψ̂̇,ψ̂,X̂o,Ŷo]`), `ref` (a `PosRef`), `params`, `asmc::ASMCController`, `dt`; `mode=:pose`.
- **Behavior:** global position error `e = (X̂o,Ŷo) − (ref.xo,ref.yo)` rotated into body frame; heading error via smooth wrap; desired body velocities via `global_to_local_frame`; sliding surfaces on position+velocity error; equivalent-control + switching wrench; same adaptive-gain update and `d̂` feedforward slot as the velocity branch. Returns `SVector{3}` wrench (same mixer contract).
- **Outputs:** `bus.W_asmc`.
- **Depends on:** `PosRef`, `global_to_local_frame`, existing `get_dynamic_lambda`/`smooth_bound`.
- **Note:** replaces the current `error("asmc_wrench! :pose mode not implemented")` stub. MPC/PID pose modes are **out of scope** here (pose runs use ASMC as the nominal controller).

### `run_hybrid` ref-dispatch (scheduler)
- **Type:** small extension to the existing `function`
- **Responsibility:** select `cfg.tracking = :velocity | :pose` per run, build the correct `ref` (`VelRef` vs `PosRef`), publish it so `current_posref()`/`active_ref()` resolve, and pass `mode=cfg.tracking` into the ASMC callback (already threaded).
- **Depends on:** `Profiles.pick_and_build`, `publish!`.

### `finalize_probe_log` / `build_estimator_log` (scheduler)
- **Type:** `function`s finalizing the existing `ESTIMATOR_PROBE_LOG` scaffold
- **Responsibility:** the sensor callback already pushes `(t, xhat, d_hat, u)` per estimator tick keyed by `objectid(bus)`; convert that raw buffer into the structured, time-aligned record the tuner consumes.
- **Extract per tick:** `t`; `v_true = u[1:3]`; `pose_true = (u[17], u[18], u[4])`; `v_hat = xhat[1:3]`; `pose_hat = (xhat[5], xhat[6], xhat[4])`; `d_hat`; and (optionally precomputed here or offline by the harness) the ground-truth slip from `u`.
- **Outputs:** an `EstimatorLog`-shaped object (matches the tuner brief's fields), returned via `get_last_probe_log`/a builder; cleared per run (`clear_probe_log!` already exists).
- **Decimation:** support logging every N-th tick (the buffer is 1 kHz × full state; parallel runs must not blow the commit limit) — a documented knob.

### `_wheel_jacobian` (reuse, unchanged)
- The O-config wheel map `Hω`; used by both estimators for the pseudo-measurement and by `slip_detect`.

---

## 5. File & Directory Structure

```
code_insights/
├── hybrid_ctrl/
│   ├── estimators.jl     # REWRITE: accel-fused KF+SMO, Rn fix, bias states,
│   │                     #          slip-adaptive R / gate + ZUPT, slip_detect,
│   │                     #          PoseFixModel + sample_pose_fix (exteroceptive
│   │                     #          pose correction, :pose only), all knobs as kwargs
│   ├── controllers.jl    # EXTEND: asmc_wrench! :pose branch (port asmc_torques)
│   └── scheduler.jl      # FINALIZE: structured probe log + ref-dispatch
│                         #           (ESTIMATOR_PROBE_LOG scaffold already present)
├── run_one.jl            # REFERENCE ONLY (asmc_torques pose law) — not modified
└── Mecanum_Hybrid_..._sonnet5.ipynb  # REFERENCE ONLY (accel-fusion) — not modified
```

No new files; three edited modules. The tuner (`instructions/estimator-kf-smo-tuner.md`) consumes these unchanged.

---

## 6. Key Interfaces

Signatures + docstrings only. Bodies are always `# ...`.

```julia
"""
    estimator_update!(bus, y, est::KalmanEstimator, params, dt)

Accel-fused EKF tick. Predict velocity from measured proper acceleration + gyro
(minus accel-bias), dead-reckon pose; update with the wheel-velocity pseudo-
measurement + gyro (3-D innovation ⇒ Rn 3×3), inflating wheel-R when slip_detect
fires. Writes bus.xhat; keeps P as a confidence signal.
  y :: NamedTuple  (θ, ω, a_x, a_y, g_z)   -- accel/gyro now USED
"""
function estimator_update!(bus, y, est::KalmanEstimator, params, dt)
    # ...
end

"""
    estimator_update!(bus, y, est::SMOEstimator, params, dt)

Accel-driven SMO tick. Predict from accel (minus bias); slide toward the wheel-
velocity pseudo-measurement through a slip-GATED correction; integral term ζ →
bus.d_hat is the slip/disturbance estimate; ZUPT re-anchor on detected grip.
"""
function estimator_update!(bus, y, est::SMOEstimator, params, dt)
    # ...
end

"""
    slip_detect(y, x_pred, params) -> Float64

Runtime slip indicator from MEASURED signals only: ‖ Hω\\y.ω − x_pred ‖ (wheel-
derived vs accel-predicted body velocity). Drives KF R-inflation, SMO gate, ZUPT.
Must NOT use ground truth.
"""
function slip_detect(y, x_pred, params)
    # ...
end

"""
    sample_pose_fix(u, fix::PoseFixModel, t) -> Union{SVector{3},Nothing}

Exteroceptive absolute-pose "measurement" (x,y,ψ) for :pose runs: true pose +
Gaussian noise at fix_rate_hz, with dropout (→ Nothing this tick) and rare
outlier jumps. Ground truth used only to GENERATE the fix. The estimator applies
it as a gated low-rate correction on the pose block (R_ext=diag(σ_pos²,σ_pos²,σ_ψ²)).
"""
function sample_pose_fix(u, fix::PoseFixModel, t)
    # ...
end

"""
    asmc_wrench!(bus, xhat, ref, params, asmc, dt; mode=:velocity) -> SVector{3}

Adaptive-SMC task-space wrench. `:velocity` unchanged. `:pose` (new) ports
run_one.jl:asmc_torques: global position error on the ESTIMATED pose
(X̂o,Ŷo,ψ̂) → body frame → sliding surface + equivalent control, same adaptive-
gain + d̂ slot. `ref::PosRef` in :pose mode.
"""
function asmc_wrench!(bus, xhat, ref, params, asmc, dt; mode=:velocity)
    # ...
end

"""
    build_estimator_log(bus) -> NamedTuple

Convert the per-tick ESTIMATOR_PROBE_LOG buffer for `bus` into aligned series:
t, v_true(=u[1:3]), v_hat(=xhat[1:3]), pose_true(=u[17,18,4]),
pose_hat(=xhat[5,6,4]), d_hat, slip. Supports tick decimation.
"""
function build_estimator_log(bus)
    # ...
end
```

---

## 7. Data Flow

Per estimator tick (1 kHz sensor callback, unchanged cadence):
1. `simulate_measurement` → `y = (θ, ω, a_x, a_y, g_z)` on `bus.y_last`.
2. **Predict:** accel + gyro → `x̂_pred` velocity (minus accel-bias); dead-reckon pose.
3. `slip_detect(y, x̂_pred, params)` → measured slip magnitude.
4. **Update (KF):** wheel-velocity + gyro innovation, with wheel-`R` inflated by slip; correct bias only while gripping. **(SMO):** gated sliding correction toward wheel velocity; integral → `bus.d_hat` (slip); ZUPT re-anchor on grip.
5. `bus.xhat` (+ `bus.d_hat`) written; `P` retained (KF confidence).
6. Probe callback pushes `(t, xhat, d_hat, u)`; `build_estimator_log` later assembles the aligned record.

Per pose-fix tick (`:pose` runs only, `use_pose_fix`, at `fix_rate_hz` ≪ 1 kHz):
1. `sample_pose_fix` → a noisy `(x,y,ψ)` fix (or `Nothing` on dropout).
2. If present and it passes innovation gating: **correct the pose block** `(X̂o,Ŷo,ψ̂)` with `R_ext` (velocity states untouched). This bounds the otherwise-unbounded pose drift. Between fixes, pure dead reckoning.

Per control tick (PosRef runs):
1. `asmc_wrench!(…, ref::PosRef, …; mode=:pose)` reads **estimated** pose `x̂[4,5,6]`, forms global→body position error, sliding surface, wrench → `bus.W_asmc` → mixer (unchanged).

**Estimation ≠ truth (enforce):** estimators read only `y` + bus; `slip_detect` uses only `y`/estimate; the pose fix uses true pose **only to synthesize the noisy measurement** (as a real sensor would output), never as a clean truth input; unmasked ground truth appears solely in `build_estimator_log`/the tuner objective.

---

## 8. Implementation Sequence

1. **`Rn` 3×3 fix + accel prediction (KF), `f_x`→accel (SMO)** — restore the sonnet5 accel-fusion first; verify velocity estimate tracks truth in-grip (high μ) before adding slip logic.
2. **Accel-bias states (both)** — verify bias converges on a constant-velocity segment and removes accel drift.
3. **`slip_detect`** — measured-only indicator; sanity-check it rises during known slip (low μ / hard accel) and stays low in grip.
4. **Slip-adaptive `R` (KF) + gate/ZUPT (SMO)** — the payoff; verify in-slip velocity error drops vs step 1.
5. **Expose all knobs as kwargs** — every threshold/gain a `@kwdef` field so the tuner's `apply_params!` can set them.
6. **`asmc_wrench!` `:pose`** — port `asmc_torques` on estimated pose; verify an ellipse run closes the loop and drifts gracefully (no divergence).
7. **`run_hybrid` ref-dispatch** — `:velocity`/`:pose` selection + `PosRef` publish.
8. **`PoseFixModel` + pose-correction** — low-rate exteroceptive fix on the pose block (`:pose` + `use_pose_fix`); verify pose is bounded with the fix and unbounded without, and that outliers are gated. Add the transit/docking tier presets and dropout.
9. **`build_estimator_log` + decimation** — finalize the probe into the tuner's record; verify time alignment and pose extraction (log the fix samples + a `fix_available` flag too, for the dropout experiment).

---

## 9. Numerical / Estimator / Integration Considerations

- **Two distinct slip signals — keep them separate:** `slip_detect` (measured-only, drives the estimator) vs the tuner's ground-truth `slip_indicator` (evaluation/weighting). Never let truth leak into `slip_detect`.
- **Model-free estimator:** do **not** import a LuGre/friction model to correct the wheel map — that circularity would defeat the "physical mismatch as disturbance" framing and collide with the PINN. Slip is handled by accel + gate + ZUPT, and *falls out* as the SMO `d̂` / KF innovation.
- **Observability limit + how the fix resolves it:** DC velocity under *sustained* slip is unobservable from IMU+encoder (both estimators drift there, bounded only by transient slip + ZUPT). Pose is unobservable from proprioception alone — so **without** `use_pose_fix` (VelRef, or PosRef in the fallback), pose dead-reckons and scoring uses drift *rate*; **with** `use_pose_fix` (PosRef), the low-rate exteroceptive correction bounds pose, and pose RMSE becomes a fair metric between fixes. The pose controller must still tolerate the sawtooth (drift-then-snap) that intermittent fixes produce.
- **Accel bias vs gravity/Coriolis:** the accel reconstruction needs the Coriolis term (`ψ̇×V̂`) and a bias state; without bias estimation the prediction drifts. Bias is observable only while gripping — tie its correction to the ungated (grip) condition.
- **Chatter into control:** the SMO switch feeds `x̂` which feeds controllers — keep the smoothed switching (`δ` boundary layer) and expose `δ` for the tuner's chatter penalty.
- **Callback ordering unchanged:** estimator runs before controllers in the `CallbackSet` (sensor callback first) so controllers see the fresh `x̂` — preserve this when editing the scheduler.
- **Probe-log memory:** 1 kHz × full-state copy × parallel tuner runs can hit the Windows commit limit — decimate the probe (log every N-th tick) and/or store only the needed fields; make N a knob.
- **State I/O unchanged:** do not alter the plant state layout or `abstol`; the estimator reads `u[1:3]`/`u[17,18,4]` for logging only.
- **Reproducibility:** identical `sensor_seed` ⇒ identical estimator trace; thread the seed as before.

## 10. Success Criteria

- [ ] `KalmanEstimator` innovation is 3-D and `Rn` is 3×3 — no dimension-mismatch crash (current bug gone).
- [ ] Both estimators consume `y.a_x, y.a_y` (accel appears in the prediction) — verified by inspection and by drift behavior when the wheel channel is starved.
- [ ] In-grip (high μ): velocity NRMSE vs truth is low and comparable to the sonnet5 reference.
- [ ] In-slip (low μ / high accel): accel-fused + adaptive estimator beats the constant-velocity/no-gate baseline on velocity NRMSE (the design's premise).
- [ ] `slip_detect` uses only measurements (no truth); rises in slip, low in grip.
- [ ] SMO `d̂` tracks the ground-truth slip in sign/timing (validated against `slip_indicator`, not fed it).
- [ ] `asmc_wrench!(:pose)` closes the loop on an ellipse run using estimated pose; bounded drift, no divergence.
- [ ] With `use_pose_fix` on a `:pose` run, pose drift is **bounded** (stays within the fix noise band between corrections); with it off, pose drift is unbounded — the difference is visible and the fix is gated against injected outliers.
- [ ] `run_hybrid` runs both `:velocity` and `:pose`; `build_estimator_log` returns aligned `v_true/v_hat/pose_true/pose_hat/d_hat/slip` at the (decimated) estimator rate.
- [ ] All estimator hyperparameters are settable as kwargs (tuner `apply_params!` compatibility).
- [ ] Reproducible: identical seed ⇒ identical log.

## 11. Out of Scope

- The estimator **tuner** itself (`instructions/estimator-kf-smo-tuner.md`) — this brief only unblocks it.
- **MPC/PID pose modes** — deferred to the controller-phase brief (needed for the full 3-controller PosRef comparison: PID→outer position/cascade loop, MPC→position-augmented model). This brief wires only ASMC `:pose` (the fixed nominal controller for estimator tuning).
- **The fix-dropout robustness experiment and the docking-vs-transit comparison protocol** — specified in the session summary (§6) and consumed downstream; this brief only provides the `PoseFixModel` mechanism (tiers, dropout, outliers) they exercise.
- Any **plant / mixer / motor / sensor** change; the state layout and `simulate_measurement` are untouched.
- Importing a **friction/slip model** into the estimator (kept model-free by decision).
- Controller-gain tuning, the control-law comparison, PINN/GPU/hardware.
