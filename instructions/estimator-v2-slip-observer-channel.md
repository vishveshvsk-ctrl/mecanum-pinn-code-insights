# Estimator v2 — Slip-Observer Channel (SMO / ESO slip observer feeding ESKF fusion)

> **Generated:** 2026-08-03 (supersedes `estimator-v2-smo-slip-channel.md`, which covered
> only the SMO variant and had a flawed slip readout — see §2.1 note)
> **Stack:** Julia 1.x, StaticArrays, LinearAlgebra, existing `hybrid_ctrl_v2` modules
> **Scope:** Estimator architecture — additive slip-observation channel, replay evaluation only
> **Companion briefs:** `sensors-suite-consolidation-and-physical-noise.md`,
> `estimator-v2-ensemble-replay-tuning.md`, `estimator-v2-iae-adaptive.md`
> **Origin:** user hypothesis (2026-08-03 session) — first-order error accumulation in the
> Gauss–Markov slip model is where `ESKFEstimatorV2` underperforms during sustained slip;
> a model-free finite-/fixed-time slip observer should track slip better than the
> parametric mean-reverting prior, while the ESKF keeps its sensor-fusion/covariance role.

---

## 0. Preservation constraint — read this first

These files are **FROZEN** (regression gate of
`instructions/estimator-v2-ensemble-replay-tuning.md` §0). **Do not edit any of them:**

```
hybrid_ctrl_v2/scheduler_v2.jl                     run_hybrid_v2 — full ODE + callbacks
hybrid_ctrl_v2/estimators_v2.jl                    ESKFEstimatorV2
hybrid_ctrl_v2/sensors_v2.jl                       SensorModV2, build_suite
hybrid_ctrl_v2/estimator_tuning/harness_v2.jl      run_and_log_v2 AND run_and_log_replay_v2
hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl  eskf_param_space_v2, apply_params_v2!
hybrid_ctrl_v2/estimator_tuning/objective_v2.jl    make_replay_objective_v2
hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl  the 11-trajectory manifest
hybrid_ctrl_v2/estimator_tuning/run_estimator_replay.jl
hybrid_ctrl_v2/runs_estimator_v2_replay/**         the 5-seed baseline results (read-only reference)
hybrid_ctrl/**                                      all v1
```

Everything in this brief is **new files only**, extending v1/v2 generics by multiple
dispatch — the exact pattern `estimators_v2.jl` documents in its header
(`estimators_v2.jl:5-17`): v1 is never edited, the new estimator is a NEW struct with its
own `estimator_update!`/`apply_pose_fix!` methods added to the ORIGINAL generic
functions, reusing `Main.EstimatorMod._wheel_jacobian`/`slip_detect`/`_wrap_angle` and
`Main.EstimatorModV2.gauss_markov_q`/`derive_process_noise` unchanged.

**The existing replay pipeline must remain runnable and produce identical results after
this work lands.** §8 makes that a gate.

---

## 1. Motivation and hypothesis

`ESKFEstimatorV2` models slip `(sx, sy)` as a mean-reverting Gauss–Markov (OU) process:
decay `A[10,10] = A[11,11] = −dt/τ_slip` with `τ_slip = 0.06 s`, `σ_slip = 0.014 m/s`,
both *measured* from one spin_creep trace (`estimators_v2.jl:127-143`). This is correct
for grip-recovery transients but structurally wrong during **sustained slip**: the prior
pulls ŝ toward zero on a 60 ms timescale regardless of how long the platform has been
sliding, and the single-point τ/σ measurement is trajectory-dependent by the struct's own
admission. The correction must then come entirely from the wheel+gyro update, whose H
(`estimators_v2.jl:291-293`) measures `v + s` jointly — so slip error and velocity error
trade against each other through one covariance, and first-order propagation error
accumulates exactly in the regime (long slip excursions, continuous rotation) where the
PCRLB brief found the current filter degrading.

**Hypothesis to test:** a model-free observer driven by the wheel-vs-IMU disagreement
estimates slip with finite-time (super-twisting) or exponential (ESO) convergence and no
parametric slip model; feeding that estimate into the ESKF as a direct 2-D slip
measurement improves in-slip velocity RMSE and slip RMSE without degrading grip-phase
metrics. The ESKF remains the fusion backbone and keeps full covariance discipline over
how much to trust the observer (NIS gate + R_s). Two observer variants are tested on the
SAME channel so the experiment isolates the observer core:

- **SMO (super-twisting):** finite-time, exact rejection of the matched slip ramp within
  gain bounds, continuous injection (no chattering into the ESKF innovation).
- **ESO (linear extended-state / GPIO):** exponential convergence, one bandwidth knob —
  the SMO's linear cousin; if it matches the SMO, prefer it (simpler, fewer knobs, no
  switch ripple). This is the deterministic disturbance-observer view the plant already
  uses for yaw (super-twisting DOB), and the frequency-domain Q-filter DOB is its
  equivalent — do not implement the Q-filter form separately.

**Explicitly NOT proposed:** removing the GM slip model, removing the flow channel,
touching the yaw/gyro-bias channel, NN-based slip prediction (deferred by user), or
closed-loop integration (replay only).

---

## 2. Architecture pattern

**Observer-agnostic slip channel.** The observer consumes IMU + wheel odometry and
produces a 2-D slip estimate ŝ; the ESKF consumes ŝ as a pseudo-measurement of its slip
states. The channel (H_s, R_s, NIS gate, Joseph update, placement in the tick) is
identical for both observer variants — only the observer core differs.

```
per 1 kHz tick:
  IMU (ax, ay), encoders ω, ESKF state from previous tick (b̂, v3)
        │
        ├──► open-loop IMU velocity integration:  v̂_imu⁺ = v̂_imu + dt·(a − b̂ + coriolis)
        ├──► measured discrepancy:                m = z_w − v̂_imu     (2-D, ≈ slip + slow drift)
        ├──► observer core (variant :smo or :eso) tracks m → ŝ
        │
        └──► ESKF tick:
               1. nominal propagation (unchanged)
               2. covariance propagation (unchanged; GM slip prior stays)
               3. NEW: slip pseudo-measurement update  z_s = ŝ
                  (H selects cols 10,11; Joseph; NIS-gated)
               4. wheel+gyro update   (unchanged, now sees slip-corrected x and P)
     flow update (unchanged, own callback/rate)
     pose fix    (unchanged, own callback/rate)
```

### 2.1 Observer core — shared structure, two variants

**Why the discrepancy signal m is the right object** (this corrects the superseded SMO
brief, which defined `e = z_w − v̂` with the injection driving `e → 0` and read the slip
as `z_w − v̂` — that readout tends to zero precisely when the observer converges; the
slip information lives in the *injection the observer must apply*, not in the residual).
The correct construction: keep an **open-loop** IMU velocity integration

```
v̂_imu⁺ = v̂_imu + dt·( a − b̂ + [ v3·v̂_imu_y ; −v3·v̂_imu_x ] )
```

(`b̂ = est.x[8:9]`, `v3 = est.x[3]` from the ESKF's previous tick; Coriolis terms as in
`estimators_v2.jl:235-236`). Then the measured discrepancy

```
m = z_w − v̂_imu  =  s  +  δv_bias  +  n
```

contains the slip `s`, a SLOW residual drift `δv_bias` (from accel-bias estimation error
and IMU noise integration — near-DC), and wheel/IMU noise `n`. The observer's job is to
track `m` while rejecting `n`. The near-DC `δv_bias` component is acceptable — the
ESKF's accel-bias states observe the same discrepancy through the wheel update and
partition the slow part; `ρ_smo` bounds how much the filter trusts ŝ overall. Document
this coupling in the code; it is the known limitation of any proprioceptive slip
observer.

Define the observer error `e = m − ŝ` (NOT `z_w − v̂`). Both variants track `m`:

**Variant `:smo`** (super-twisting, 2-D per-axis):

```
ŝ⁺ = ŝ + dt·( k1·Φ(e) + w )
w⁺ = w + dt·k2·Φ(e)
Φ(e) = e ./ sqrt.(e.^2 .+ δ^2)      (smooth unit switch, per-axis; repo rule —
                                     cf. v1 `_smoothswitch`, estimators.jl:407,
                                     and the plant's super-twisting DOB)
```

Finite-time convergence to `m` when |ṁ| < k2; the Φ boundary layer and the integrator
provide the noise filtering. Defaults scaled to the measured slip magnitude
(σ_slip = 0.014 m/s, τ_slip = 0.06 s — provenance comment required):
`k1 = 0.1`, `k2 = 5.0`, `δ = 1e-2` (k2 must exceed worst-case |ṡ|; with slip swings of
~0.05 m/s over ~10 ms, |ṡ| ~ 5 m/s² — verify against a spin_creep trace and adjust the
default comment to the measured number).

**Variant `:eso`** (2nd-order linear ESO / GPIO, 2-D per-axis):

```
ŝ⁺ = ŝ + dt·( d̂ + β1·e )
d̂⁺ = d̂ + dt·β2·e
β1 = 2·ω_o,  β2 = ω_o²              (both poles at ω_o — one bandwidth knob)
```

`d̂` estimates ṁ; ŝ tracks `m` exponentially with bandwidth ω_o. Default
`ω_o = 30 rad/s` (~5 ms time constant — comfortably faster than τ_slip, slower than the
1 kHz tick; the `omega_o_psi = 6π` plant-DOB precedent sets the scale). d̂ is extra
observer state; do not confuse it with `bus.d_hat`.

**Reset behavior (both variants):** `v̂_imu` is re-anchored to the ESKF's velocity
estimate at every **flow update acceptance** and **pose fix acceptance** — the moments
when the ESKF has slip-immune velocity information (`v̂_imu ← (x[1], x[2])`). This bounds
the open-loop integration drift between anchors and is the only coupling from the fusion
side back into the observer. Implement as `reanchor_imu_velocity!(est)` called from the
`apply_flow_sobs!`/`apply_pose_fix!` methods on acceptance (not on gate rejection).

### 2.2 How ŝ enters the ESKF

New 2-D measurement update inside `estimator_update!`, placed **immediately before the
wheel+gyro update** (after covariance propagation), so the wheel update's slip detection,
R-inflation, and gain computation all see an already slip-corrected state and covariance:

```
z_s = ŝ
H_s = 2×12, H_s[1,10] = 1, H_s[2,11] = 1, else 0
R_s = ρ_s²·I₂,   ρ_s default = 3·σ_slip ≈ 0.042 m/s   (tunable, log-scale)
NIS gate: ν = e′S⁻¹e > 13.82 (χ²₂,0.999) ⇒ skip update
Joseph form + PSD symmetrization, same pattern as the wheel update
```

The GM slip model stays in A and Q **unchanged** — it is now the prior between
corrections and still correct for grip-recovery transients; the observer channel supplies
the correction the GM dynamics cannot. Do not zero `q_slip`, do not remove the decay term.

Structural parallel to `apply_flow!` (`estimators_v2.jl:356-381`): flow measures `Vx, Vy`
with **zeros in the slip columns**; the observer channel measures `sx, sy` **directly** —
complementary, not redundant.

---

## 3. File 1 — `hybrid_ctrl_v2/estimators_v2_slipobs.jl` (NEW, ~330 lines)

Module `EstimatorModV2SlipObs`. Header comment must document the include contract:

```
Must be `include`d AFTER hybrid_ctrl_v2/estimators_v2.jl (needs EstimatorModV2)
and after tune_controller.jl / hybrid_ctrl/estimators.jl (extends
Main.EstimatorMod.estimator_update! / apply_pose_fix!).
```

Export: `ESKFSlipObsEstimatorV2`, `init_eskf_slipobs_v2!`, `apply_flow_sobs!`.

### 3.1 Struct

`Base.@kwdef mutable struct ESKFSlipObsEstimatorV2` — **field-identical copy of
`ESKFEstimatorV2`'s fields** (`estimators_v2.jl:121-182`) with the same defaults and the
same τ_slip/σ_slip measurement-provenance comment (copy it — provenance must travel with
the numbers), **plus**:

```julia
    # --- slip-observer config (brief §2.1) ---
    observer_kind::Symbol = :smo    # :smo | :eso  (channel identical; only the core differs)
    smo_k1::Float64     = 0.1       # super-twist injection gain  (:smo only)
    smo_k2::Float64     = 5.0       # super-twist integral gain, > max|ṡ|  (:smo only)
    smo_delta::Float64  = 1e-2      # boundary layer of Φ — PINNED, not searched
    eso_omega_o::Float64 = 30.0     # ESO bandwidth [rad/s]  (:eso only)
    rho_s::Float64      = 0.042     # slip pseudo-measurement std [m/s] — default 3*sigma_slip
    use_slipobs::Bool   = true      # ablation: false ⇒ identical to ESKFEstimatorV2

    # --- observer internal state ---
    obs_v_imu::MVector{2,Float64} = MVector{2}(0.0, 0.0)  # open-loop IMU velocity
    obs_s::MVector{2,Float64}     = MVector{2}(0.0, 0.0)  # ŝ
    obs_w::MVector{2,Float64}     = MVector{2}(0.0, 0.0)  # :smo integrator / :eso d̂
```

### 3.2 Functions

- `_smooth_unit(e::SVector{2}, δ) = e ./ sqrt.(e.^2 .+ δ^2)`.
- `init_eskf_slipobs_v2!(est, params)` — mirror `init_eskf_v2!`
  (`estimators_v2.jl:187-214`) exactly (same P0 assembly, same pinned `slip_threshold`,
  same τ_slip assertion), then zero the three observer state vectors.
- `slipobs_tick!(est, z_w::SVector{2}, ax, ay, v3, dt) -> SVector{2,Float64}` —
  implements §2.1: integrate `obs_v_imu`, form `m`, dispatch on `est.observer_kind`
  (`:smo`/`:eso`; error on anything else), return updated `obs_s`.
- `reanchor_imu_velocity!(est)` — `obs_v_imu .= (est.x[1], est.x[2])`.
- `Main.EstimatorMod.estimator_update!(bus, y, est::ESKFSlipObsEstimatorV2, params, dt)` —
  duplicate the `ESKFEstimatorV2` tick body (`estimators_v2.jl:225-338`) with exactly two
  insertions:
  1. after nominal propagation and the `z`/`slip_meas` computation (v2 lines 244-247):
     `if est.use_slipobs` → `ŝ = slipobs_tick!(est, z[1:2], ax, ay, x[3], dt)`;
  2. immediately before the wheel+gyro measurement update (before v2 line 304):
     `if est.use_slipobs` → the §2.2 slip pseudo-measurement update (2×12 H_s,
     R_s = ρ_s²·I, NIS gate at 13.82, Joseph form, symmetrize, `x .+= dx`,
     `_renorm_heading!(x)`).
  When `use_slipobs == false` the method must execute **exactly** the v2 sequence,
  bit-for-bit (the §8 regression gate — no reordering, no extra flops in the ablation
  path).
- `apply_flow_sobs!(bus, est::ESKFSlipObsEstimatorV2, m, z_flow, params)` and
  `Main.EstimatorMod.apply_pose_fix!(bus, est::ESKFSlipObsEstimatorV2, fix, z_fix)` —
  bodies identical to v2's (`estimators_v2.jl:356-416`), PLUS: on acceptance (return
  true path only), call `reanchor_imu_velocity!(est)`. (Separate name for flow because
  v2's `apply_flow!` is a free function in `EstimatorModV2`, not a v1 generic.)

`bus.xhat`/`bus.d_hat` writes identical to v2 (`estimators_v2.jl:335-336`).

---

## 4. File 2 — `hybrid_ctrl_v2/estimator_tuning/harness_v2_slipobs.jl` (NEW, ~120 lines)

Module `HarnessV2SlipObsMod`. Include-after: `harness_v2.jl` + `estimators_v2_slipobs.jl`.

- `build_estimator_v2_slipobs(est_cfg::NamedTuple, suite)` — same kwarg pass-through as
  `build_estimator_v2` (`harness_v2.jl:52-60`) for the 10 tuning params + pass-through of
  `:observer_kind, :smo_k1, :smo_k2, :smo_delta, :eso_omega_o, :rho_s, :use_slipobs,
  :tau_slip, :sigma_slip, :sigma_gyro_bias_rw, :use_dhat, :rate_hz`; constructs
  `Main.EstimatorModV2SlipObs.ESKFSlipObsEstimatorV2(; kw...)`.
- `run_and_log_replay_v2_slipobs(est_cfg, traj_entry, suite; seed=42, rate_hz=1000.0,
  data_dir="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov") -> EstimatorLogV2` —
  thin copy of `run_and_log_replay_v2` (`harness_v2.jl:226-305`) with two changes:
  (a) builds via `build_estimator_v2_slipobs`; (b) the flow call uses
  `Main.EstimatorModV2SlipObs.apply_flow_sobs!`. Reuse
  `Main.HarnessV2Mod._load_replay_data_v2`, `Main.HarnessV2Mod._interp_scalar_v2`,
  `Main.HarnessV2Mod.EstimatorLogV2`, and `Main.HarnessV2Mod.REPLAY_CACHE_V2` unchanged —
  cross-module internal access via `Main.`. (Verify the module name at `harness_v2.jl`'s
  `module` line and use that; do not guess.) The slip-log line uses `est.wheel_H`
  (`harness_v2.jl:298`) — the new struct keeps that field name, so the logging contract
  is unchanged. Return type stays `EstimatorLogV2` (its `d_hat_log` carries the ESKF's
  fused slip states — exactly what the comparison needs; no struct change).

---

## 5. File 3 — `hybrid_ctrl_v2/estimator_tuning/param_space_v2_slipobs.jl` (NEW, ~70 lines)

Module `ParamSpaceV2SlipObsMod`. Include-after: `param_space_v2.jl`.

- `eskf_slipobs_param_space_v2(kind::Symbol)`: the existing 10 dims **reused verbatim**
  from `ParamSpaceV2Mod.eskf_param_space_v2()` (call it and append; do not re-type the
  dims) plus variant-specific log-scale dims:

  | dim         | len | scale | bounds        | used by |
  |-------------|-----|-------|---------------|---------|
  | smo_k1      | 1   | log   | 0.01 .. 2.0   | :smo |
  | smo_k2      | 1   | log   | 0.5 .. 50.0   | :smo |
  | eso_omega_o | 1   | log   | 5.0 .. 200.0  | :eso |
  | rho_s       | 1   | log   | 1e-3 .. 0.2   | both |

  `:smo` space = 10 + {smo_k1, smo_k2, rho_s} (13 dims);
  `:eso` space = 10 + {eso_omega_o, rho_s} (12 dims).
  `ParamSpace(:eskf_slipobs_smo_v2, ...)` / `ParamSpace(:eskf_slipobs_eso_v2, ...)`.
  `smo_delta` stays pinned at 1e-2 in both.
- `apply_params_v2_slipobs!(theta, space)` — mirror `apply_params_v2!`
  (`param_space_v2.jl:86-111`), returning the 10 shared keys plus the variant's observer
  keys, `observer_kind`, `use_slipobs=true`, `use_dhat=false`.

---

## 6. File 4 — `hybrid_ctrl_v2/estimator_tuning/compare_slipobs_baseline.jl` + `.bat` (NEW)

**This is the first thing to run** — fixed defaults, no tuning; it answers "does the
observer channel help at all, and which core?" before any optimizer budget is spent.

For each trajectory in the frozen 11-trajectory manifest (`replay_trajset.jl:46-60` —
include it and iterate; do not re-type the list), with the seed's
`runs_estimator_v2_replay/seed<S>/best_config.json` supplying the 10 shared params:

1. Regression gate (once, first trajectory only): `ESKFSlipObsEstimatorV2` with
   `use_slipobs=false` vs `ESKFEstimatorV2`, same params, same suite, same seed; assert
   `max|Δv_hat| < 1e-12` and `max|Δpose_hat| < 1e-12`. Hard gate.
2. Baseline: `Main.HarnessV2Mod.run_and_log_replay_v2`.
3. SMO variant: `run_and_log_replay_v2_slipobs` with `observer_kind=:smo`, default knobs.
4. ESO variant: same with `observer_kind=:eso`, default knobs.
5. Score all three with the existing objective terms (`ObjectiveV2Mod` — read
   `objective_v2.jl` for the exact entry points) plus **slip RMSE per axis**: true signed
   slip is `Hw \ ω_true − v_true` (recompute signed per-axis from the cached replay data),
   estimated slip is `d_hat_log[1:2, :]`.

CLI: `--seed S` (default 1), `--out runs_estimator_v2_slipobs_replay/compare_seedS/`.
Output: `compare_table.csv` + stdout table — per trajectory × {baseline, smo, eso}:
velocity RMSE, in-slip velocity RMSE (same mask definition as `objective_v2.jl`), slip
RMSE (x, y), position RMSE, heading RMSE, objective total; manifest mean/median summary
rows.

Companion `compare_slipobs_baseline.bat` per AGENTS.md §6: `cd /d` to the Windows tree
root (`C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\`),
`julia --project=. hybrid_ctrl_v2/estimator_tuning/compare_slipobs_baseline.jl %*`.
Match the conventions of the existing `hybrid_ctrl_v2/run_stage_*.bat` files (read one
and copy its structure).

**Decision rule (state it in the script header):** the hypothesis holds if in-slip
velocity RMSE and slip RMSE improve on the high-slip entries (spin_creep, μ=0.3 cases)
while grip-phase metrics stay within the eval-noise floor (~5%, per the pivot handoff).
If either variant holds → §7 tuning for that variant (both if both hold). If neither →
the binding constraint is proprioceptive information content, not observer design —
archive with results, do not iterate blindly; that outcome feeds the PCRLB
estimator-selection decision.

---

## 7. File 5 — `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_slipobs.jl` + `.bat` (NEW)

Only used if §6 is positive. CLI driver mirroring `run_estimator_replay.jl` (read it and
copy its structure: `--seed`, `--p1-cap`, `--p2-cap`, `--smoke`, staged dxNES → BOBYQA via
the shared optimizer) with:

- `--observer smo|eso` selecting the param space from §5; objective = same terms as
  `make_replay_objective_v2` but calling `run_and_log_replay_v2_slipobs` (write a thin
  `make_replay_objective_v2_slipobs` — reuse `ObjectiveV2Mod`'s scoring helpers; do NOT
  edit `objective_v2.jl`).
- warm-start `--init-from runs_estimator_v2_replay/seedS/best_config.json`: the 10 shared
  dims start at the baseline optimum, the observer dims at defaults (statistically honest
  — the optimizer only has to learn the 2–3 new dims).
- output dir `hybrid_ctrl_v2/runs_estimator_v2_slipobs_replay/<observer>/seed<N>/` —
  **new directory; never write into `runs_estimator_v2_replay/`**.
- companion `run_estimator_replay_slipobs.bat` (same .bat conventions as §6).

---

## 8. Verification gates

1. **Static (no Julia in the authoring WSL session):** include order documented in every
   header; dispatch signatures match the v1 generics exactly (`estimator_update!(bus, y,
   est, params, dt)`, `apply_pose_fix!(bus, est, fix, z_fix)`); module paths verified
   against the actual `module` lines of `harness_v2.jl`, `objective_v2.jl`,
   `replay_trajset.jl` (read them; do not guess names).
2. **Regression (user-side, .bat):** §6 step 1 — `use_slipobs=false` reproduces
   `ESKFEstimatorV2` bit-for-bit on one trajectory (< 1e-12 max abs diff). Hard gate.
3. **Frozen-pipeline gate:** re-run `run_estimator_replay.jl --smoke` after all files
   land; results identical to before. `git diff` must show only new files.
4. **Hypothesis test:** §6 decision rule, per variant.

---

## 9. Execution order

1. `estimators_v2_slipobs.jl`
2. `harness_v2_slipobs.jl`, `param_space_v2_slipobs.jl`
3. `compare_slipobs_baseline.jl` + `.bat` → run (gate 2 + hypothesis test, both variants)
4. `run_estimator_replay_slipobs.jl` + `.bat` → run only for positive variants
5. Write `chat-handoff/estimator_v2_slipobs_channel_handoff.md` with the compare table,
   gate results, and tuning outcome (if run); commit new files + handoff. Do not commit
   anything under `runs_estimator_v2_slipobs_replay/` until runs complete (AGENTS.md: git
   is transport + post-run archive).

## 10. Out of scope

- Closed-loop (`run_hybrid_v2` / `scheduler_v2.jl`) integration — replay only.
- Yaw-channel observer (gyro bias is a well-modeled random walk).
- Removing/replacing the GM slip model or the flow channel.
- NN-based slip prediction (explicitly deferred by user).
- The IAE adaptive-Q workstream (`estimator-v2-iae-adaptive.md`) — orthogonal; a combined
  slip-observer + IAE variant is a later decision pending both compare results.
- The ensemble-tuning expansion (`estimator-v2-ensemble-replay-tuning.md`) — independent;
  if both land, the observer channel runs inside the ensemble objective unchanged.
- GPU anything.

## 11. Rejected alternatives (recorded so they are not re-litigated)

- **Cascade pre-correction** (subtract ŝ from wheel odometry, drop slip states from the
  ESKF): the filter loses P on slip — no NIS gating, no R-inflation semantics, and a
  wrong observer silently poisons the wheel channel. The v1 covariance-discipline failure
  mode in reverse.
- **Residual-driven observer with `ŝ = z_w − v̂` readout** (the superseded SMO-only
  brief): driving the wheel-residual to zero makes that readout vanish exactly when the
  observer converges; the slip information lives in the injection, not the residual. The
  §2.1 discrepancy-signal construction is the correct one.
- **Q-filter / frequency-domain DOB as a separate variant:** mathematically equivalent to
  the ESO — do not implement twice.
- **IMM re-add** for grip/slip mode switching: deferred behind the observer-channel and
  IAE results (the pivot handoff's re-add provision stands, but the cheap experiments go
  first).

---

## Key file:line references for the executor

| What | Where |
|---|---|
| ESKFEstimatorV2 struct + defaults + τ/σ provenance | `hybrid_ctrl_v2/estimators_v2.jl:121-182` |
| v2 tick body to duplicate | `hybrid_ctrl_v2/estimators_v2.jl:225-338` |
| flow update pattern (NIS 13.82, Joseph) | `hybrid_ctrl_v2/estimators_v2.jl:356-381` |
| pose-fix update pattern | `hybrid_ctrl_v2/estimators_v2.jl:391-416` |
| gauss_markov_q / derive_process_noise | `hybrid_ctrl_v2/estimators_v2.jl:51-77` |
| v1 SMO (smooth-switch convention) | `hybrid_ctrl/estimators.jl:341-446` |
| build_estimator_v2 (kwarg pass-through) | `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl:52-60` |
| replay loop to copy | `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl:226-305` |
| 10-dim space + decoder | `hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl:61-111` |
| 11-trajectory manifest | `hybrid_ctrl_v2/estimator_tuning/replay_trajset.jl:46-60` |
| objective terms + in-slip mask | `hybrid_ctrl_v2/estimator_tuning/objective_v2.jl:33-145` |
| baseline results (read-only) | `hybrid_ctrl_v2/runs_estimator_v2_replay/seed1..5/best_config.json` |
