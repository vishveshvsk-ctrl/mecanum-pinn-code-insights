# Observer v2 (Hy3) — γ + ΔV Model, IMU-Only Regime-Split Physics Retrain

> **Generated:** 2026-07-10 (Hy3 fork of `observer-gamma-only-5phase-retrain.md` rev 4)
> **Stack:** Python 3.13, PyTorch 2.6.0+cu124 (conda `myenv`), numpy, pyarrow, pandas
> **Scope:** Parallel `*_v2hy3` modules alongside untouched v1 AND the rev-4 v2 brief.
> **Relationship to the base brief:** This document is a **standalone superset** of
> `observer-gamma-only-5phase-retrain.md`. It reuses that brief's env, data pipeline,
> sensor front-end, warm-start path, 5-phase schedule, and plateau LR scheduler
> **verbatim** unless a section here overrides it. Read this brief as authoritative
> wherever the two disagree; everything not restated here is inherited unchanged.
> **Context docs:** `chat-handoff/a1_a2_experiments_handoff.md`,
> `Mecanum_PINN_Mamba_ForceRecon_v1/FORWARD_V2_STICKSLIP_DESIGN.md`.

---

## 0. What Hy3 changes vs the base brief (delta summary)

The base brief builds a γ-only observer whose physics loss (roller torque balance) is
evaluated with **ground-truth bristle states `zx_lab`/`zy_lab`** fed in as label
tensors, and whose body velocity `V̂` comes from a **fixed complementary filter** with
no learned correction. That design is not deployable as a pure physics-dominated model
because (a) the roller residual consumes GT bristles, and (b) `V̂` carries an
unrecoverable slip bias that poisons the contact kinematics during slip episodes.

Hy3 resolves both with three coordinated changes, all **GT-free at deployment**:

1. **Add a learned velocity-correction head `ΔV̂` [B,2].** The velocity consumed by the
   physics loss is `V_used = V̂ + ΔV̂`, where `V̂` is the fixed front-end (base brief)
   and `ΔV̂` is a second global readout off the shared encoder. `V_used` never enters
   as a network *input*; `V_true` is a **loss-side supervision target only**
   (`ΔV* = V_true − V̂`). This makes the comp-filter slip-gating problem moot — the
   network learns the slip correction instead of a hand-designed gate.

2. **Replace the bristle-state force path with the steady-state (gross-slip) Adamov
   law.** The slip-branch forces are computed from `lugre_ss_friction`
   (`run_one.jl:343-355`) — bristle-eliminated, needs only `(Vp, w_z, mu, chi)`, all
   sensor-derivable. **`zx`/`zy` labels are dropped entirely** from the physics loss.

3. **Regime-split the physics loss.** The steady-state force law is inaccurate in the
   stick region (it predicts `F→0` as `Vp→0`, discarding the elastic bristle restoring
   force). So the loss switches form on a model-derived smooth slip gate `g∈[0,1]`:
   a **force-free kinematic constraint** in stick (`γ̂ → γ_kin`, pure geometry + `V_used`
   + `w`), and the **steady-state roller torque balance** in slip. This lets physics
   identify γ in *both* regimes, delivering the "phys-loss-dominated even for spin" goal.

Everything else — the per-wheel MambaLiteSSM encoder, the γ head, the 5-phase
curriculum, the plateau LR scheduler, the sensor front-end that produces `V̂`, the
warm-start subset, batch 4096 / 200 epochs — is inherited from the base brief.

**Naming:** new modules take the `_v2hy3` suffix (`config_v2hy3.py`, `models_v2hy3.py`,
`losses_v2hy3.py`, `physics_v2hy3.py`, `data_v2hy3.py`, `training_v2hy3.py`, …) so they
coexist with both v1 and the rev-4 v2 modules. v1 stays byte-identical; `physics.py`
(v1) gets comment-only edits; the base brief's `sensor_frontend_v2.py` is imported
unchanged.

---

## 1. Overview

Build **Observer v2-Hy3**: a causal observer that jointly predicts per-wheel roller
spin `γ̂ [B,4]` and a global body-velocity correction `ΔV̂ [B,2]` from
**deployment-available signals only** — IMU (`a_x, a_y`, gyro `ψ̇`), high-resolution
wheel encoders (`w_i, θ_i`), and the fixed-filter velocity estimate `V̂`. A single
shared per-wheel encoder feeds two separate heads. The two heads **interact only
through a soft physics loss** — never through inputs — so the network cannot take the
lazy shortcut `γ̂ ≈ f(V_true)` that would collapse at deployment.

The physics loss is a **PINN soft constraint** (an external scalar regularizer on the
head outputs; the encoder is never autodiffed through an ODE solver). It is
**regime-split**: kinematic in stick, steady-state roller-torque-balance in slip,
blended by a model-derived slip gate. Trained under the base brief's 5-phase curriculum
(supervised γ **and** ΔV ramped 1 → 0.1 floor; physics 0 → 1), with a single
`ReduceLROnPlateau` scheduler spanning all phases.

**System contract:**
`(Gw [B,W,5] = [V̂x, V̂y, ψ̇_gyro, a_x, a_y], Pw [B,W,4,4]) → (γ̂ [B,4], ΔV̂ [B,2])`
(final-step, causal, max-norm, w32 @ 500 Hz, frozen-p95 scaler; per-wheel raw_in = 13).
Campaign: 200 epochs (80/24/40/24/32), batch 4096.

**Deployment vs grounding.** In the *grounding* phase `ΔV̂` is supervised to
`V_true − V̂`, so `V_used → V_true` and the physics loss is evaluated against near-true
slip (matching the GT-referenced base brief). At *deployment* `ΔV̂` is network-predicted,
`V_used = V̂ + ΔV̂` is fully sensor-derived, and **no GT quantity enters the loss** —
the physics loss stays computable and, crucially, stays *valid during slip* (gyro +
accel + encoders are unaffected by wheel slip, unlike odometry).

---

## 2. Architecture Pattern

Single shared encoder, two separate readout heads, external regime-split physics loss.

```
Gw [B,W,5] (V̂x,V̂y,ψ̇,a_x,a_y)  +  Pw [B,W,4,4] (w_i,θ_i,…)
        │        (globals broadcast per wheel; wheel embedding added)
        ▼
  shared feat lifter  raw_in=13 → d_model=32   (SAME lifter for both heads — required)
        ▼
  shared per-wheel MambaLiteSSM  (v1, imported; state_dim=6, d_model=32)
        │  final-step per-wheel latents  H [B, 4, 32]
        ├─► head_γ  : per-wheel readout  Linear→SiLU→Linear(·,1)         → γ̂  [B,4]
        └─► head_ΔV : CONCAT over wheels [B, 4·32] → Linear→SiLU→Linear  → ΔV̂ [B,2]
        (both heads read the SAME fused latent; V_true is NEVER an input)

  V_used = V̂ + ΔV̂                                (ΔV̂ broadcast over the 4 wheels)
  ℒ_phys = mean_{B,i} [ (1−g_i)·r_stick,i²  +  g_i·r_slip,i² ]   ← ONLY coupling of the heads
```

**Why one shared encoder is mandatory (not optional).** `ℒ_phys` depends on both `γ̂`
and `ΔV̂`, and each depends on both modalities: `γ_kin` and `r_slip` need `V_used`
(IMU-path) *and* per-wheel geometry (encoders). If the modalities were split across two
encoders, the γ representation would be blind to the velocity it is evaluated against
and the cross-modal physics gradient would be broken. So the base brief's single
wheel-batched lifter (`raw_in = 5 globals + 4 per-wheel + 4 wheel-emb = 13 → 32`) is
reused, and both heads read its fused output. (Confirmed reusable: base brief §2, §4.)

**Why `head_ΔV` concatenates rather than mean-pools.** The 4 wheels sit in a fixed
O-configuration (distinct `δ_i`, `px_i`, `py_i`); wheel identity matters for the
4-wheels→body-velocity map (wheel-1 slip and wheel-3 slip contribute differently to
`Vx` vs `Vy`). Mean-pooling is permutation-invariant and would force the head to
approximate an asymmetric map. **Concatenate** the 4 per-wheel latents
`[B, 4·d_model] → Linear → [B,2]` at the final time step (causal, matching the γ head's
final-step contract). A small attention/1-D-conv over the 4 slots is an acceptable
equivalent; mean-pool is not.

**Why Mamba (kept over the GRU baseline).** The SSM recurrence
`h_t = Ā·h_{t-1} + B̄·x_t` is a learnable discrete-time linear dynamical system — the
same structural form as the observer equations of the real 39-D plant, so `h` is a
*physics-motivated latent state* whose propagation mirrors how `γ`, `ΔV`, bristles
actually evolve (a genuine inductive bias, independent of `ℒ_phys`). The recurrence
also gives **unbounded deployment memory**: the state carries across windows, so the
0.5–2.4 s slip episodes that broke the ~34 ms fixed filter are within reach even though
training windows are w32 (~64 ms). GRUBaseline stays importable as the comparison point
that lacks this state-space motivation. **Do NOT** hard-zero or low-rank the diagonal
`A`: exact-zero eigenvalues make undamped integrators (runaway on biased input);
memory-horizon extension is achieved by letting `A` learn slow *non-zero* decays
(retention ≈ 0.999 at 500 Hz for a ~2–4 s horizon) and/or a larger `state_dim`, not by
sparsifying `A`. Keep `A` a stable learned diagonal.

**Data-flow guard (the lazy-minimum firewall).** `head_γ` and `head_ΔV` receive
sensor-real encoder features only. `V_true` appears **nowhere as an input** — it defines
the `ΔV` supervision target on the loss side only. The supervised-γ gradient reaches
only `head_γ`; the supervised-ΔV gradient reaches only `head_ΔV`; the sole joint
coupling is `ℒ_phys`. This is what prevents `γ̂ ≈ f(V_true,…)`, a mapping that is
trivially accurate in grounding but does not exist at deployment.

---

## 3. Technology Constraints

Inherited from the base brief §3 verbatim, plus:

- **Explicit exclusions:** NO modification of v1 modules; NO modification of the rev-4
  v2 modules (Hy3 is a third parallel set); NO new encoder architecture (import
  `MambaLiteSSM`/`GRUBaseline` from v1 `models.py`); the fixed comp-filter front-end
  (`sensor_frontend_v2.py`) is imported **unchanged** — Hy3 does NOT re-open the
  slip-gating redesign (the `ΔV̂` head subsumes it); `physics.py` (v1) stays comment-only.
- **New physics helper lives in a new module** `physics_v2hy3.py` (steady-state Adamov
  force + `γ_kin` + steady-state roller residual), importing `contact_from_gamma` from
  v1 `physics.py` unchanged. No v1 physics function is edited.
- **Sensor-real rule** (unchanged): applies to model INPUTS only. `V_true` and its
  derivatives remain allowed on the LOSS side (γ target, `ΔV` target). `V̂` is
  sensor-derived (front-end output), so it is a legal input.
- **χ/μ plumbing (verified):** `phys["mu"]` and `phys["chi"]` are already carried
  per-sample in the batch dict (v1 `losses.py:57,68,104`; v1 `physics.py::lugre_forces`
  takes `chi`). Hy3's steady-state force reuses these directly — no new plumbing.

---

## 4. Component Breakdown

### `config_v2hy3.py` (new)
- Extends `ObserverConfigV2` (import, don't duplicate). Adds:
  - `w_dv: float` — weight on the supervised ΔV term inside `L_sup` (default 1.0;
    scaled by the same `w_sup` phase ramp as γ, so it also honors the 0.1 floor).
  - `dv_scale: Tuple[float,float]` — frozen-p95 normalization for `(ΔVx, ΔVy)` targets
    (reuse the `Vx`/`Vy` p95 entries from the scaler CSV; ΔV is a velocity).
  - `gate_center: float = LG_V_STR` (0.01) and `gate_width: float` — the smooth slip
    gate `g = σ((|Vp| − gate_center)/gate_width)`; `gate_width` default = `gate_center`
    (i.e. one Stribeck-velocity of transition softness).
  - `mindlin_iters: int = 2` — Picard iterations for the steady-state slip-fraction
    (matches `run_one.jl` `LuGreParams.mindlin_iters`).
  - `w_phys: float` and (retained) `w_roller` — but note Hy3's physics loss is a single
    regime-split channel (see `losses_v2hy3`), so the base brief's `physics_variant`
    (residual/integrated) selector is **not used**; default it off.
  - Run-tag: `S{fold}_train_w{W}_gamma_dv_v2hy3_phys_max_norm`.
- Physics-scale constants pulled from v1 `config.py`: `P2 = 5.78e-3`,
  `LG_V_STR = LG_W_STR = 0.01`, `LG_STICTION_RATIO = 1.1`, `LG_SIGMA0 = 1.64e3`,
  `LG_EPS_REG = 1e-4`; γ frozen-p95 scale `GAMMA_SCALE = 82.81` (from the scaler CSV).

### `sensor_frontend_v2.py` (imported unchanged from the base brief)
- Produces `Gw = (V̂x, V̂y, ψ̇_gyro, a_x, a_y)` at 500 Hz via the fixed complementary
  filter. Hy3 consumes `V̂x, V̂y` as the base of `V_used`. No changes.

### `data_v2hy3.py` (new; thin wrapper over `data_v2.py` / v1 `data.py`)
- Emits, per window:
  - `y_gamma [B,4]` (normalized γ target) — unchanged.
  - **`y_dv [B,2]` (new)**: `ΔV* = V_true − V̂`, normalized by `dv_scale`. `V_true` is
    the GT body velocity at the window's final step (loss side only); `V̂` is the
    front-end output. Both already available in the cache/front-end.
  - `phys` dict (**GT-free-capable**): `mu [B]`, `chi [B]`, `psi_dot [B]` (gyro),
    per-wheel geometry `cti [B,4], sti [B,4]` (cos/sin θ̃ from encoder θ), `w [B,4]`,
    and the **front-end base contact velocities** `Vpx0_hat [B,4], Vpy0_hat [B,4]`
    computed from `V̂` (NOT `V_true`) via the γ=0 contact kinematics
    (`Vpx0 = V̂x − ψ̇(py_i+DY_i) − w_i·R`, `Vpy0 = V̂y + ψ̇·px_i`). These are the
    sensor-derived bases the physics loss shifts by `ΔV̂` in-graph (see §6).
  - `slip_mag [B,4]`, `sup_weight [B]` — unchanged (high-slip upweighting).
- **`zx_lab`/`zy_lab` are DROPPED** from the physics path (steady-state force needs no
  bristles). They may still be loaded **only** for the optional audit cross-check
  (§8 step 1), never for the trained loss.
- **Cache policy:** unchanged from base brief (cache stores clean exact-dynamics accel +
  raw states/labels; `V̂`, `ΔV*`, and any noise computed at load). Bump the cache
  version key if `Vpx0_hat/Vpy0_hat` are precomputed into the cache; otherwise assemble
  them at load from cached `V̂` + geometry.

### `WheelObserverV2Hy3` (class in `models_v2hy3.py`, new)
- v1 feature path + shared encoder (imported) → per-wheel latents `H [B,4,32]`.
- `head_gamma`: per-wheel `Linear(32,H)→SiLU→Linear(H,1)` → `γ̂ [B,4]` (normalized).
- **`head_dv` (new)**: concatenate the 4 latents → `Linear(4·32, H)→SiLU→Linear(H,2)`
  → `ΔV̂ [B,2]` (normalized by `dv_scale`).
- `forward(Gw, Pw) → (gamma_hat [B,4], dv_hat [B,2])`.
- Warm-start: identical subset to the base brief (encoder core + `wheel_emb` transfer;
  `feat` lifter and `head_gamma` init fresh because raw_in changed 11→13); **`head_dv`
  always inits fresh** and is added to the skipped-keys log.

### `physics_v2hy3.py` (new — the only net-new physics code)
- `lugre_ss_forces(xp, mu, N, chi, w_z, Vpx, Vpy, mindlin_iters)` — **1:1 transcription
  of `lugre_ss_friction` (`run_one.jl:343-355`), the authoritative gross-slip Adamov
  law**, bristle-eliminated:
  ```
  Vp  = sqrt(Vpx² + Vpy² + eps²);   awz = sqrt(w_z² + eps²)
  c_t = (8/3π)·awz·chi
  fsl = mindlin_fslip(Vp, c_t, mindlin_iters)          # Picard, ≥1 iter (see below)
  s_t = fsl·c_t + Vp
  s_s = (16/3π)·awz·chi + 5·Vp
  g_t = mu·(1 + (STICTION_RATIO−1)·exp(−(s_t/V_STR)²)) # stribeck_g, == physics.py:52
  g_s = mu·(1 + (STICTION_RATIO−1)·exp(−(s_s/W_STR)²))
  Fx  = −N·g_t·Vpx/s_t;   Fy = −N·g_t·Vpy/s_t;   Mz = −N·chi²·g_s·w_z/s_s
  ```
  with `mindlin_fslip(V, c, iters)` the Picard fixed-point of `run_one.jl:336-342`:
  ```
  x = V/(c+V); fs = 1
  repeat iters:  b = max(1−x, 1e-9);  fs = 1 − b^(2/3);  x = V/(fs·c + V)
  b = max(1−x, 1e-9);  return 1 − b^(2/3)
  ```
  Backend-agnostic (`xp` = torch here), fp32. `mu, chi` broadcast from `phys` (per
  sample); `N = C.N_PER_ROLLER`. **No `zx/zy/zs` arguments** — this is the whole point.
- `gamma_kin(Vpx0, Vpy0, dVpx_dg, dVpy_dg)` — the stick-region kinematic γ from the
  pure-rolling (zero contact-slip) least-squares condition (over-determined 2 eqns
  `Vpx=Vpy=0`, 1 unknown γ):
  ```
  γ_kin = −(Vpx0·dVpx_dg + Vpy0·dVpy_dg) / (dVpx_dg² + dVpy_dg² + eps)
  ```
  Pure geometry + `V_used` (via `Vpx0/Vpy0`) + `w`; **no forces, no GT**. `dVpx_dg`,
  `dVpy_dg` come from `contact_from_gamma` (v1, pure geometry, γ-independent).
- `roller_residual_ss(gamma, mu, chi, psi_dot, Vpx0, Vpy0, cti, sti, mindlin_iters)` —
  the base brief's roller balance with the steady-state force substituted for the
  bristle force:
  ```
  Vpx,Vpy,w_z,dVpx,dVpy,_ = contact_from_gamma(gamma, psi_dot, Vpx0, Vpy0, cti, sti)
  Fx,Fy,_ = lugre_ss_forces(xp, mu, N, chi, w_z, Vpx, Vpy, mindlin_iters)
  r = P2·gamma + Fx·dVpx + Fy·dVpy          # per wheel, → 0 in quasi-static slip
  ```
  (The `Mz·dwz/dg` lever stays dropped, as in v1 `roller_residual`: `Mz ∝ chi² ≈ 2.5e-5`.)

### `losses_v2hy3.py` (new)
- **Supervised loss** (two targets, one weight):
  `L_sup = w_sup · [ MSE(γ̂, y_gamma) + w_dv · MSE(ΔV̂, y_dv) ]` (normalized scale,
  γ term uses `sup_weight` per-window upweighting). `y_dv = (V_true − V̂)/dv_scale`.
- **Physics loss (regime-split, the core Hy3 change):**
  ```
  V_used   = V̂ + ΔV̂                                  # ΔV̂ broadcast over 4 wheels
  Vpx0_u   = Vpx0_hat + ΔV̂x ;  Vpy0_u = Vpy0_hat + ΔV̂y   # affine, unit coeff (see §6)
  γ̂_phys   = denorm(γ̂)  (× GAMMA_SCALE, in-graph)
  # geometry sensitivities (γ-independent, pure geometry):
  _,_,_,dVpx_dg,dVpy_dg,_ = contact_from_gamma(0, psi_dot, Vpx0_u, Vpy0_u, cti, sti)
  γ_kin    = gamma_kin(Vpx0_u, Vpy0_u, dVpx_dg, dVpy_dg)
  r_roll   = roller_residual_ss(γ̂_phys, mu, chi, psi_dot, Vpx0_u, Vpy0_u, cti, sti, iters)
  # model-derived slip magnitude for the gate (GT-free):
  Vpx,Vpy,_,_,_,_ = contact_from_gamma(γ̂_phys, psi_dot, Vpx0_u, Vpy0_u, cti, sti)
  Vp_mag   = sqrt(Vpx² + Vpy² + eps²)
  g        = sigmoid((Vp_mag − gate_center)/gate_width)          # g→0 stick, g→1 slip

  # ---- SCALING (per user decision — resolves the stick/slip unit mismatch) ----
  r_stick  = (γ̂_phys − γ_kin) / GAMMA_SCALE                     # normalized-γ units
  r_slip   = r_roll / (P2 · GAMMA_SCALE)                          # normalized-γ units

  L_phys   = mean_{B,i}[ (1 − g)·r_stick² + g·r_slip² ]
  ```
- **Total:** `total = L_sup + w_phys · L_phys`  (phase weights `w_sup`, `w_phys` from the
  base brief's `phase_weights`; `w_sup ≥ 0.1` floor covers both γ and ΔV).
- Log dict: per-wheel `phys_stick_w{i}`, `phys_slip_w{i}`, mean gate `g_mean`, and the
  supervised `sup_gamma`, `sup_dv`.

### `training_v2hy3.py` (new)
- Base brief's 5-phase runner + single `ReduceLROnPlateau` + ramp-freeze, **unchanged**,
  with two edits confined to the loss/metric layer:
  1. `L_sup` now spans γ **and** ΔV (additive, same `w_sup`). The 0.1 floor thereby
     enforces "min ground loss even for spin" for both heads automatically.
  2. `terminal_val_loss` must evaluate at terminal weights `(w_sup=0.1, w_phys=1)` and
     **include the ΔV supervised term** — otherwise the scheduler/best-ckpt selector
     could pick a checkpoint with a bad correction head. This is the one required metric
     edit.
- Grounding-first ordering is load-bearing (not optional): `w_sup=1` holds through
  grounding + phys_rampup + overlap so `ΔV̂` (hence `V_used` and `γ_kin`) is accurate
  *before* physics dominates; only then does `w_sup→0.1`. Assert `w_sup` never 0.
- **fp32 physics recompute stays mandatory** (autocast-exempt): `r_roll` carries
  `P2·γ` with `P2 = 5.78e-3`; bf16 drops it against O(1) force terms.
- Reproduce-and-fix the base brief's physics-phase launch failure on a `--limit-files`
  dummy run before any campaign. (Hy3's loss is lighter than the integrated variant —
  no one-step rollout — so OOM/timeout is less likely, but keep the pre-flight.)

### `evaluation_v2hy3.py` + `make_observability_report_v2hy3.py` (new, thin)
- γ and derived-ω_z metrics as base brief, **plus** a `ΔV̂` diagnostics block: `V_used`
  RMSE vs `V_true` binned by slip/profile, and the low-slip-bin `V_used` error (the
  region where `γ_kin` matters). Emit `gamma_error_by_slip.csv` (A1-v2 consumer
  contract) unchanged.

### `physics.py` (v1, comment-only)
- Add a docstring note pointing to `physics_v2hy3.py` for the steady-state (bristle-free)
  force path and the regime-split usage. No code change.

---

## 5. File & Directory Structure

```
observer_v1_py/
├── train_observer_v2hy3.py             # NEW — Hy3 entry point (mirrors v2 CLI + --w-dv)
├── make_observability_report_v2hy3.py  # NEW — γ/ΔV/ω_z report
├── launch_parallel.py                  # (already v2-aware) add --entry train_observer_v2hy3.py
├── mecanum_observer/
│   ├── config_v2hy3.py                 # NEW — extends ObserverConfigV2 (+ΔV, gate, mindlin)
│   ├── data_v2hy3.py                   # NEW — +y_dv target, Vpx0_hat/Vpy0_hat, drop zx/zy
│   ├── models_v2hy3.py                 # NEW — WheelObserverV2Hy3 (γ head + ΔV concat head)
│   ├── physics_v2hy3.py                # NEW — lugre_ss_forces, gamma_kin, roller_residual_ss
│   ├── losses_v2hy3.py                 # NEW — supervised γ+ΔV, regime-split physics
│   ├── training_v2hy3.py               # NEW — 5-phase runner (+ΔV in L_sup and terminal metric)
│   ├── evaluation_v2hy3.py             # NEW — γ/ΔV/ω_z metrics
│   ├── sensor_frontend_v2.py           # imported unchanged (base brief)
│   ├── physics.py                      # v1, comment-only note
│   └── (all v1 AND rev-4 v2 modules untouched)
└── runs/S{1,2}_train_w32_gamma_dv_v2hy3_phys_max_norm/
```

---

## 6. Key Interfaces & the in-graph V_used shift

```python
# models_v2hy3.py
class WheelObserverV2Hy3(nn.Module):
    def forward(self, Gw: Tensor, Pw: Tensor) -> Tuple[Tensor, Tensor]:
        """Gw [B,W,5] sensor-real globals; Pw [B,W,4,N] per-wheel measurables.
        Returns (gamma_hat [B,4], dv_hat [B,2]), both normalized."""

# physics_v2hy3.py
def lugre_ss_forces(xp, mu, N, chi, w_z, Vpx, Vpy, mindlin_iters=2): ...  # -> (Fx,Fy,Mz)
def gamma_kin(Vpx0, Vpy0, dVpx_dg, dVpy_dg): ...                          # -> [.,4]
def roller_residual_ss(gamma, mu, chi, psi_dot, Vpx0, Vpy0, cti, sti,
                       mindlin_iters=2): ...                              # -> r [.,4]

# losses_v2hy3.py
def supervised_loss(gamma_hat, dv_hat, y_gamma, y_dv, sup_weight, w_dv): ...   # -> (scalar, log)
def physics_loss_hy3(gamma_hat, dv_hat, V_hat, phys, cfg): ...                 # -> (scalar, log)
#   assembles V_used = V_hat + dv_hat, shifts Vpx0/Vpy0, computes regime-split L_phys.
```

**The in-graph `V_used` shift (why it is exact and cheap).** The γ=0 contact velocities
are **affine in body velocity with unit coefficient**:
`Vpx0 = Vx − ψ̇(py+DY) − wR`, `Vpy0 = Vy + ψ̇·px` ⇒ `∂Vpx0/∂Vx = 1`, `∂Vpy0/∂Vy = 1`
(cross terms 0). So applying the learned correction is a plain additive shift of the
**front-end base**:
```
Vpx0_used = Vpx0_hat + ΔV̂x         # Vpx0_hat built from V̂ in data_v2hy3
Vpy0_used = Vpy0_hat + ΔV̂y
```
`ΔV̂` (broadcast over 4 wheels) is the only route `V_used` enters the loss, and its
gradient flows to `head_dv` through **both** the stick term (`γ_kin` depends on
`Vpx0_used/Vpy0_used`) and the slip term (`r_roll` forces depend on `Vp`). No GT `Vx/Vy`
is touched: `Vpx0_hat` is sensor-derived (from `V̂`), and `ΔV̂` is a network output.

---

## 7. Data Flow

1. Raw 2000 Hz → `sensor_frontend_v2` (unchanged) → `Gw` (incl. `V̂x,V̂y`) at 500 Hz →
   `data_v2hy3` windowing → `(Gw [B,W,5], Pw [B,W,4,4])` normalized inputs,
   `y_gamma [B,4]`, `y_dv [B,2] = (V_true−V̂)/dv_scale`, and `phys` with
   `mu, chi, psi_dot, cti, sti, w, Vpx0_hat, Vpy0_hat, slip_mag, sup_weight`.
   **No `zx/zy` in `phys`.**
2. `WheelObserverV2Hy3(Gw, Pw) → (γ̂ [B,4], ΔV̂ [B,2])` (shared encoder, two heads).
3. Supervised branch: `w_sup·[MSE(γ̂,y_gamma; sup_weight) + w_dv·MSE(ΔV̂,y_dv)]`.
4. Physics branch: `V_used = V̂ + ΔV̂` → shift `Vpx0/Vpy0` in-graph → `contact_from_gamma`
   geometry → `γ_kin` (stick) and steady-state `r_roll` (slip) → model-derived gate `g`
   → `L_phys = mean[(1−g)·((γ̂_phys−γ_kin)/GAMMA_SCALE)² + g·(r_roll/(P2·GAMMA_SCALE))²]`.
5. `total = L_sup + w_phys·L_phys`; per-epoch `terminal_val_loss` (incl. ΔV) →
   `scheduler.step` only in constant-objective phases (ramp-freeze) → log LR.
6. Gradient-flow: `V_true` enters only as the `y_dv` target (assert not an input);
   γ̂-denorm and the `Vpx0/Vpy0 += ΔV̂` shift stay in-graph; the Stribeck negative-slope
   band admits multiple balance roots — the 0.1 supervised floor is the disambiguator in
   slip, while `γ_kin` is a hard single-valued constraint in stick.

---

## 8. Implementation Sequence

1. **(Optional) roller audit** — reuse the base brief's `roller_audit.py`. Additionally,
   validate `lugre_ss_forces` against the stored Arrow forces in the **gross-slip regime
   only** (high `|Vp|` bins): the steady-state law should match the dynamic
   `lugre_forces` (and stored forces) to a few-percent where bristles are saturated, and
   deviate in stick (expected — that is why the loss regime-splits). This is the
   numerical justification for the steady-state substitution.
2. `config_v2hy3.py` — ΔV/gate/mindlin knobs; pull `P2, LG_V_STR, GAMMA_SCALE`, etc.
3. `physics_v2hy3.py` — unit-test: (a) `lugre_ss_forces` vs Julia `lugre_ss_friction`
   on matched inputs (residual ~0); (b) `gamma_kin` recovers the true γ when
   `Vpx0/Vpy0` are built from *true* V and the wheel is in pure roll (`|Vp|≈0`);
   (c) `roller_residual_ss` is small on GT states in low/mid-slip bins.
4. `data_v2hy3.py` — `y_dv` matches `V_true − V̂`; `Vpx0_hat/Vpy0_hat` reproduce the
   γ=0 contact velocities from `V̂` (cross-check against v1's `Vpx0/Vpy0` built from the
   same V).
5. `models_v2hy3.py` — shapes; two-head forward; warm-start skips exactly
   `{v1 head bank, feat, head_dv}`; 10-batch overfit of γ **and** ΔV decreases.
6. `losses_v2hy3.py` — verify the regime-split blend is dimensionless (both branches in
   normalized-γ units); gate `g` sweeps 0→1 across a synthetic `|Vp|` ramp; confirm
   grounding-phase `L_phys` with `ΔV̂→ΔV*` matches a GT-V reference within tolerance.
7. `training_v2hy3.py` — 0.1 floor asserted for both heads; terminal metric includes ΔV;
   reproduce-and-fix the launch failure on `--limit-files`.
8. `evaluation_v2hy3.py` + report; `train_observer_v2hy3.py` + launcher entry.
9. Campaign: S1+S2 w32, warm-started AND from-scratch (does the 3-state trunk transfer
   or fight the γ+ΔV objective?).

---

## 9. ML-Specific Considerations

- **Physics-loss scaling (the resolved unit mismatch).** Stick residual `(γ̂−γ_kin)` is
  in γ units; slip residual `r_roll` is in N·m — they cannot be blended raw. Per the
  design decision: **normalize the stick term by `GAMMA_SCALE` and the slip term by
  `P2·GAMMA_SCALE`.** The `P2·GAMMA_SCALE` divisor makes the slip residual's leading
  `P2·γ` term collapse to `γ/GAMMA_SCALE` — i.e. the *same normalized-γ scale* as the
  stick term. Both branches then carry "error in normalized γ," so `(1−g)·r_stick² +
  g·r_slip²` is a consistent, dimensionless blend and the physics gradient magnitude to
  `γ̂` is continuous across the stick↔slip transition. (This supersedes the base brief's
  `/ROLLER_SCALE` non-dimensionalization for the roller term.)
- **Gate reliability during warm-up.** `g` is computed from the model's own
  `γ̂, V_used` — noisy when `γ̂` is untrained. The grounding-first schedule (supervised
  γ,ΔV before `w_phys` ramps) ensures `γ̂` and `V_used` are accurate before the gate
  governs anything. Use a smooth sigmoid (not a hard switch) so microslip transitions
  are continuous.
- **Steady-state vs dynamic force (modeling approximation, stated).** The slip branch
  uses the bristle-eliminated Adamov law: exact only where the bristle deflection is
  saturated (gross slip). The regime split confines it there; the stick branch is
  force-free. The residual pre-sliding microslip discrepancy is absorbed by the 0.1
  supervised floor — its job.
- **Lazy-minimum guards.** (a) `V_true` is never an input. (b) `d_model=32` is
  sufficient capacity for the ~10–20 dominant slip modes, so the network is not forced
  into the trivial `ΔV̂≈0` ("just trust V̂") collapse. (c) In the physics-dominated
  phase, `w_sup→0.1` on **both** heads keeps the firewall active after grounding.
- **fp32 physics** (autocast-exempt): `P2·γ` term, `P2=5.78e-3`. **Numerical:** clamp
  `slip_mag` and `(dVpx_dg²+dVpy_dg²)` denominators before division; γ-denorm uses the
  frozen p95 (82.81), never a per-batch statistic; `LG_EPS_REG=1e-4` in all `sqrt`.
- **Batch/step-count** and **checkpointing** as base brief §9 (record `w_dv`, gate
  params, `mindlin_iters`, and `model="ssm_v2hy3_gamma_dv"` in metrics.json).

---

## 10. Success Criteria

- [ ] `lugre_ss_forces` matches Julia `lugre_ss_friction` on matched inputs (~0 residual)
      and matches dynamic/stored forces in high-slip bins.
- [ ] `gamma_kin` recovers true γ under pure-roll GT velocities; `roller_residual_ss`
      small on GT states in low/mid-slip.
- [ ] `WheelObserverV2Hy3` two-head 10-batch overfit decreases monotonically for γ and
      ΔV; GRU option still constructs; v1 + rev-4 v2 untouched.
- [ ] `load_warm_start` skips exactly {v1 head bank, feat, head_dv} (logged).
- [ ] Regime-split blend verified dimensionless; gate `g` sweeps 0→1 across `|Vp|`;
      grounding-phase `L_phys` (with `ΔV̂→ΔV*`) matches a GT-V reference.
- [ ] Phase weights logged; `w_sup` reaches exactly 0.1 (both heads) and never 0.
- [ ] LR trace plateau-only (starts 2e-3, no boundary steps, ramp-freeze honored);
      scheduler+counter survive kill-and-resume.
- [ ] Campaign: cross-subset γ RMSE ≤ v1 baseline (0.056–0.067 norm) overall; high-slip
      γ RMSE materially below v1's ~0.13–0.15; **`V_used` RMSE < `V̂` RMSE in slip bins**
      (proof the ΔV head earns its keep); no regression in derived-ω_z.
- [ ] Trained model's binned roller residual approaches the quasi-static floor in slip.
- [ ] `gamma_error_by_slip.csv` emitted per run.
- [ ] **Deployability check:** an eval pass with **all GT removed from the loss path**
      (ΔV network-predicted, no `zx/zy`, no `V_true`) runs and produces a finite,
      sensible `L_phys` — confirming the model is physics-dominated GT-free.

## 11. Out of Scope

- Any change to v1 or rev-4 v2 modules, runs, or checkpoints.
- Re-opening the comp-filter slip-gating redesign (the `ΔV̂` head replaces it).
- Bristle-state prediction / dynamic LuGre in the loss (`zx/zy` dropped from training).
- `Mz` / spin-torque channel in the residual (`chi²`-scaled, low-SNR; lever dropped).
- Sim-to-real adaptation mechanics; stage-2 noise calibration (the `noise_stage` toggle
  ships; calibrated noisy campaign is follow-up).
- Learned front-end (`V̂` stays the fixed comp filter; only the additive `ΔV̂` is learned).
- Window/batch re-ablation (w32, max-norm, b4096 pinned).
```
