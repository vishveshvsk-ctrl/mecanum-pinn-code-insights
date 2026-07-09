# Observer v2 — γ-Only Model + 5-Phase Physics Retrain (observer_v1_py)

> **Generated:** 2026-07-09 (rev 4: sensor-real input contract — IMU accelerations + gyro + encoder channels replace direct Vx/Vy; rev 3: plateau LR scheduler; rev 2: v2 parallel modules, head bank dropped)
> **Stack:** Python 3.13, PyTorch 2.6.0+cu124 (conda `myenv`), numpy, pyarrow, pandas
> **Scope:** New v2 model + training path alongside the untouched v1; data pipeline wrapped
> **Context docs:** `chat-handoff/a1_a2_experiments_handoff.md` (env/data/conventions),
> `Mecanum_PINN_Mamba_ForceRecon_v1/FORWARD_V2_STICKSLIP_DESIGN.md` (why γ̂ is the only A2 output A1-v2 consumes)

## 1. Overview

Build **Observer v2**: a γ-only (roller spin rate) causal observer as parallel `*_v2`
modules inside `observer_v1_py/mecanum_observer/`, leaving every v1 module, run dir,
and checkpoint untouched. The v1 per-state head bank (`nn.ModuleList` of 3 MLPs) is
**dropped entirely** — v2 has a single γ head on the shared wheel-batched encoder.
Design rationale (pinned): LuGre will NOT be grounded in sim-to-real (A1-v2 keeps only
a generic slip-based leaky-integrator form), so ẑx/ẑy have no consumer; γ̂ is the
deliverable (feeds A1-v2's slip refinement, gate, and anchor detection). zx/zy labels
remain in the v2 data pipeline **only as physics-loss auxiliary tensors** (the LuGre
force recompute needs them); no head predicts them. Retrain under the 5-phase
curriculum with physics loss enabled — including the **roller torque-balance residual
promoted from monitor-only to trained** — with the supervised weight ramped 1 → **0.1
floor, never zero** (W_SUP_MIN). **Sensor-real input contract (rev 4):** the observer consumes ONLY
deployment-available signals — IMU (raw accelerations a_x, a_y; gyro yaw rate ψ̇),
high-resolution wheel encoders (w_i, θ_i), and actuator torque (Msat_i). Direct body
velocities are NOT inputs: V̂x, V̂y are produced by a **fixed causal complementary
filter** (wheel-odometry kinematic velocity as the low-frequency anchor + IMU
acceleration path for high frequency) that is part of the deployed stack and runs
identically in the training pipeline. Yaw ANGLE is needed nowhere (body-frame
formulation); ψ̇ is the direct gyro output. System contract:
`(Gw [B,W,5] = [V̂x, V̂y, ψ̇_gyro, a_x, a_y], Pw [B,W,4,4]) → γ̂ [B,4]` (final-step,
causal, max-norm, w32 @ 500 Hz, frozen-p95 scaler; per-wheel raw_in = 5+4+emb(4) = 13). Campaign schedule: **200 epochs total
(80/24/40/24/32), batch 4096** — the step-count-per-epoch consequences of the larger
batch are deliberate (see §9).

## 2. Architecture Pattern

Single-target causal filter with physics-regularized refinement: the v1 encoder
(MambaLiteSSM / GRUBaseline, imported — not copied) is unchanged; the head collapses
to one γ MLP, and the loss gains the one physics channel whose primary unknown is γ —
the quasi-static roller torque balance (`p2·γ + Fx·∂Vpx/∂γ + Fy·∂Vpy/∂γ = 0`, already
implemented at `mecanum_observer/physics.py::roller_residual`). zx/zy enter the
physics recompute as **label tensors** (constants), so all physics gradients flow to
γ̂ alone — the gradient-routing safeguard comes for free.

## 3. Technology Constraints

- **Python:** 3.13 via `C:\Users\vishv\miniforge3\envs\myenv\python.exe` (no `conda activate`; workers need `-u` + `PYTHONUTF8=1`)
- **PyTorch:** 2.6.0+cu124; no `torch.compile` on Windows paths
- **Required libraries:** numpy, pyarrow (existing loader), pandas (report CSVs)
- **Device targets:** CUDA 6 GB tier (b2048 preset) and 24 GB box (b4096); CPU smoke test
- **Explicit exclusions:** NO modification of v1 modules (`config.py`, `data.py`, `models.py`, `losses.py`, `training.py`, `evaluation.py` stay byte-identical; `physics.py` comment-only edits allowed); NO new encoder architecture (import `MambaLiteSSM`/`GRUBaseline` from v1 `models.py`); NO multi-state head bank or `target_states` plumbing in v2 — γ is hard-wired; NO change to the decimated cache format (normalization-agnostic, stores raw arrays incl. zx/zy — reused as-is); zs stays dropped (zeros); ≤8 workers; run from `code_insights/` with `--cache-dir C:/Users/vishv/mecanum_cache_decim` on the laptop. **Sensor-real rule (rev 4):** applies to model INPUTS only — ground-truth Vx/Vy/derivatives remain allowed on the LOSS side (supervision + physics labels in the `phys` dict); the velocity front-end is a FIXED deterministic causal filter (coefficients in config), never learned, single implementation shared by training and deployment

## 4. Component Breakdown

### `config_v2.py` (new)
- **Type:** dataclass (`ObserverConfigV2`) + module constants (imports physical/scale constants from v1 `config.py` — single source, no duplication)
- **Responsibility:** v2 run configuration: encoder knobs (same defaults: d_model 32, state_dim 6, w32, max-norm, frozen-p95 scaler), the 5-phase schedule with `W_SUP_MIN = 0.1` floor — **v2 uses the ~200-epoch scaled schedule: grounding 80 / phys_rampup 24 / overlap 40 / grnd_rampdown 24 / physics 32 (= 200 total; proportional 0.8× of the v1 250-epoch PHASE_SCHEDULE), and the schedule carries epoch counts and weight ramps ONLY — the per-phase `lr_scale` column is dropped**; `batch_size` default **4096** (A1-style; the model is tiny, batch is a throughput knob — fall back to 2048 only if the 6 GB tier thrashes); physics knobs (`physics_variant`, `w_roller: float`, `roller_slip_weighting: bool`, `gamma_high_slip_upweight: float`), LR/scheduler knobs (`lr: float` — same starting value as v1, 2e-3; `sched_factor: float`, `sched_patience: int` (pinned default 10), `sched_min_lr: float`, `sched_rel_threshold: float`), sensor knobs (`SensorNoiseSpec` — accel noise/bias/tilt, gyro noise/bias; encoders clean/high-res; **`noise_stage: {"none", "real"}`, default `"none"` for campaign 1** — the architecture is validated on exact-by-construction IMU first, corruption lands as campaign 2 with NO other change), filter knobs (`vel_filter_crossover_hz: float`; type fixed = first-order complementary), anti-alias LPF cutoff for synthesized accelerations (applied before decimation in BOTH stages), `warm_from: str`, run-tag convention `S{fold}_train_w{W}_gamma_v2_phys_max_norm` (+ `_noisy` suffix for stage 2).
- **Depends on:** v1 `config.py` (constants only)

### `sensor_frontend_v2.py` (new module in `mecanum_observer/`)
- **Type:** functions (numpy, data-pipeline side) — also the deployment front-end
- **Responsibility:** Synthesize and process the sensor channels: (a) IMU synthesis from native 2000 Hz sim arrays — PRIMARY route is **exact dynamics**: evaluate the body EOM with the stored per-wheel forces and coupling terms (`a = M_body⁻¹·NE(F_stored, ...)`; all inputs are Arrow columns), then convert to the **accelerometer observable** (specific force resolved in body axes: `a_x = V̇x − ψ̇·Vy`, `a_y = V̇y + ψ̇·Vx`; **code-verify against `run_one.jl` whether the sim EOM's V̇ terms already carry the ω×V coupling**; authority rule), then a light anti-alias LPF (LuGre chatter above 250 Hz must not fold into the 500 Hz band; real IMUs low-pass internally — stays on in both stages) and decimate. A central-FD-of-states route is implemented ONCE as a label-independent cross-check (must agree with the dynamics route to the solver's dense-output tolerance — the "residual 0.000" idiom) and is not used for data generation. Gyro ψ̇ direct. Corruption per `SensorNoiseSpec` applied ONLY when `noise_stage="real"` (stage-1 campaign runs noiseless: the wrench channel is the true dynamics, isolating architecture capacity from sensor realism); (b) wheel-odometry velocity — kinematic least-squares map (w, θ) → V_odom (the standard mecanum inverse-kinematics estimate; slip-corrupted by nature, used only as the low-frequency anchor); (c) the fixed first-order **complementary filter** → V̂x, V̂y at 500 Hz: low-frequency branch = V_odom (LPF); high-frequency branch = **body-frame strapdown mechanization**, NOT naive integration — the accelerometer observable includes the ω×V transport term, so the fast path integrates `V̂x += Δt·(a_x + ψ̇·V̂y)`, `V̂y += Δt·(a_y − ψ̇·V̂x)` (measured gyro rate, own estimate in the coupling; no yaw angle anywhere), blended at `vel_filter_crossover_hz`. Accel bias is high-passed out by the blend; the ψ̇·δV coupling error is second-order and bounded by the odometry anchor. Integrator: exact planar rotation for the ω×V term (cos/sin ψ̇Δt, norm-preserving) + 2nd-order causal accel term (AB2: `Δt·(1.5a[k] − 0.5a[k−1])`), optionally run at the IMU's native rate then decimated — all a-few-flops-per-step, MCU-deployable; plain Euler is kept as the audit baseline. Function contracts mirror A1's `imu_features.py` (duplicated small functions, NOT a cross-package import — flag for a later shared module).
- **Inputs:** raw 2000 Hz per-trajectory arrays (`Vx, Vy, psi_dot, w, theta`), `SensorNoiseSpec`, filter crossover, decim factor, seed
- **Outputs:** `Gw_channels [T500, 5]` = (V̂x, V̂y, ψ̇_gyro, a_x, a_y)
- **Depends on:** `config_v2.py` (spec + geometry constants from v1 `config.py`)

### `data_v2.py` (new)
- **Type:** window builder / `Dataset` (thin wrapper reusing v1 `data.py` internals — Arrow read, decimation, cache, regime split, normalization)
- **Responsibility:** Emit γ-only targets, and **always** load zx/zy (physical units) plus the existing measurable/aux terms into the `phys` dict for the physics recompute; provide `slip_mag [B,4]` and a per-window supervised weight upweighting high-slip windows (they are ~1% of data but carry 3× the γ error).
- **Inputs:** `ObserverConfigV2`, regime TOML, scaler CSV, cache dir
- **Outputs:** batch `(Gw [B,W,5], Pw [B,W,4,4], y [B,4] γ normalized, phys: Dict incl. zx_lab [B,4], zy_lab [B,4], slip_mag [B,4], sup_weight [B])` — `Gw` is sensor-real (from `sensor_frontend_v2`); the `phys` dict keeps GROUND-TRUTH kinematics (loss side, never inputs); the 5 Gw channels get their own frozen-p95 scaler entries. **Cache policy:** the decimated .npz cache stores the CLEAN exact-dynamics accel channels ALONGSIDE the existing raw states/labels (ground-truth Vx/Vy stay, as loss-side labels and audit reference); V̂x/V̂y and stage-2 noise are computed AT LOAD — the cache stays filter- and stage-agnostic (bump the cache version key once)
- **Depends on:** `config_v2.py`, v1 `data.py` (imported functions)

### `WheelObserverV2` (class in `models_v2.py`, new)
- **Type:** `nn.Module`
- **Responsibility:** v1 feature path (wheel embedding, wheel-batched shared encoder) with the head bank **replaced by a single γ head** (`Linear → SiLU → Linear(·,1)`); returns per-wheel γ̂ directly.
- **Inputs:** `Gw [B,W,5]` (sensor-real globals), `Pw [B,W,4,N_PERWHEEL]`
- **Outputs:** `gamma_hat [B,4]` (normalized). Per-wheel raw_in = 13 (5+4+4).
- **Key constructor params:** `cfg: ObserverConfigV2`
- **Depends on:** v1 `models.py` (`MambaLiteSSM`, `GRUBaseline` imported), `config_v2.py`

### `losses_v2.py` (new)
- **Type:** functions
- **Responsibility:** (a) supervised γ loss: weighted MSE on normalized scale using `sup_weight`; (b) physics loss: de-normalize γ̂ → assemble the 4-state tuple `(γ̂_phys, zx_lab, zy_lab, 0)` → existing variant channels (residual: wheel+body; integrated: one-step) **plus the promoted roller term** `w_roller · mean(r_roll²)` with `r_roll = roller_residual(...)/ROLLER_SCALE`, optionally slip-weighted per wheel; (c) phase weighting `total = w_sup·L_sup + w_phys·(L_variant + w_roller·L_roller)`.
- **Inputs:** `gamma_hat [B,4]` normalized, `y [B,4]`, `phys` dict, phase weights
- **Outputs:** `(total, log dict)` — per-wheel `phys_roller_w{i}` keys now trained (no `_MON` suffix)
- **Depends on:** v1 `physics.py` (unchanged functions), `config_v2.py`

### `physics.py` (v1, comment-only)
- **Responsibility:** already provides everything needed (`contact_from_gamma`, `roller_residual`, `wheel_residual`, `body_residual`, `integrated_step`). Docstring note only: v2 trains the roller term with label-sourced zx/zy; v1 keeps it monitor-only.

### `training_v2.py` (new)
- **Type:** stage runner
- **Responsibility:** 5-phase loop (grounding → phys_rampup → overlap → grnd_rampdown → physics) with physics weight 0 → 1 over `phys_rampup` and supervised weight 1 → **0.1** over `grnd_rampdown`, **held at 0.1 through the final phase** (assert never 0). **LR policy (v2 change): a single `ReduceLROnPlateau` scheduler replaces the v1 per-phase LR scaling** — starting LR = the v1 default (2e-3), `patience = 10` epochs, factor/min_lr/rel-threshold from config; ONE scheduler instance spans all five phases (no reset and no LR step at phase boundaries). Monitored metric: the validation loss evaluated at the **terminal phase weights** (w_sup = 0.1, w_phys = 1, incl. the roller term) every epoch regardless of the current training phase — this keeps the monitored quantity phase-invariant, so weight ramps cannot fake a plateau or an improvement. **Ramp freeze:** during the two ramp phases (`phys_rampup`, `grnd_rampdown`) the plateau counter is FROZEN — `scheduler.step()` is not called (the terminal metric is still computed and logged every epoch, and best-checkpoint selection still runs); the counter resumes with its pre-ramp value when a constant-objective phase begins, so LR reductions can only be triggered by non-improvement observed in grounding, overlap, or physics. AMP with the physics recompute autocast-exempt; `warm_from` shape-tolerant loader — NOTE the rev-4 input change shrinks what transfers from v1 3-state checkpoints: the input feature projection (`feat` Linear, raw_in 11 → 13) no longer matches, so the loadable subset is the encoder core (in_proj, dt_proj, B_proj, C_proj, A_log, D, norm) + wheel_emb; the γ head AND the feat layer initialize fresh; skipped keys logged and must be exactly {v1 head bank, feat}; scheduler always starts fresh, even when warm-started; reproduce-and-fix gate for the physics-phase launch failure (`runs/sweep_results_phys_integrated_S1.csv` row: FAILED rc=1 @ 0.9 min) on a `--limit-files` dummy run before any campaign launch.
- **Depends on:** `data_v2.py`, `losses_v2.py`, `models_v2.py`

### `evaluation_v2.py` (new)
- **Type:** functions
- **Responsibility:** γ and derived-ω_z metrics only (WZ_P95-normalized ω_z as in v1); slip- and spin-binned tables; emit **`gamma_error_by_slip.csv`** per run (bin_center, n, rmse_norm, rmse_phys per regime/split/wheel) — the consumer contract for A1-v2's `gamma_noise` injection table; roller-residual-of-predictions binned table for comparison against the audit floor.
- **Depends on:** `models_v2.py`, `data_v2.py`, v1 `physics.py`

### `roller_audit.py` (new, standalone, v1-compatible)
- **Type:** script (read-only diagnostic)
- **Responsibility:** Step-0 headroom check run against EXISTING v1 3-state checkpoints (imports v1 `models.py`/`data.py`): evaluate `roller_residual` on (a) ground-truth states and (b) checkpoint predictions, binned by slip speed, per wheel/regime. Ground truth defines the quasi-static floor (~1–3% inertia); the prediction-vs-floor gap in high-slip bins is the recoverable error justifying the campaign. Must numerically reproduce the logged `phys_roller_w*_MON` values on matched batches (this validates the loss wiring v2 will train).
- **Inputs:** `--run-dir` (v1 run), `--regime`, `--cache-dir`, `--bins`
- **Outputs:** `observer_v1_py/report_max_norm/roller_residual_audit.csv`
- **Depends on:** v1 modules only

### `make_observability_report_v2.py` (new, thin)
- **Type:** script
- **Responsibility:** γ-only version of the v1 report (overall + binned RMSE, same-vs-cross gap, ω_z) over the v2 run dirs; reuses v1 report plotting idioms (static matplotlib only).

### `train_observer_v2.py` (new entry point) + `launch_parallel.py` (minimal adapter edit)
- **Responsibility:** CLI mirroring v1 `train_observer.py` (`--regime`, `--window`, `--norm`, `--scaler-csv`, `--cache-dir`, `--w-roller`, `--warm-from`, `--vram`); the launcher adapter gains an `--entry train_observer_v2.py` (or equivalent job field) so the existing Job/heartbeat/resume-skip core is reused unchanged — this is the ONLY permitted edit outside new files.

## 5. File & Directory Structure

```
observer_v1_py/
├── roller_audit.py                    # NEW — step-0 headroom diagnostic (v1-compatible)
├── train_observer_v2.py               # NEW — v2 entry point
├── make_observability_report_v2.py    # NEW — γ-only report
├── launch_parallel.py                 # minimal edit: v2 entry-point support
├── mecanum_observer/
│   ├── config_v2.py                   # NEW — ObserverConfigV2 + phase/roller/sensor knobs
│   ├── sensor_frontend_v2.py          # NEW — IMU synthesis (staged noise), odometry, complementary filter
│   ├── data_v2.py                     # NEW — γ targets; zx/zy → phys aux; sup_weight
│   ├── models_v2.py                   # NEW — WheelObserverV2 (single γ head)
│   ├── losses_v2.py                   # NEW — supervised γ + physics incl. roller
│   ├── training_v2.py                 # NEW — 5-phase runner, 0.1 floor, warm_from
│   ├── evaluation_v2.py               # NEW — γ/ω_z metrics + gamma_error_by_slip.csv
│   ├── physics.py                     # v1, comment-only note
│   └── (all other v1 modules byte-identical)
└── runs/S{1,2}_train_w32_gamma_v2_phys_max_norm/   # v2 campaign run dirs
```

## 6. Key Interfaces

```python
# mecanum_observer/models_v2.py
class WheelObserverV2(nn.Module):
    def forward(self, Gw: Tensor, Pw: Tensor) -> Tensor:
        """
        Args:
            Gw: [B, W, 5]  sensor-real globals (V̂x, V̂y, psi_dot_gyro, a_x, a_y), normalized
            Pw: [B, W, 4, N_PERWHEEL]  per-wheel measurables, normalized
        Returns:
            gamma_hat: [B, 4]  per-wheel roller spin, normalized (max-norm p95)
        """
        ...

# mecanum_observer/losses_v2.py
def supervised_gamma_loss(gamma_hat: Tensor, y: Tensor,
                          sup_weight: Tensor) -> Tuple[Tensor, Dict[str, float]]:
    """
    Args:
        gamma_hat, y: [B, 4] normalized;  sup_weight: [B] per-window weight
    Returns: (scalar, log dict)
    """
    ...

def physics_loss_v2(gamma_hat_phys: Tensor, phys: Dict[str, Tensor],
                    variant: str, w_roller: float, roller_slip_weighting: bool,
                    Minv: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
    """
    Args:
        gamma_hat_phys: [B, 4] predicted roller spin, PHYSICAL units, in-graph
        phys: measurables + mu/chi + zx_lab [B,4], zy_lab [B,4] (label tensors,
              physical units, requires_grad=False) + slip_mag + variant terms
    Returns:
        (scalar loss, log dict) — per-wheel 'phys_roller_w{i}' (trained) plus the
        variant channels. All gradients reach gamma_hat_phys only.
    """
    ...

# mecanum_observer/training_v2.py
def load_warm_start_v2(model: WheelObserverV2, ckpt_path: str) -> List[str]:
    """
    Weights-only load from a v1 3-state checkpoint: encoder core + wheel_emb
    transfer; skipped keys (returned for logging) must be exactly the v1 head bank
    AND the input `feat` projection (raw_in 11 -> 13); γ head + feat init fresh.
    """
    ...

def phase_weights(epoch: int, schedule: List[Tuple[str, int]]
                  ) -> Tuple[float, float]:
    """
    Returns (w_sup, w_phys) for the epoch. NO lr_scale — LR is owned entirely by
    the plateau scheduler in v2. Invariant: w_sup >= 0.1 for all epochs (asserted).
    """
    ...

def terminal_val_loss(model: WheelObserverV2, val_loader, cfg: ObserverConfigV2
                      ) -> float:
    """
    Validation loss at the TERMINAL phase weights (w_sup=0.1, w_phys=1, incl.
    w_roller term), independent of the current training phase. This is the sole
    metric fed to ReduceLROnPlateau.step() and the best-checkpoint selector.
    """
    ...

# roller_audit.py (CLI)
# args: --run-dir --regime --cache-dir --bins
# output CSV columns: source{gt|pred}, regime, split, wheel, slip_bin, n, rms_residual
```

## 7. Data Flow

1. Raw 2000 Hz arrays → `sensor_frontend_v2` (accel channel defined per the
   code-verified EOM convention; FD → anti-alias LPF → decimate; gyro ψ̇; wheel-odometry
   velocity; complementary filter → V̂x, V̂y; corruption only when `noise_stage="real"`)
   → sensor-real Gw at 500 Hz → v1 windowing internals via `data_v2` →
   `(Gw [B,W,5], Pw)` normalized inputs, `y [B,4]` normalized γ target, `phys` dict
   with GROUND-TRUTH `zx_lab`, `zy_lab` (physical), `slip_mag`, `sup_weight`, and the
   variant-specific terms.
2. `WheelObserverV2` → `gamma_hat [B,4]` normalized (shared encoder over the wheel
   axis, single γ head on the final-step representation).
3. Supervised branch: weighted MSE(gamma_hat, y).
4. Physics branch: de-normalize γ̂ (in-graph, p95 scale) → `contact_from_gamma` →
   slip velocities → LuGre force recompute with `(zx_lab, zy_lab, zs=0)` → variant
   channels (wheel+body residual, or integrated one-step) + promoted roller residual.
5. `total = w_sup(phase)·L_sup + w_phys(phase)·(L_variant + w_roller·L_roller)`;
   `w_sup`: 1 → 0.1 over `grnd_rampdown`, held at 0.1 (never 0); `w_phys`: 0 → 1 over
   `phys_rampup`, held at 1. Per epoch: compute `terminal_val_loss` (always) →
   `scheduler.step(metric)` ONLY in constant-objective phases (grounding, overlap,
   physics; skipped during `phys_rampup`/`grnd_rampdown` — ramp freeze) → log the
   resulting LR; LR changes ONLY via the plateau scheduler (patience 10), never at
   phase boundaries and never from ramp-phase transients.
6. Gradient-flow notes: `zx_lab`/`zy_lab` are constructed outside the graph (assert
   `requires_grad == False`); γ̂ de-normalization stays in-graph; the roller
   residual's force terms depend on γ̂ through the slip velocities — that dependence
   IS the training signal, do not detach it; Stribeck negative-slope band admits
   multiple balance roots — the 0.1 supervised floor is the disambiguator.

## 8. Implementation Sequence

1. `roller_audit.py` — v1-only imports; runs against existing 3-state checkpoints;
   go/no-go headroom evidence + numerical reference the v2 loss must reproduce.
2. `config_v2.py` — everything downstream reads it.
3. `sensor_frontend_v2.py` — unit-tests BEFORE anything downstream: (a) dynamics-route
   accel agrees with the FD-of-states cross-check to dense-output tolerance
   (label-independent consistency gate); (b) the accelerometer-observable conversion
   is verified against the sim EOM convention (ω×V transport terms; `run_one.jl`
   authority); (c) stage-1 V̂x/V̂y track sim Vx/Vy after the filter transient; (d) **front-end
   audit** (torch-free, over the cache, BEFORE any training): V̂ error binned by
   slip/profile across candidate crossovers × integrators (Euler vs rotation+AB2)
   × noise stages → selects `vel_filter_crossover_hz`. Acceptance: overall V̂ RMSE
   ≤ 2% of the p95 scale (≈ 0.04 m/s) and ≤ v_str = 0.01 m/s in low-slip bins
   (the gate-boundary region); stress case = spin_creep (sustained slip = weak
   odometry anchor). Note the wrench combos never consume V̂ — tolerance is set by
   the slip/gate features only.
4. `data_v2.py` — γ targets, sensor-real Gw, phys-aux labels, sup_weight (unit-test:
   zx_lab/zy_lab match the columns v1 served as targets, physical units).
5. `models_v2.py` — `WheelObserverV2`; smoke-test shapes; verify v1 checkpoints load
   via `load_warm_start_v2` with exactly {head bank, feat} keys skipped.
6. `losses_v2.py` — verify the roller term reproduces the audit reference values on
   matched batches before it is ever trained.
7. `training_v2.py` — phase-weight invariant (0.1 floor asserted); reproduce and fix
   the physics-phase launch failure (FAILED rc=1 @ 0.9 min in
   `runs/sweep_results_phys_integrated_S1.csv`) on a `--limit-files` dummy run;
   default `physics_variant="residual"` unless the integrated bug proves trivial.
8. `evaluation_v2.py` + `make_observability_report_v2.py` — γ/ω_z reports +
   `gamma_error_by_slip.csv` (A1-v2 consumer contract).
9. `train_observer_v2.py` + `launch_parallel.py` entry support — launch S1+S2 w32,
   warm-started AND from-scratch variants (the pair decides whether the 3-state trunk
   transfers or fights the γ-only objective).

## 9. ML-Specific Considerations

- **Sensor front-end:** single implementation shared by training and deployment (no
  train/deploy divergence); fixed-coefficient causal filter, crossover logged in
  metrics.json; anti-alias LPF always on (both stages); `noise_stage` + corruption
  seeds recorded for reproducibility. Stage-1 (noiseless) results are the
  architecture's upper bound — stage-2 deltas attribute to sensors, not capacity.
- **Batch / step-count interplay:** batch 4096 at 200 epochs halves the optimizer
  steps per epoch relative to the old b2048 runs — a fixed-epoch comparison against
  v1 baselines is therefore step-confounded (the documented A2 batch-ablation lesson);
  compare v2 vs v1 on converged cross-subset metrics, not on same-epoch curves. The
  plateau scheduler's patience (10) is measured in epochs, so at b4096 it also spans
  half the steps — intended, but log steps-per-epoch in metrics.json so this is
  auditable.
- **Numerical stability:** keep the `/ROLLER_SCALE` (0.5 N·m) non-dimensionalization;
  clamp `slip_mag` before any division in weighting terms; γ de-normalization uses the
  frozen p95 scale (gamma 82.81) — never a per-batch statistic.
- **Gradient flow:** all physics gradients reach γ̂ only (z from labels); no detach
  points other than labels-by-construction; supervised floor never scheduled to 0.
- **Device handling:** v1 `precision="auto"` and device-resolution idioms; label
  tensors move with the batch.
- **Mixed precision:** physics recompute (LuGre + residuals) fp32/autocast-exempt —
  bf16 loses the p2·γ term (p2 = 5.78e-3) against O(1) force terms; encoder may stay
  under autocast as in v1.
- **Checkpointing:** v1 run-dir contract (checkpoint.pt, metrics.json, norm.npz,
  split.json, LOSS_AND_NORM.md) + optimizer AND `ReduceLROnPlateau` state (num_bad_epochs,
  best metric, current LR) saved for resume; metrics.json records `model="ssm_v2_gamma"`,
  `w_roller`, phase schedule, scheduler params (patience/factor/min_lr), the per-epoch
  LR trace, and warm_from — so cross-eval tooling can distinguish v2 runs; v1 artifacts
  untouched. Best-checkpoint selection uses `terminal_val_loss` (same metric as the
  scheduler).

## 10. Success Criteria

- [ ] `roller_audit.py` reproduces the logged `phys_roller_w*_MON` magnitudes on
      matched batches (wiring validation) and shows the prediction-vs-ground-truth
      residual gap concentrated in high-slip bins (headroom validation)
- [ ] `WheelObserverV2` 10-batch overfit decreases monotonically; GRU encoder option
      still constructs; v1 test suite / v1 training run unaffected (byte-identical v1)
- [ ] `load_warm_start_v2` skips exactly {v1 head bank, feat projection} (logged)
- [ ] Stage-1 sensor sanity gates pass: synthetic accel matches the sim EOM
      convention; back-calculated wrench matches sim net contact forces; V̂x/V̂y track
      sim Vx/Vy post-transient; γ RMSE additionally reported against a clean-Vx/Vy
      input ablation to quantify the cost of the sensor-real contract
- [ ] Phase weights logged per epoch; `w_sup` reaches exactly 0.1 and never 0
- [ ] LR trace confirms plateau-only behavior: starts at 2e-3, no LR change at any
      phase boundary, no reduction fires during `phys_rampup`/`grnd_rampdown` (ramp
      freeze), reductions occur only after ≥10 non-improving stepped epochs of
      `terminal_val_loss`; scheduler + frozen-counter state survives a
      kill-and-resume test
- [ ] Campaign: cross-subset γ RMSE ≤ v1 baseline (0.056–0.067 norm) overall, and
      high-slip-bin γ RMSE materially below v1's ~0.13–0.15 — the headline metric
- [ ] Trained model's binned roller residual approaches the audit's ground-truth floor
- [ ] `gamma_error_by_slip.csv` emitted per run (A1-v2 noise-table contract)
- [ ] No regression in derived-ω_z (WZ_P95-normalized)

## 11. Out of Scope

- Any change to v1 observer modules, runs, or checkpoints (comparability preserved)
- Any change to A1 / `Mecanum_PINN_Mamba_ForceRecon_v1` (separate brief exists)
- Sim-to-real adaptation mechanics (kinematic pseudo-labels, A1-force roller balance,
  rehearsal buffer) — sim retrain only
- zs / Mz channels; roller inertial term (J_r·γ̇)
- Learned velocity estimation (KalmanNet-style front-end) — fixed complementary filter only in v2; ẇ_i as an input feature (encoders make it cheap; deferred)
- Stage-2 noise calibration (levels from a real IMU datasheet) — the `noise_stage` toggle ships now; the calibrated noisy campaign is follow-up work
- Window/batch re-ablation (w32, max-norm, b2048/b4096 presets are pinned)
