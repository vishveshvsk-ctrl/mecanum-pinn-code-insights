# Mecanum ForceRecon Forward Model v2 — Stick/Slip Encoder with Wrench-Restructured Output

> **Generated:** 2026-07-09 (rev 3: sensor-real input contract mirrored from Observer v2 rev 4 — accel sidecars + complementary-filtered V̂, staged noise, single-realization rule; rev 2: forward isolated from the inverse; γ̂ from Observer v2)
> **Stack:** Python 3.13, PyTorch 2.6.0+cu124 (conda `myenv`, RTX 3060 6 GB / Quadro 24 GB), numpy, pyarrow
> **Scope:** Training + model — FORWARD ONLY. Zero dependency on the inverse model; the inverse redesign proceeds in parallel in its own session (see `chat-handoff/inverse_v2_brainstorm_handoff.md`)
> **Design authority:** `Mecanum_PINN_Mamba_ForceRecon_v1/FORWARD_V2_STICKSLIP_DESIGN.md`
> (read it first; this brief is the implementation contract for that design)

## 1. Overview

Rebuild the A1 FORWARD model as a stick/slip-decomposed force estimator. Inputs are
measurable-only sequences (body velocities, wheel speeds/angles, saturated torques)
**plus new IMU-derived channels** (body accelerations, yaw angular acceleration,
wheel accelerations) and a noise-injected roller-spin feature γ̂. Output is the
8-vector of roller-frame contact forces [Fpar_1..4, Fperp_1..4], produced as
**measurement fusion of 6 wrench-pinned combinations + 2 predicted null-space
scalars**, with each wheel's force blended from a stick branch and a slip branch by
a physical gate α. System contract: `(S [B,L,11], U [B,L,4], IMU [B,L,3],
wdot [B,L,4], gamma_hat [B,L,4], mu [B], chi [B]) → F [B,L,8]` physical Newtons,
plus auxiliary outputs (α, ŝ, s1, s2) for losses and diagnostics. The v1 package
(`mecanum_pinn/`) stays intact; v2 lands as parallel modules so v1 runs remain
reproducible. **Isolation (rev 2):** forward v2 has ZERO dependency on the inverse
model — no shared `MecanumPINN` coordinator, no inverse stage in the schedule, no
consistency term — so forward training and the inverse redesign proceed in parallel.
The γ̂ input channel is explicitly **Observer v2's estimated roller spin**
(`WheelObserverV2`, γ-only; `instructions/observer-gamma-only-5phase-retrain.md`),
precomputed offline per trajectory from a designated trained checkpoint;
noise-injected ground truth survives only as an ablation fallback.
**Sensor-real contract (rev 3, mirrors Observer v2 rev 4):** direct Vx/Vy are NOT
inputs — S carries V̂x/V̂y from the fixed complementary filter (crossover + integrator
pinned by `instructions/frontend-drift-audit.md`'s ACCEPTANCE.md), ψ̇ is the gyro, and
the IMU channels are the accelerometer observable built from the exact-dynamics
`accel/` sidecars (`instructions/arrow-accel-augmentation.md`), never finite
differencing. Staged noise `noise_stage ∈ {"none","real"}`, campaign 1 noiseless.
**Single-realization rule:** A1 consumes the IMU twice — encoder features AND measured
wrench combos — so ONE corruption realization per trajectory is shared by both paths.
Sensor-real applies to model INPUTS only; ground truth stays legal on the loss side.

## 2. Architecture Pattern

Two-timescale gated mixture-of-experts over a measurement-fusion output frame:
each wheel runs two diagonal selective SSMs (fast/slip, slow/stick with integrator
channels and carried state), their heads are blended by a monotonicity-constrained
physical gate, and the blended forces are projected onto the wrench-constraint
decomposition (6 measured combos + fixed 2-D null basis). Rationale: slip force is
quasi-static (needs no memory), stick force is an unbounded-memory integrator
observable only through the measured wrench — so memory, gating, and observability
each get a dedicated structural element instead of one generic encoder.

## 3. Technology Constraints

- **Python:** 3.13 (conda `myenv`, call by absolute path; no `conda activate` in tool shells)
- **PyTorch:** 2.6.0+cu124; `torch.compile` guarded off on Windows (no Triton) — reuse v1's `maybe_compile_pinn` idiom
- **Required libraries:** numpy (feature math), pyarrow (Arrow reads via existing loader)
- **Device targets:** CUDA (6 GB laptop tier and 24 GB box tier via the existing `vram_gb` presets); CPU fallback for smoke tests
- **Explicit exclusions:** NO `mamba_ssm`/`causal_conv1d` CUDA kernels (unavailable; scans stay plain-PyTorch loops); NO LuGre/Dahl/GMS friction law inside the model (pinned design decision); NO bristle states (zx, zy, zs) as inputs or supervision for the model trunk; μ, χ, per-wheel forces never inputs; existing decimated-cache idiom must be reused (cache stays normalization-agnostic; pass `--cache-dir` on the laptop); the velocity front-end is the FIXED complementary filter matching `tools_accel/comp_filter.py`'s contract — never learned, single implementation train/deploy; V̂ and stage-2 noise are computed AT LOAD (cache stores clean channels only — filter- and stage-agnostic)

## 4. Component Breakdown

### `imu_features` (module: `mecanum_pinn/imu_features.py`)
- **Type:** functions (numpy, data-pipeline side)
- **Responsibility:** Build the sensor-real channels from the original + `accel/<stem>_accel.arrow` sidecar pair (via the augmentation brief's `load_with_accel` join contract): (a) accelerometer observable from the sidecar's exact-dynamics `dVx, dVy` (transport-term conversion per the code-verified EOM convention) + ψ̈ from `dpsidot` + ẇ from `dw1..4`; (b) anti-alias LPF + decimate to 500 Hz; (c) staged corruption per `SensorNoiseSpec` ONLY when `noise_stage="real"` — drawn ONCE per trajectory and returned as the single arrays that BOTH the feature builder and `wrench.measured_combos` consume (single-realization rule); (d) V̂x/V̂y via the fixed complementary filter, contract-matched to `tools_accel/comp_filter.py` (strapdown mechanization + odometry anchor; crossover/integrator from the drift-audit ACCEPTANCE.md). NO finite differencing anywhere — the sidecar generator already did the FD cross-check.
- **Inputs:** original+sidecar table pair, `SensorNoiseSpec`, `noise_stage`, `vel_filter_crossover_hz`, decim factor, seed
- **Outputs:** `imu [T500, 3]` (a_x, a_y, ψ̈ observable), `wdot [T500, 4]`, `v_hat [T500, 2]`
- **Key constructor params:** `SensorNoiseSpec` (shape shared with A2's `sensor_frontend_v2` spec), `seed: int`
- **Depends on:** `tools_accel/load_with_accel`, `tools_accel/comp_filter.py` contract; called from `data_v2`

### γ̂ sources (`precompute_gamma_hat.py` script + fallback function in `mecanum_pinn/imu_features.py`)
- **Type:** offline script (primary) + function (fallback)
- **Responsibility:** Provide the γ̂ input channel in two modes, selected by `gamma_source`. PRIMARY `"observer_v2"`: run a designated trained Observer-v2 checkpoint (`WheelObserverV2`, γ-only) over every whitelisted trajectory offline — sliding causal w32 windows, batched inference, observer's own normalization — and cache `gamma_hat [T500,4]` per trajectory, keyed by (trajectory, observer run tag). Realistic, error-correlated estimates; this is the deployment-faithful mode and the campaign default. FALLBACK `"gt_noise"` (ablation only): inject noise into ground-truth γ using Observer v2's `gamma_error_by_slip.csv`.
- **Inputs:** primary — trajectory measurables per the observer input contract + observer checkpoint/norm; fallback — `gamma [T500,4]`, `slip_speed [T500,4]`, binned-noise table, seed
- **Outputs:** `gamma_hat [T500, 4]` (cached artifact in the primary mode)
- **Depends on:** `observer_v1_py/mecanum_observer/models_v2.py` (checkpoint inference; read-only import), `wrench.py` slip features. Requires at least one completed Observer-v2 run; until then, development proceeds on the fallback.

### `data_v2` (module: `mecanum_pinn/data_v2.py`)
- **Type:** `Dataset` + loader builders (wraps/extends v1 `data.py`)
- **Responsibility:** Same Arrow→decimated-cache pipeline as v1, extended to (a) emit the new channels (imu, wdot, gamma_hat, plus aux labels Vpx/Vpy and slip speed for losses/eval), (b) serve **burn-in windows**: long sequences of length `L = L_burn + L_loss` (default 384 + 128 @ 500 Hz ≈ 1.02 s) with stride on the loss tail, (c) provide per-window sampling weights that upweight low-|v_s|/low-|w| content.
- **Inputs:** config dict (v2 keys added), regime TOML, whitelist CSV, cache dir
- **Outputs:** batches `(S [B,L,11], U [B,L,4], imu [B,L,3], wdot [B,L,4], gamma_hat [B,L,4], F_sim [B,L,8], aux {vs [B,L,4], vpx0/vpy0 [B,L,4], regime_label [B,L,4]}, mu [B], chi [B])`
- **Key constructor params:** `seq_burn: int`, `seq_loss: int`, `stride: int`, `stick_upweight: float`, `gamma_source: str`, `observer_ckpt: str`
- **Depends on:** `imu_features`, v1 `data.py` internals (decimation, normalization, regime split)

### `wrench` (module: `mecanum_pinn/wrench.py`)
- **Type:** functions + registered-buffer helper class (torch)
- **Responsibility:** All wrench-constraint algebra in one verified place: (a) measured pinned combos — per-wheel drive diagonals `(Msat − Jw·wdot − p1·w)/R` [4] and IMU body-y / yaw net-force rows [2], consuming the SAME `imu`/`wdot` arrays `imu_features` returned (single-realization rule — never a second corruption draw); (b) the fixed roller-frame null basis n1, n2 (per-wheel free diagonals `(sin δ_i, cos δ_i)`, pair-antisymmetric); (c) assembly `F = lift(combos6) + n1·s1 + n2·s2` and its inverse decomposition `F → (combos6, s1, s2)` for loss/eval projections; (d) slip surrogate Vpx0/Vpy0 and γ̂-corrected slip velocity features.
- **Inputs/Outputs:** `combos6 [B,L,6]`, `s [B,L,2]` ↔ `F [B,L,8]` (exact linear bijection); feature builders return `[B,L,4,·]`
- **Key constructor params:** `RobotParams` (reuse v1 `physics.py`)
- **Depends on:** v1 `physics.py` constants; MUST be numerically verified against Arrow labels (reproduce the "VERIFIED residual 0.000" idiom in the module docstring) before anything downstream is built

### `StickSlipEncoder` (class in `mecanum_pinn/models_v2.py`)
- **Type:** `nn.Module`
- **Responsibility:** Per-wheel dual encoder: two v1-style `SelectiveSSM` cores with disjoint τ-band `A_log` inits — slip core (τ ∈ [0.4, 10] ms) and stick core (τ ∈ [50 ms, 5 s] plus `n_integrator` channels with a ≈ 0) — both with **explicit carried-state API** (accept `h0`, return `hT`) so burn-in and streaming inference work.
- **Inputs:** `feats [B,L,4,f_in]`, `h0_slip`, `h0_stick` (each `[B*4, D, N]` or None)
- **Outputs:** `y_slip [B,L,4,D_slip]`, `y_stick [B,L,4,D_stick]`, `(hT_slip, hT_stick)`
- **Key constructor params:** `f_in: int`, `d_model_slip: int`, `d_state_slip: int`, `d_model_stick: int`, `d_state_stick: int`, `n_integrator: int`, `dt_min/dt_max: float`, `tau bands: Tuple[float, float]` per core
- **Depends on:** v1 `SelectiveSSM` (modified: parameterized τ-band init + h0/hT passthrough)

### `SlipHead` (class in `models_v2.py`)
- **Type:** `nn.Module`
- **Responsibility:** Slip-branch force: dimensionless multiplier field with softcircle cap **1.0**, direction anchored to the γ̂-corrected slip-velocity unit vector with a small learned angular correction; output `F_slip = μ_c·N_i·softcircle(m_i)` in roller frame. μ multiplicative (preserves the test-time μ-readout contract).
- **Inputs:** `y_slip [B,L,4,D_slip]`, slip-direction features `[B,L,4,2]`, `mu [B]`, `N_per_wheel [4]`
- **Outputs:** `F_slip [B,L,4,2]` (physical), `m [B,L,4,2]` (diagnostic)
- **Key constructor params:** `in_dim: int`, `hidden: int`, `max_angle_correction: float`
- **Depends on:** `StickSlipEncoder`, `wrench` feature builders, v1 `soft_circle`

### `StickHead` (class in `models_v2.py`)
- **Type:** `nn.Module`
- **Responsibility:** Stick-branch force: reads the stick latent + load-balance features (per-wheel pinned drive combo, body net-wrench features) and predicts deviation-from-load-balance; output bounded by the stiction cap, `‖F_stick,i‖ ≤ 1.1·μ_c·N_i` (softcircle cap 1.1 relative to μ_c·N).
- **Inputs:** `y_stick [B,L,4,D_stick]`, load-balance features `[B,L,4,k]`, `mu [B]`, `N_per_wheel [4]`
- **Outputs:** `F_stick [B,L,4,2]` (physical)
- **Key constructor params:** `in_dim: int`, `hidden: int`, `stiction_ratio: float` (pinned 1.1)
- **Depends on:** `StickSlipEncoder`, `wrench`

### `RegimeGate` (class in `models_v2.py`)
- **Type:** `nn.Module`
- **Responsibility:** Per-wheel α ∈ (0,1): `α = σ(w1·(1−ρ) − w2·ŝ − w3·|v̂_s0|/v_str + b)` with w1..w3 constrained ≥ 0 via softplus reparameterization (structural monotonicity), ρ = ‖F_stick‖/(μ_s·N) breakaway proximity, ŝ the aux log-slip readout, v_str **fixed at 0.01 m/s** (config constant, not a Parameter), b a learnable bias. Also owns the aux readout: `SlipReadout`, a linear head on the slip latent predicting `ŝ ≈ log(|v_s|/v_str)`.
- **Inputs:** `F_stick [B,L,4,2]`, `y_slip [B,L,4,D_slip]`, `vs0_mag [B,L,4]`, `mu [B]`, `N_per_wheel [4]`
- **Outputs:** `alpha [B,L,4]`, `s_hat [B,L,4]`
- **Key constructor params:** `d_slip: int`, `v_str: float`
- **Depends on:** `SlipHead`/`StickHead` outputs (per-step DAG: ŝ → ρ → α → blend; no circularity)

### `NullHead` (class in `models_v2.py`)
- **Type:** `nn.Module`
- **Responsibility:** Predict the two null-space scalars (s1, s2) from the pair-pooled stick/slip latents; output added along the fixed `wrench` null basis. Kept as a **separately addressable module** (own parameter group) so sim-to-real can freeze/anchor it and anchor-event losses can target it.
- **Inputs:** `y_stick [B,L,4,D_stick]`, `y_slip [B,L,4,D_slip]`
- **Outputs:** `s [B,L,2]`
- **Key constructor params:** `in_dim: int`, `hidden: int`
- **Depends on:** `StickSlipEncoder`, `wrench` (basis)

### `MecanumForwardModelV2` (class in `models_v2.py`)
- **Type:** `nn.Module` (coordinator)
- **Responsibility:** Feature build (v1 wheel features ++ IMU/wdot/γ̂/load-balance channels) → dual encoder → branch heads → gate blend → wrench-frame assembly: decompose blended `F_blend` into combos, **fuse** its 6 measured combos toward the sensor-derived combos (learned small-correction residual), re-assemble with `NullHead`'s (s1, s2). Exposes carried-state streaming API and returns all diagnostics. **Conditioning interface (design decision 10):** accepts `mu` as `[B]` (uniform — what training data supplies) OR `[B,4]` (per-wheel — the condition-adaptation hook); broadcast internally; same for `stiction_ratio` if exposed. Also exposes a fusion-bypass flag so TWIN/ROLLOUT mode returns `F_blend` (open-loop force law) — the physics-loss k-step integration MUST consume `F_blend`, never the fused output (fusing measured acceleration into the force that predicts that acceleration trivializes the residual).
- **Inputs:** the full batch contract of §1
- **Outputs:** `F [B,L,8]` physical + `Dict` diagnostics `{alpha, s_hat, s_null, F_slip, F_stick, combos_pred, combos_meas}`
- **Key constructor params:** config dict (v2 keys)
- **Depends on:** everything above

### `losses_v2` (module: `mecanum_pinn/losses_v2.py`)
- **Type:** functions
- **Responsibility:** Compose the training loss on the **loss tail only** (`[:, L_burn:]`): supervised force MSE (F_MAX-normalized, permanent `w_sup ≥ W_SUP_MIN` floor — NO physics-only phase); wrench-measurement MSE (pred combos vs measured combos); gate warm-up BCE vs `α* = exp(−(|v_s|/v_str)²)` with annealed weight; aux slip-readout MSE (ŝ vs log(|v_s|/v_str), clamped); passivity hinge penalty on window-mean `F·v_s`; per-window stick upweighting.
- **Inputs:** model outputs + batch aux labels + phase weights
- **Outputs:** `Dict[str, Tensor]` scalar losses (named, for logging) + weighted total
- **Depends on:** `wrench` (decomposition for the combo loss), config schedule

### `training_v2` (module: `mecanum_pinn/training_v2.py`)
- **Type:** functions (stage runner)
- **Responsibility:** Forward-ONLY training loop — v1's forward→inverse stage sequencing (`stages.py`) is NOT used and no inverse model is constructed anywhere in v2; AMP, grad clip, plateau patience (reuse v1 idioms); phase schedule = grounding (gate warm-up high) → blend (warm-up annealed) → consolidation (all losses at final weights); checkpointing with `_orig_mod.` stripping; per-epoch binned validation via `evaluation_v2`.
- **Depends on:** `data_v2`, `losses_v2`, `MecanumForwardModelV2`

### `evaluation_v2` (module: `mecanum_pinn/evaluation_v2.py`)
- **Type:** functions
- **Responsibility:** Binned reporting — force RMSE by (|v_s|, |w|) bins overall and per component-group (combos vs null coords vs branch attribution), gate calibration (α vs true-regime label), null-coordinate error at detected anchor events (stick↔slip transitions of each pair), and the test-time μ-readout evaluated as forward SELF-CONSISTENCY only (recover the conditioning μ from the slip-branch multiplier basis on F_fwd). μ-ID from reconstructed forces is deferred to the inverse-v2 redesign (own session) — no F_inv appears anywhere in v2 evaluation.
- **Depends on:** `wrench`, model diagnostics

### `config_v2` (module: `mecanum_pinn/config_v2.py`)
- **Type:** config builder (extends v1 `build_config`)
- **Responsibility:** All v2 knobs with pinned defaults: `v_str = 0.01`, `stiction_ratio = 1.1`, τ bands, `n_integrator`, `seq_burn/seq_loss`, sensor knobs (`SensorNoiseSpec`; `noise_stage: {"none","real"}` default `"none"` — campaign 1 noiseless, campaign 2 flips the toggle with no other change; `vel_filter_crossover_hz` + integrator pinned from the drift-audit ACCEPTANCE.md), γ̂ source (`gamma_source: {"observer_v2","gt_noise"}` — default `"observer_v2"`; `observer_ckpt` run-dir path; fallback noise-table path = the observer run's `gamma_error_by_slip.csv`), loss weights/schedules, VRAM tiers (burn-in windows shrink batch: retier for 6 GB).
- **Depends on:** v1 `config.py` (imports and overlays)

### `train_v2.py` (script, package root)
- **Type:** entry-point script
- **Responsibility:** CLI mirroring v1 `train.py` (regime TOML, cache dir, vram tier, run tag) building config_v2 and invoking training_v2; registers with the existing `launch_parallel.py` adapter pattern.

## 5. File & Directory Structure

```
Mecanum_PINN_Mamba_ForceRecon_v1/
├── FORWARD_V2_STICKSLIP_DESIGN.md    # design authority (exists)
├── train_v2.py                        # v2 entry point, forward-only (new)
├── precompute_gamma_hat.py            # offline Observer-v2 γ̂ cache builder (new)
├── mecanum_pinn/
│   ├── imu_features.py                # synthetic IMU + wdot + γ̂ noise (new)
│   ├── wrench.py                      # constraint algebra + null basis + slip features (new)
│   ├── data_v2.py                     # burn-in windows, new channels, stick upweighting (new)
│   ├── models_v2.py                   # StickSlipEncoder, heads, gate, coordinator (new)
│   ├── losses_v2.py                   # composed loss on loss-tail (new)
│   ├── training_v2.py                 # stage runner (new)
│   ├── evaluation_v2.py               # binned reports + anchor-event eval (new)
│   ├── config_v2.py                   # v2 config overlay (new)
│   └── (v1 modules unchanged: models.py, data.py, physics.py, losses.py, ...)
└── runs/                              # v2 run dirs alongside v1 (existing convention)
```

## 6. Key Interfaces

```python
# mecanum_pinn/imu_features.py
def build_sensor_channels(orig: Path, spec: SensorNoiseSpec, noise_stage: str,
                          crossover_hz: float, decim: int, seed: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sidecar-sourced sensor synthesis (load_with_accel join; no finite differencing).
    Corruption (stage "real" only) drawn ONCE — the returned arrays are the single
    realization consumed by BOTH the feature builder and wrench.measured_combos.
    Returns:
        imu:   [T500, 3]  accelerometer observable (a_x, a_y, psi_ddot)
        wdot:  [T500, 4]  wheel accelerations
        v_hat: [T500, 2]  complementary-filtered body velocity (V̂x, V̂y)
    """
    ...

# mecanum_pinn/wrench.py
def measured_combos(U: Tensor, w: Tensor, wdot: Tensor, imu: Tensor,
                    rp: RobotParams) -> Tensor:
    """
    Sensor-side pinned combinations.
    Args:  U [B,L,4] physical Msat; w, wdot [B,L,4]; imu [B,L,3]
    Returns: combos6 [B,L,6] = 4 wheel drive diagonals ++ body-y ++ yaw rows
    """
    ...

def decompose(F: Tensor, rp: RobotParams) -> Tuple[Tensor, Tensor]:
    """F [B,L,8] roller frame -> (combos6 [B,L,6], s [B,L,2]). Exact linear map."""
    ...

def assemble(combos6: Tensor, s: Tensor, rp: RobotParams) -> Tensor:
    """Inverse of decompose. Round-trip must be exact to fp32 tolerance."""
    ...

# mecanum_pinn/models_v2.py
class StickSlipEncoder(nn.Module):
    def forward(self, feats: Tensor,
                h0: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            feats: [B,L,4,f_in] per-wheel measurable features
            h0:    carried states (h_slip, h_stick), each [B*4, D, N] or None
        Returns:
            y_slip:  [B,L,4,D_slip]
            y_stick: [B,L,4,D_stick]
            hT:      (h_slip_T, h_stick_T) for streaming / TBPTT
        """
        ...

class MecanumForwardModelV2(nn.Module):
    def forward(self, S: Tensor, U: Tensor, imu: Tensor, wdot: Tensor,
                gamma_hat: Tensor, mu: Tensor, chi: Tensor,
                h0: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Dict[str, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns:
            F:    [B,L,8] physical roller-frame forces (fused + null-completed)
            diag: {alpha [B,L,4], s_hat [B,L,4], s_null [B,L,2],
                   F_slip [B,L,4,2], F_stick [B,L,4,2],
                   combos_pred [B,L,6], combos_meas [B,L,6]}
            hT:   carried encoder states
        """
        ...

# mecanum_pinn/losses_v2.py
def forward_losses_v2(F: Tensor, diag: Dict[str, Tensor], batch: Dict[str, Tensor],
                      weights: Dict[str, float], burn: int) -> Dict[str, Tensor]:
    """
    All terms computed on [:, burn:] only. Keys of the returned dict:
    'sup', 'wrench', 'gate_warmup', 'aux_slip', 'passivity', 'total'.
    """
    ...
```

## 7. Data Flow

1. Original + `accel/` sidecar pair → v1 decimation path, extended: the decimated cache entry holds the CLEAN observable channels `imu [T,3]`, `wdot [T,4]` plus aux `vs/vpx0/vpy0`, `gamma` (raw) and ground-truth kinematics (loss side); V̂x/V̂y and stage-2 corruption are computed AT LOAD via `build_sensor_channels` (cache stays filter- and stage-agnostic; version bump once). S's body-velocity slots are filled with V̂x, V̂y, gyro ψ̇ — never sim Vx/Vy.
2. `data_v2` windows a trajectory into `[L_burn + L_loss]` sequences; `gamma_hat` comes from the per-trajectory Observer-v2 cache (primary; fixed across epochs — the estimator's errors are what they are) or from fresh-per-epoch `gt_noise` injection (ablation fallback); normalization uses the frozen p95 scaler idiom (new channels get p95 entries appended to the scaler CSV).
3. `MecanumForwardModelV2.forward`: `build_wheel_features_v2` concatenates v1 per-wheel features with imu (broadcast), wdot_i, γ̂_i, and load-balance features from `wrench` → `feats [B,L,4,f_in]`.
4. `StickSlipEncoder` runs both scans (h0 = None in burn-in training; carried at inference) → `y_slip`, `y_stick`.
5. `SlipHead` → `F_slip` (cap 1.0 on multipliers, direction-anchored); `StickHead` → `F_stick` (cap 1.1); `RegimeGate` computes ŝ from `y_slip`, ρ from `F_stick`, then α; blend → `F_blend [B,L,4,2]` → flatten to `[B,L,8]`.
6. `wrench.decompose(F_blend)` → predicted combos + implicit s; the 6 predicted combos are fused toward `measured_combos(...)` (learned residual correction of the measurement); `NullHead` provides (s1, s2); `wrench.assemble` → final `F`.
7. Losses (`losses_v2`, tail only): supervised force vs `F_sim`; wrench MSE (combos_pred vs combos_meas); gate warm-up vs α*(|v_s|); aux ŝ loss; passivity hinge. Weighted sum per the phase schedule (supervised weight never below the floor).
8. Gradient-flow notes: the wrench loss has zero gradient on `NullHead` by construction (null direction) — do NOT "fix" this; α* warm-up targets are computed from aux labels with `no_grad`; ρ uses `F_stick` **without** detach (breakaway proximity should shape the stick head) but the gate warm-up loss applies only to gate parameters (detach ŝ and ρ inputs inside that specific loss term); burn-in prefix runs under the same graph but contributes no loss (do not `detach` mid-sequence in v2.0 — sequences are short enough at L≈512, D·N small).

## 8. Implementation Sequence

1. `wrench.py` — pure algebra, everything depends on it; verify `decompose`/`assemble` round-trip and `measured_combos` against Arrow force labels + noiseless finite-difference IMU on a real file (target: residual ≈ 0, reproducing the v1 physics.py verification idiom). This test doubles as the null-basis correctness proof.
2. `imu_features.py` — numpy-only; PREREQUISITES: `accel/` sidecars present (fleet or pilot; `instructions/arrow-accel-augmentation.md`) and the drift-audit ACCEPTANCE.md crossover/integrator selection (`instructions/frontend-drift-audit.md`). Validate: stage-1 observable channels reproduce sim dynamics; stage-1 `measured_combos` from these channels reproduce the sim net wrench exactly; V̂ tracks sim Vx/Vy post-transient.
3. `precompute_gamma_hat.py` — offline Observer-v2 inference cache (requires one completed Observer-v2 run: checkpoint + `gamma_error_by_slip.csv`; until it exists, wire the `gt_noise` fallback and proceed).
4. `data_v2.py` — burn-in windows + new channels (incl. γ̂ from the cache) + cache-format extension (bump the cache key/version so old .npz entries regenerate cleanly).
5. `models_v2.py`: `StickSlipEncoder` first (τ-band init + h0/hT API; unit-test carried-state equivalence: one long scan == two chained half scans), then heads, gate, `NullHead`, coordinator.
6. `losses_v2.py` — needs model diagnostics dict finalized.
7. `training_v2.py` + `config_v2.py` + `train_v2.py` — forward-only runner and CLI.
8. `evaluation_v2.py` — binned reports and anchor-event eval; wire into per-epoch validation last.

## 9. ML-Specific Considerations

- **Numerical stability:** ŝ target is `log(|v_s|/v_str)` — clamp |v_s| below (e.g. 1e-5 m/s) before the log; softplus-reparameterized gate weights initialized so σ(·) starts near α* statistics; `soft_circle` eps as in v1; the stick SSM's a ≈ 0 channels must use exact `exp(Δ·a)` (no first-order approximation) so integrators don't accumulate drift bias.
- **Gradient flow:** wrench loss → fusion path + trunk only (null direction receives zero — by design); gate warm-up term detaches its ŝ/ρ inputs (trains gate parameters only); burn-in prefix contributes no loss but stays in-graph; passivity hinge uses window-mean (not per-step) products.
- **Device handling:** never hardcode `"cuda"`; follow v1's device-resolution + `set_normalization_torch` pattern; new channels need their torch-side scaler tensors registered the same way.
- **Mixed precision:** AMP as per v1 tiers; the long stick-scan accumulation and `wrench.decompose/assemble` should run in fp32 (autocast-exempt) — integrator channels in fp16 lose micro-slip increments.
- **Checkpointing:** model weights (strip `_orig_mod.`), optimizer, scheduler, epoch, phase, loss-weight schedule position, RNG state for γ̂/IMU noise; run dir layout mirrors v1 (`runs/<tag>/checkpoint.pt`, `metrics.json`); parameter groups named so `NullHead` and `RegimeGate.bias` are individually freezable later (sim-to-real hook).
- **VRAM:** L≈512 windows at batch ~128–256 replace L=5 at 1024 on the 6 GB tier — retier presets and verify no OOM before sweeps; ≤8 workers hard cap.

## 10. Success Criteria

- [ ] `wrench` round-trip exact (fp32 tol) and `measured_combos` residual ≈ 0 vs Arrow labels with noiseless IMU on a real trajectory
- [ ] Carried-state equivalence test passes (one scan == chained scans) for both cores
- [ ] Single-realization test: in stage "real", the IMU/ẇ arrays inside the feature path and inside `measured_combos` are the identical objects/values (no independent noise draws); in stage "none" the two stages differ ONLY in the corruption step
- [ ] Sensor-real ablation reported: campaign metrics vs a clean-Vx/Vy input variant (quantifies the cost of the deployment contract, mirroring the A2 criterion)
- [ ] 10-batch overfit: total loss decreases monotonically; gate does not collapse (α spans (0.1, 0.9) across regimes on the overfit set)
- [ ] Full training: forward force grnd MSE ≤ 0.004 (F_MAX-normalized) overall — i.e. at least matches the v1 *inverse* (~5 N RMSE), vs v1 forward's 0.039–0.046
- [ ] Stick-bin (|v_s| < 0.05 m/s) force RMSE reported and materially below v1 forward's in the same bins
- [ ] Combo channels reconstruct near the injected sensor-noise floor
- [ ] No CUDA OOM at the retiered 6 GB batch size

## 11. Out of Scope

- Inverse-model architecture, training, and the (μ̂, χ̂)-from-F_inv readout — owned by a separate brainstorm session (`chat-handoff/inverse_v2_brainstorm_handoff.md`); nothing in this brief blocks on it or is blocked by it
- Sim-to-real fine-tuning implementation (rehearsal, anchor-event losses, adapters) — architecture hooks only (freezable parameter groups)
- χ identification, Mz / zs channels (dropped by design)
- Online (in-the-loop) Observer-v2 inference during training — γ̂ is precomputed offline to a per-trajectory cache; live streaming wiring is a deployment concern
- The condition-monitoring/adaptation layer itself (per-wheel RLS estimators, CUSUM/GLR detectors, fault-injection campaign — design decision 10): this brief only future-proofs the `[B,4]` μ conditioning interface and the fusion-bypass twin mode
- The PINN velocity observer (design decision 11): v2 trains on the fixed-filter V̂; the estimator swap is a later drop-in behind the same interface
- Any modification to v1 modules, checkpoints, or the shared parallel-launcher core
