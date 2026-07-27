# Observer v3 — Split ΔV Heads (three-head, gradient-isolated readout)

> **Generated:** 2026-07-22
> **Stack:** Python 3.13 (conda `myenv`), PyTorch 2.6.0+cu124, numpy 2.4, pyarrow 24.0
> **Scope:** Model architecture + training variant (no data-pipeline changes)

## 1. Overview

Replace the single two-output ΔV readout in the v2-Hy3 gamma-kin observer with **two
independent single-output heads**, one per body-velocity axis, giving three heads total
(`head_gamma`, `head_dvx`, `head_dvy`). Today `head_dv` maps the concatenated four-wheel
latent `[B, 4*d_model]` to `[B, 2]` through a shared `Linear(128, 32)` block: **98.4% of
its 4,194 parameters (4,128) are shared between the two axes**, and that shared block is
shaped roughly 84.5% by the ΔVy loss. ΔVx therefore reads features it had almost no part
in forming, and it is the measurably weaker axis (73% captured vs 88–92% for y).

v3 gives each axis its own hidden layer and its own supervised loss. Because the two heads
are disjoint subgraphs, gradient isolation is **automatic** — no `.detach()` is required or
wanted. The encoder remains shared and receives all losses; the slip-consistency loss
continues to reach all three heads through `derived_contact_slip`.

System contract is unchanged: `(Gw [B,W,5], Pw [B,W,4,4]) → (gamma_hat [B,4], dv_hat [B,2])`.

**Everything ships as new `*_v3_splitheads.py` files.** `models_v2hy3.py`,
`config_v2hy3*.py`, `training_v2hy3*.py` and `losses_v2hy3.py` are not modified, so the
`_noslip / _slip02 / _wslip1 / _slipLOG / _slipLOG02eq` runs stay bit-reproducible.

## 2. Architecture Pattern

**Shared encoder, disjoint per-target readouts, with one coupling loss.**

The encoder learns a representation serving all targets; each target owns its readout so no
readout is shaped by another target's objective; a single physics-derived term
(slip consistency on |Vp|) deliberately re-couples them at the loss level rather than the
parameter level. This follows the precedent already established in this project — the
gradient-conflict probe measured `cos(∇Lγ, ∇LΔV) ≈ 0` at the shared trunk, which justified
keeping the encoder shared and rejecting GradNorm. v3 extends the same reasoning one level
down: split where objectives differ, share where the representation is common.

Capacity is held fixed so the experiment isolates *separation* from *capacity*.

## 3. Technology Constraints

- **Python:** 3.13 — `C:\Users\vishv\miniforge3\envs\myenv\python.exe`
- **PyTorch:** 2.6.0+cu124. Uses `torch.autocast`, `torch.cuda.amp.GradScaler`,
  `torch.utils.data.DataLoader` (`IterableDataset`). No `torch.compile`, no `torch.func`.
- **Required libraries:** `numpy` (array plumbing) · `pyarrow.feather` (must be imported
  before `torch` on Windows — preserve the existing import order) · `torch`
- **Device targets:** single CUDA device (RTX 3060 Laptop, 6 GB) with CPU fallback. Never
  hardcode `"cuda"`; resolve through the existing device/precision helpers.
- **Explicit exclusions:** no `mamba_ssm`, no `causal_conv1d`, no `einops`, no `lightning`,
  no KAN library — none are installed. The encoder stays `MambaLiteSSM` (pure PyTorch).
- **Concurrency:** ≤2 training processes (Windows commit-limit ceiling, not VRAM).

## 4. Component Breakdown

### `ObserverConfigV3SplitHeads`
- **Type:** `@dataclass`, subclass of `ObserverConfigV2Hy3GammaKin`
- **Responsibility:** carry the v3 head-split and per-axis loss-weight settings while
  inheriting every gamma-kin field (`gamma_residual`, `dgamma_scale`, `dv_scale`,
  `vy_label_*`, `slip_loss_kind`, `slip_log_eps`, `use_vp_components`, gate params).
- **Inputs:** none (dataclass)
- **Outputs:** validated config via `resolved()`
- **Key constructor params (new only):**
  - `dv_head_hidden: int` — hidden width of each ΔV head. Default **16** (memory-neutral).
  - `w_dvx: float`, `w_dvy: float` — explicit per-axis loss weights. Defaults **0.5 / 0.5**,
    which reproduces the v2hy3 `mean(dim=-1)` averaging exactly.
  - `dv_axis_normalize: bool` — when True, divide each axis loss by a frozen constant so the
    weights express the true gradient split. Default **False**.
  - `dv_axis_norm_x: float`, `dv_axis_norm_y: float` — those frozen constants; only read when
    `dv_axis_normalize` is True.
- **Depends on:** `ObserverConfigV2Hy3GammaKin`
- **Validation in `resolved()`:** call `super().resolved()`; require `dv_head_hidden > 0`;
  require both `w_dvx, w_dvy >= 0` and not both zero; when `dv_axis_normalize`, require both
  norm constants strictly positive.
- **`run_tag` property:** override so the default tag is distinguishable from gamma-kin runs
  (suffix `_v3split`), while `run_tag_override` still wins.

### `WheelObserverV3SplitHeads`
- **Type:** `nn.Module`, subclass of `WheelObserverV2Hy3`
- **Responsibility:** reuse the inherited `wheel_emb`, `feat`, `encoder` and `head_gamma`
  unchanged; delete the inherited `head_dv` and replace it with two independent heads.
- **Inputs:** `Gw: Tensor [B, W, 5]`, `Pw: Tensor [B, W, 4, N_PERWHEEL]` (both normalised)
- **Outputs:** `gamma_hat: Tensor [B, 4]` (normalised), `dv_hat: Tensor [B, 2]`
  (normalised by `dv_scale`) — **identical signature to v2hy3**
- **Key constructor params:** `cfg: ObserverConfigV3SplitHeads`
- **Submodules replaced:**
  - `head_dvx`: `Linear(N_WHEELS*d_model, dv_head_hidden) → SiLU → Linear(dv_head_hidden, 1)`
  - `head_dvy`: same shape, separate parameters
  - the inherited `head_dv` attribute must be removed, not merely unused, so it cannot appear
    in `state_dict()` and silently bloat checkpoints.
- **Depends on:** `WheelObserverV2Hy3`, `MambaLiteSSM`, `ObserverConfigV3SplitHeads`

**Critical design choice — concatenate inside `forward`.** The two heads each emit `[B,1]`;
`forward` joins them into `dv_hat [B,2]` before returning. Concatenation is a differentiable
join, so gradients still flow back only to the originating head — isolation is preserved —
while every downstream consumer (`derived_contact_slip`, `physics_loss_hy3`, `_gamma_terms`,
`assemble_gamma`, all eval scripts) continues to work unmodified against a `[B,2]` tensor.
Per-axis losses are then taken as slices of `dv_hat`. This is what keeps the v3 diff small.

**Parameter budget (d_model=32, head_hidden=32, dv_head_hidden=16):**

```
                     v2hy3 gamma-kin                v3 split
head_gamma           1,089   (unchanged)            1,089
head_dv              4,194                          removed
head_dvx                                            2,081
head_dvy                                            2,081
        head total   5,283                          5,251     (-32, -0.6%)
        model total  8,357                          8,325
```

### `build_model_v3`
- **Type:** `function`
- **Responsibility:** construct `WheelObserverV3SplitHeads` from a v3 config.
- **Inputs:** `cfg: ObserverConfigV3SplitHeads`
- **Outputs:** `WheelObserverV3SplitHeads`
- **Depends on:** `WheelObserverV3SplitHeads`

### `load_warm_start_v3`
- **Type:** `function`
- **Responsibility:** weights-only transfer from a v2hy3/gamma-kin checkpoint into a v3
  model — carry `feat`, `encoder`, `wheel_emb`, `head_gamma`; skip all `head_dv.*` keys
  (shape-incompatible with the split heads, which start fresh).
- **Inputs:** `model: WheelObserverV3SplitHeads`, `ckpt_path: str`
- **Outputs:** `List[str]` of skipped keys (for logging, mirroring `load_warm_start_v2hy3`)
- **Depends on:** `WheelObserverV3SplitHeads`

### `train_v3_splitheads`
- **Type:** `function` (training entry)
- **Responsibility:** near-copy of `train_v2hy3_gammakin` with the ΔV loss computed per axis;
  every other behaviour (5-phase plan, `vy_label` ramp, ramp-freeze of scheduler/ES/best-ckpt,
  cosine diagnostics, slip-loss dispatch, `sf`/`slip_rmse` logging) carried over unchanged.
- **Inputs:** `cfg: ObserverConfigV3SplitHeads`
- **Outputs:** none (writes run directory)
- **Depends on:** `build_model_v3`, `load_warm_start_v3`, `data_v2hy3`, `losses_v2hy3`,
  `physics_v2hy3`, and the shared helpers already imported from `training_v2hy3`
  (`_cfg_dict`, `_phys_batch`, `_resolve_precision`, …)

### `_slip_loss_gk` (reuse)
- **Type:** `function` — already exists in `training_v2hy3_gammakin.py`
- **Responsibility:** dispatch slip loss on `cfg.slip_loss_kind`.
- **Note:** import and reuse; do not duplicate. It already reads `slip_loss_kind` /
  `slip_log_eps` via `getattr` with safe defaults, so a v3 config satisfies it unchanged.

### `train_observer_v3_splitheads` (CLI)
- **Type:** `script`
- **Responsibility:** argparse → `ObserverConfigV3SplitHeads` → `train_v3_splitheads`.
- **New flags beyond the gamma-kin CLI:** `--dv-head-hidden`, `--w-dvx`, `--w-dvy`,
  `--dv-axis-normalize`, `--dv-axis-norm-x`, `--dv-axis-norm-y`.
- **Startup banner:** extend the existing `[cli-gammakin]` line (retag `[cli-v3split]`) to
  print `dv_head_hidden`, `w_dvx`, `w_dvy` and the total parameter count, so a log immediately
  identifies the architecture it came from.

### Eval-tooling dispatch (modification, not new file)
- **Type:** edits to `regime_split_attrib.py` (and any sibling eval script that rebuilds a cfg
  from `metrics.json` — check `eval_hy3_endpoints.py`, `roller_audit.py`, `calibrate_gate.py`)
- **Responsibility:** detect a v3 run and rebuild the correct config *and* model class.
- **Why this is mandatory:** `_is_gammakin()` returns True for a v3 checkpoint (it subclasses
  gamma-kin, so `gamma_residual` is True). Without a v3 branch the loader would rebuild an
  `ObserverConfigV2Hy3GammaKin`, silently drop `dv_head_hidden`, call `build_model_v2hy3`, and
  fail on a `state_dict` mismatch — or worse, load partially. Add `_is_v3(m)` keyed on the
  presence of `split_dv_heads`/`dv_head_hidden` in `m["cfg"]`, checked **before** the gamma-kin
  branch, and route to `ObserverConfigV3SplitHeads` + `build_model_v3`.

## 5. File & Directory Structure

```
code_insights/observer_v1_py/
├── mecanum_observer/
│   ├── config_v3_splitheads.py        # NEW  ObserverConfigV3SplitHeads
│   ├── models_v3_splitheads.py        # NEW  WheelObserverV3SplitHeads, build_model_v3,
│   │                                  #      load_warm_start_v3
│   ├── training_v3_splitheads.py      # NEW  train_v3_splitheads
│   │
│   ├── config_v2hy3_gammakin.py       # UNCHANGED (v3 subclasses it)
│   ├── models_v2hy3.py                # UNCHANGED (v3 subclasses WheelObserverV2Hy3)
│   ├── training_v2hy3_gammakin.py     # UNCHANGED (v3 imports _slip_loss_gk from it)
│   ├── training_v2hy3.py              # UNCHANGED (shared helpers)
│   ├── losses_v2hy3.py                # UNCHANGED (slip_consistency_loss{,_log})
│   ├── data_v2hy3.py                  # UNCHANGED (use_vp_components already supported)
│   └── physics_v2hy3.py               # UNCHANGED
│
├── train_observer_v3_splitheads.py    # NEW  CLI entry point
└── regime_split_attrib.py             # EDIT add _is_v3 dispatch (see §4)
```

## 6. Key Interfaces

```python
# ---- config_v3_splitheads.py -------------------------------------------------
# @dataclass
# class ObserverConfigV3SplitHeads(ObserverConfigV2Hy3GammaKin)

def resolved(self) -> "ObserverConfigV3SplitHeads":
    """Validate v3 fields, then delegate to the gamma-kin validator.

    Raises:
        ValueError: dv_head_hidden <= 0; negative or all-zero axis weights;
                    non-positive axis normalizers when dv_axis_normalize is True.
    """
    ...

@property
def run_tag(self) -> str:
    """Default run tag, suffixed to distinguish v3 runs. run_tag_override wins."""
    ...


# ---- models_v3_splitheads.py -------------------------------------------------
# class WheelObserverV3SplitHeads(WheelObserverV2Hy3)

def __init__(self, cfg: ObserverConfigV3SplitHeads) -> None:
    """Build the v2hy3 stack, then remove head_dv and install head_dvx/head_dvy.

    The inherited head_dv must be deleted from the module tree, not just left
    unused, so it does not enter state_dict().
    """
    ...

def forward(self, Gw: Tensor, Pw: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        Gw: sensor-real globals (V_hat_x, V_hat_y, psi_dot, a_x, a_y)  [B, W, 5], normalised
        Pw: per-wheel measurables [Msat, w, sin_tt, cos_tt]            [B, W, 4, N_PERWHEEL]
    Returns:
        gamma_hat: per-wheel roller spin, normalised                   [B, 4]
        dv_hat:    body-velocity correction, normalised by dv_scale    [B, 2]
                   column 0 from head_dvx, column 1 from head_dvy;
                   joined by concatenation so each column's gradient
                   reaches only its own head.
    """
    ...

def build_model_v3(cfg: ObserverConfigV3SplitHeads) -> WheelObserverV3SplitHeads:
    """Construct the v3 observer."""
    ...

def load_warm_start_v3(model: WheelObserverV3SplitHeads, ckpt_path: str) -> List[str]:
    """Transfer feat/encoder/wheel_emb/head_gamma from a v2hy3 or gamma-kin
    checkpoint; skip every head_dv.* key (split heads start fresh).

    Returns:
        Names of source keys that were skipped.
    """
    ...


# ---- training_v3_splitheads.py ----------------------------------------------

def _dv_axis_losses(dv_hat: Tensor, y_dv: Tensor,
                    cfg: ObserverConfigV3SplitHeads) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-axis supervised ΔV loss.

    Args:
        dv_hat: predicted correction, normalised   [B, 2]
        y_dv:   target correction, normalised      [B, 2]
    Returns:
        l_dv:   weighted scalar total used in the objective   []
        se_x:   unweighted per-sample squared error, axis x   [B]
        se_y:   unweighted per-sample squared error, axis y   [B]

    se_x / se_y are returned unweighted so the epoch logger reports a metric
    comparable across w_dvx / w_dvy settings and against the v2hy3 runs.
    With w_dvx = w_dvy = 0.5 and dv_axis_normalize False, l_dv equals the
    v2hy3 `((dv_hat - y_dv) ** 2).mean(dim=-1).mean()` exactly.
    """
    ...

def train_v3_splitheads(cfg: ObserverConfigV3SplitHeads) -> None:
    """Run the 5-phase curriculum with three-head readout. Writes checkpoints,
    metrics.json, norm.npz, split.json into the run directory."""
    ...
```

## 7. Data Flow

**Forward pass (unchanged up to the readout):**

1. `Gw [B,W,5]` and `Pw [B,W,4,4]` arrive normalised from `WindowDatasetHy3`.
2. Globals are broadcast across the wheel axis and concatenated with per-wheel measurables
   and the wheel embedding → `[B, W, 4, 5+4+emb_dim]`.
3. Permute and fold the wheel axis into the batch → `[B*4, W, in_dim]`.
4. `feat` lifts to `d_model`; `MambaLiteSSM` scans causally → `rep [B*4, d_model]`.
5. `head_gamma` applies per-wheel (weight-tied) → `gamma_hat [B, 4]`. **Unchanged.**
6. `rep` is reshaped to `[B, 4, d_model]` and flattened to `H_cat [B, 4*d_model]`.
7. `head_dvx(H_cat) → [B,1]` and `head_dvy(H_cat) → [B,1]`, computed independently.
8. Concatenate along dim 1 → `dv_hat [B, 2]`.

**Loss chain:**

9. `_gamma_terms` de-normalises γ in-graph, builds the `γ_noslip` base from
   `V_y_used = V_hat_y + (lam*dV_true + (1-lam)*dV_hat)_y * dv_scale`, and returns
   `gh_norm`, the Δγ squared error, and the p95-normalised γ error for logging.
10. `l_gamma` is the supervision-weighted mean of the Δγ error.
11. `_dv_axis_losses` slices `dv_hat` and forms the weighted per-axis ΔV loss `l_dv`.
12. `derived_contact_slip(gh_norm, dv_hat, …)` builds |Vp| from the *joined* `dv_hat`, so the
    slip term reaches `head_gamma`, `head_dvx` and `head_dvy` simultaneously — this is the
    only place the three heads are coupled.
13. `_slip_loss_gk` dispatches to the linear or half-square-log slip loss on
    `cfg.slip_loss_kind`.
14. Supervised total = `l_gamma + l_dv + w_slip * l_slip`, scaled by the phase `w_sup`;
    the physics term (when `w_phys > 0`) adds on top, exactly as in gamma-kin.

**Gradient flow:**

- `head_dvx` and `head_dvy` are disjoint subgraphs. `∂l_dv_y/∂head_dvx = 0` holds structurally
  — **do not add `.detach()` anywhere to "enforce" it.** A detach here would also sever the
  slip loss from the heads, which is the one coupling the design intends to keep.
- `gamma_base_detach=True` is inherited and stays True: it cuts the γ→ΔV path through the
  `γ_noslip` base, which only becomes active once `vy_label` drops below 1.
- The `w_slip == 0` branch must keep computing `v_slip` under `torch.no_grad()`. A
  grad-tracked `l_slip` that is never backpropped retains its graph and OOMs at batch 4096.
- The per-epoch `sf` / `vp_rmse` / detached-linear-slip diagnostics must stay inside
  `torch.no_grad()` with `v_slip.detach()`, for the same reason.

## 8. Implementation Sequence

1. **`config_v3_splitheads.py`** — no dependencies; every later component needs the config
   fields and the `resolved()` contract.
2. **`models_v3_splitheads.py`** — needs #1. Verify the parameter count lands at 8,325 and
   that `state_dict()` contains no `head_dv.*` key before proceeding.
3. **`load_warm_start_v3`** (same file as #2) — needs #2's module tree to know which keys to
   skip.
4. **`training_v3_splitheads.py`** — needs #1–#3. Port from `training_v2hy3_gammakin.py`,
   changing only the ΔV loss block and the logging line; import `_slip_loss_gk` rather than
   re-implementing it.
5. **`train_observer_v3_splitheads.py`** — needs #1 and #4.
6. **`regime_split_attrib.py` dispatch** — needs #1 and #2 to exist as importable names. Must
   land before any v3 run is evaluated, or the eval fails on a `state_dict` mismatch.
7. **Reproduction check** — needs #1–#5. See §10, criterion 1. Run before any new experiment.

## 9. ML-Specific Considerations

- **Numerical stability:** none of the v3 changes touch the log-domain slip loss, but its
  guarantees must survive the port — `slip_log_eps` floors the log argument (vpm contains
  exact zeros), and worst-case gradient scales as 1/eps. Do not alter `losses_v2hy3.py`.
- **Gradient flow:** no `.detach()` is to be added for head isolation (see §7). Preserve
  `gamma_base_detach`, the `no_grad` slip branch, and detached diagnostic reductions.
- **Device handling:** resolve device and AMP dtype through the existing
  `_resolve_precision(cfg, dev)` helper; never hardcode `"cuda"`. All scale tensors
  (`gamma_std_t`, `gamma_mean_t`, `dv_scale_t`, `vhat_scale_t`) are built on the resolved
  device — extend the same pattern, do not introduce new host-side constants in the inner loop.
- **Mixed precision:** the model forward runs under `torch.autocast`; head outputs are cast
  back with `.float()` immediately after, before any de-normalisation. Keep that cast — the
  in-graph de-normalisation (γ ×82.8, ΔV ×1.2967) and the |Vp| construction must be fp32.
  `GradScaler` stays enabled only for fp16.
- **Checkpointing:** save `model`, `opt`, `scheduler`, `epoch`, `_cfg_dict(cfg)`,
  `best_metric`, `lr_trace`, `cosine_trace` — the existing schema. `_cfg_dict` uses
  `dataclasses.asdict`, so the v3 fields serialise automatically; this is what the `_is_v3`
  eval dispatch keys on, so do not filter the cfg dict on save.
- **Memory:** batch 4096 × stride 16 is the tested footprint (~3.2 GB). v3 is 32 parameters
  smaller, so the footprint is unchanged; keep ≤2 concurrent training processes.
- **Reproducibility of prior work:** the v2hy3 and gamma-kin modules must show a zero-line
  diff after this task. If anything in them needs to change, stop and raise it rather than
  editing.

## 10. Success Criteria

- [ ] **Reproduction (highest priority):** with `w_dvx = w_dvy = 0.5`,
      `dv_axis_normalize = False`, `dv_head_hidden = 32`, and a fixed seed, a short v3 run
      (≤3 epochs, `--limit-files 40`) matches the equivalent gamma-kin run's `g_mse`, `dv_mse`
      and `slip_rmse` to within float noise. This proves the split changed only *where*
      parameters live, not *what* the objective is. (Run at hidden 32, not 16, so the only
      difference under test is the split itself.)
- [ ] `git diff --stat` reports **zero changes** to `models_v2hy3.py`,
      `config_v2hy3*.py`, `training_v2hy3*.py`, `losses_v2hy3.py`, `data_v2hy3.py`.
- [ ] Parameter count is **8,325** at `d_model=32, head_hidden=32, dv_head_hidden=16`, and
      `state_dict()` contains `head_dvx.*` / `head_dvy.*` and **no** `head_dv.*`.
- [ ] `forward` returns `gamma_hat [B,4]`, `dv_hat [B,2]` — unchanged shapes, so
      `derived_contact_slip` and `physics_loss_hy3` run without modification.
- [ ] Gradient isolation verified numerically: backward on an x-only loss leaves every
      `head_dvy` parameter with zero (or `None`) grad, and vice versa.
- [ ] `--cosine-diag` reports `dvx` and `dvy` as separate tasks, giving
      `cos(∇L_dvx, ∇L_dvy)` at the shared trunk as a byproduct of the first real run.
- [ ] `regime_split_attrib.py --run <v3_run> --use-vp-components` loads the checkpoint and
      writes a `*_vpc.json` without a `state_dict` mismatch.
- [ ] No CUDA OOM at batch 4096, stride 16, two concurrent processes.
- [ ] Loss decreases monotonically on a 10-batch overfit test.

## 11. Out of Scope

- **Changing the loss shape on ΔV or γ.** MSE stays. The relative/log/arcsinh question was
  investigated and closed: γ is not tail-dominated (`max/p95 = 1.68`, 9.0% of gradient in the
  top 1%), and ΔV's real failure is at *large* values where MSE already looks — while |Vx|
  correlates **negatively** with |Vp| (−0.442), so a relative loss would weight the
  gate-irrelevant samples.
- **Any change to `head_gamma`.** It stays per-wheel and weight-tied; `γ_noslip_i` depends
  only on wheel *i*'s own geometry plus shared body `(V_y, Ω)`, so the tied structure is the
  correct relabeling-equivariant bias.
- **Pooling the four wheel latents** in place of concatenation (would cut the ΔV heads to
  ~1,090 params) — requires explicit per-wheel geometry first, since concatenation is
  currently what preserves the asymmetric O-configuration. Separate task.
- **KAN / KAE lift.** Sited at `self.feat`, not the heads; specced separately in
  `session_summaries/observer_encoder_kan_equivariance_sparsity.md`.
- **The `V_hat_y` geometric clamp.** A front-end defect (estimate exceeds the physical
  envelope on 2.994% of samples, up to 211.32 cm/s against a 115.94 cm/s max) and the
  highest-value open item — but a `sensor_frontend_v2.py` change, not a head change.
- **Choosing the production `w_dvx : w_dvy`.** v3 only makes the knob explicit. Note for
  whoever tunes it: equal *weights* do not give an equal *gradient split* — the measured
  init split is 15.5% x / 84.5% y and is driven by target magnitudes, so a 50/50 split needs
  `w_dvx ≈ 5.45 × w_dvy`, or `dv_axis_normalize=True` with measured constants.
- **Encoder, data pipeline, `dv_scale`, and the 5-phase curriculum** — all unchanged.
