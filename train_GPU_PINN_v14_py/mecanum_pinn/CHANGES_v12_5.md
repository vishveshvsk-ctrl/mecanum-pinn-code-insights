# v12.5 — Factored forward + per-wheel inverse, with zero-init wheel embeddings

This release replaces both the forward and inverse model architectures
with structurally factored, per-wheel processing aligned to the body-frame
EOM. Per-wheel learnable embeddings are added everywhere there is per-wheel
processing, all zero-initialized — this provides a clean sim-to-real
adaptation path without requiring any retraining-from-scratch when wheels
later need to be modeled with individual differences (wear, manufacturing,
mounting).

## Combined parameter accounting

|  | v12 | v12.5 |
|---|---|---|
| forward | ~167,000 | **3,235** |
| inverse | ~82,000 | **1,875** |
| **combined** | **~249,000** | **5,110** |

**~48× total reduction**, achieved by replacing wide MLPs with per-wheel
shared networks and replacing inferential bulk with explicit physical
structure.

## Forward model

Five components with per-wheel embedding broadcast into every per-wheel
head:

| component | shape | params | role |
|---|---|---|---|
| `wheel_embed` | `Embedding(4, 4)`, zero-init | 16 | per-wheel identity, shared across heads |
| `gru_wheel` | shared per-wheel GRU; in 12 dim (embed + 8), hidden 4 | 216 | encoder, integrates roller-spin γᵢ |
| `dec1` | shared per-wheel; in 12 dim (embed + h_wheel + 4), 32×32×32 → 5 | 2,693 | friction-factor head |
| `dec2_linear_F` | `Linear(12, 3, bias=False)` on F | 36 | force-summing M⁻¹ path |
| `dec2_coriolis` | `Linear(3+16, 3, bias=False)` on poly features + H_compound | 57 | Coriolis path |
| `dec2_u_mode` | `Linear(3+16, 3, bias=False)` on torque modes + H_compound | 57 | U-mode path |
| `dec3` | shared per-wheel; in 8 dim (embed + 4), 8 → 1 (no bias) | 80 | ω-velocity head (sim-to-real hedge) |
| `dec_theta` | shared per-wheel; in 8 dim (embed + 4), 8 → 1 (no bias) | 80 | θ head with atan2/12 wrap |

Total: **3,235 params**.

## Inverse model

Per-wheel shared inverse network, called four times with tied weights:

| component | shape | params |
|---|---|---|
| `wheel_embed` | `Embedding(4, 4)`, zero-init | 16 |
| `inverse_net` | per-wheel input 21 dim (use_H=True) or 17 dim (use_H=False), 32 → 32 → 3 | 1,859 (use_H=True) |

Per-wheel input layout (use_H=True):
```
[ embed_i (4),         # zero-init wheel embedding
  H_wheel_i (4),       # split from compound H
  U_i (1),             # commanded torque this wheel
  V (3), ΔV (3),       # platform velocity + change
  ω_i (1), ω_next,i (1),   # wheel angular velocity pair
  Vpx_i (1), Vpy_i (1),    # observed slip components
  sin(12·θ_i) (1), cos(12·θ_i) (1) ]   # 12-fold roller phase
                                          (matches forward's encoding)
```

Total dim: 21 per wheel (or 17 if `use_H=False`).

Total: **1,875 params** with use_H=True (vs v12's 82,188).

### Output bounds — explicitly μ-agnostic

The disk and Mz caps remain at the absolute physical upper bound:

```
|F_i|² ≤ N_i²            (asymptotic soft contraction)
|Mz_i| ≤ mz_bound_factor · N_i
```

**Critically, the disk does NOT scale with μ.** The inverse is a force
*observer* — its job is to read whatever force the wheel is actually
producing from how the state evolved. Conditioning the disk on `mu_b`
would defeat the observer: a sudden μ change (low-friction patch, wear,
contamination) manifests as a sudden |F| change, and we need the observer
free to *report* that change. The forward model carries the μ
conditioning; the inverse reports what it sees, and any disagreement
between them is signal.

## Per-wheel embeddings: design and rationale

A `nn.Embedding(n_wheels, embed_dim)` with `nn.init.zeros_(...)` is added
to both forward and inverse models. The embedding is shared by every
per-wheel head in each model.

### Why share weights at all?

The simulator's friction model is identical for every wheel. Sharing
network weights across wheels matches this symmetry, gives a 4× sample
efficiency advantage, and prevents spurious per-wheel correlations in
training.

### Why add embeddings if sharing is correct?

Real wheels are not identical. Manufacturing tolerances, mounting torque,
bearing wear, surface contact wear histories all introduce per-wheel
differences. Fully shared weights cannot represent any of this; fully
separate per-wheel networks throw away the symmetry that holds at
sim-time.

The hybrid (shared body + per-wheel embedding) is the clean middle:

- **Zero initialization** means at training start every wheel sees the
  same input distribution from the network's perspective. Behavior is
  identical to a fully shared network.
- **At sim-time** the data is wheel-symmetric, so the embedding receives
  zero average gradient and stays near zero. The smoke test on
  wheel-asymmetric random data shows embeddings grow to ~0.17 norm over
  50 steps; on real wheel-symmetric simulator data they should stay near
  zero throughout training.
- **At sim-to-real fine-tuning** the embeddings have capacity to grow,
  capturing per-wheel quirks without any architectural surgery or
  re-training from scratch.

### Cost

Forward: 4 × 4 = 16 params (embed table) + ~240 params (first-layer width
of dec1, dec3, dec_theta, gru_wheel each grew by embed_dim). Total
addition to forward: 256 params.

Inverse: 4 × 4 = 16 params (embed table) + ~128 params (first-layer width
of inverse_net grew). Total addition to inverse: 144 params.

To disable embeddings entirely, set `embed_dim=0` in the config.

## Files changed

### `mecanum_pinn/models.py` — fully rewritten

- `MecanumForwardModel`: factored encoder + 4 decoder heads + per-wheel embedding
- `MecanumInverseModel`: per-wheel shared inverse network + per-wheel embedding
- `physical_diagnostics()` now also returns `wheel_embed` and `wheel_embed_norms` for monitoring whether embeddings stay near zero at sim-time
- `PurePyTorchGRU`, `MecanumPINN`, `maybe_compile_pinn`, `set_grad`, `build_empty_pinn` — interfaces unchanged

### `mecanum_pinn/losses.py` — single change

- Discrete inverse residual: `K = min(k_steps, L - 1)` instead of v12's `min(k_steps, L)`. Fixes a buffer over-run at seq_len = 5 with k_steps ≥ 5. Invisible at seq_len > k_steps.
- Forward physics residual is JVP-based and unchanged from v12.

### `mecanum_pinn/config.py` — defaults reset

| key | v12 | v12.5 |
|---|---|---|
| `hidden_dim` | 128 | 16 |
| `hidden_dim_wheel` | n/a | 4 |
| `dec1_hidden` | n/a (256) | 32 |
| `dec3_hidden` | n/a | 8 |
| `dec_theta_hidden` | n/a | 8 |
| `inv_hidden` | n/a (256) | 32 |
| `embed_dim` | n/a | 4 |
| `seq_len` | 100-125 | 5 |
| `stride` | 20-25 | 1 |
| `k_steps` | 5-10 | 4 |
| `ckpt_dir` | `checkpoints_v12_1` | `checkpoints_v12_5` |
| `figure_dir` | `figures_v12_1` | `figures_v12_5` |

### `mecanum_pinn/training.py` — helper additions

- `_forward_model_kwargs(config)` — forwards new forward-model knobs
- `_inverse_model_kwargs(config, geom, use_H)` — forwards new inverse-model knobs

Used by `train_forward`, `train_inverse_ablation` for consistent
construction. Everything else (phase runners, EarlyStopper, L-BFGS) is
unchanged.

### Files unchanged

`physics.py`, `data.py`, `evaluation.py`, `stages.py`, `trajectory_eval.py`,
`plotting.py`, `manifest.py`, `__init__.py` (version bumped to 12.5.0),
`plot_ood.py`, `make_manifest.py`, `README.md`, `CLI_QUICKREF.md`,
`pinn_training_whitelist.txt`. Top-level `train.py` only had its default
`CKPT_DIR` / `FIGURE_DIR` strings updated.

## Force normalization: F_max / Mz_max as a single source of truth

The algebraic friction reconstruction in `MecanumForwardModel._reconstruct_forces`
produces forces in physical Newtons (via `mu · N_per_wheel · ...`), but
`F_sim` from the dataset is normalized by `force_max`. The original
v12.5 reconstruction returned the physical values directly, creating a
~350× unit mismatch in the `grnd` MSE loss — at initialization the model
saw `(20 − 0.05)² ≈ 400` per element, and minimized it by collapsing
`sigmoid(smoother_raw) → 0`, which then saturated the sigmoid and
froze gradients across the whole forward model. Symptom: state loss
plateauing at 346 from epoch 2 onward, `dec2_linear_F.weight.norm()`
exactly constant.

**Fix:** the reconstruction now divides Fx/Fy by `F_max` and Mz by
`Mz_max`, returning values in the same normalized space as `F_sim`.

**Where the constants live:** `physics.py` is the single source of
truth. Module-level `F_MAX` and `MZ_MAX` (Python floats) are computed
from default robot mass parameters and imported by `data.py` for
trajectory normalization at `__getitem__` time. `RobotParams.finalize()`
also stores `F_max` and `Mz_max` as device-resident torch tensors
computed from the actual (possibly customized) `m` and `m_wheel`.
`MecanumForwardModel` registers them as non-persistent buffers
(`_F_max`, `_Mz_max`) at construction. The chain:

```
physics.py:  F_MAX = (m + 4*m_wheel) * g       ← canonical value (default robot)
            RobotParams.F_max ← finalize()       ← per-instance, respects customization
data.py:    force_max[0:8] = F_MAX               ← imports from physics
models.py:  model.forward._F_max ← buffer        ← set from rp.F_max via constructor
            _reconstruct_forces: Fx /= self._F_max, Mz /= self._Mz_max
```

If anyone customizes `m` or `m_wheel`, `rp.F_max` and the model buffer
update accordingly. The only piece that doesn't auto-update is the
module-level `F_MAX` constant that data.py uses to normalize Arrow
trajectories at import time — that tracks default robot mass. To use a
custom-mass setup you'd need to either re-normalize the dataset or
override `physics.F_MAX` before importing data.py. Documented inline.

**API change:** `build_empty_pinn(config, rp, use_H=True)` now takes a
finalized `RobotParams` instance instead of just `Geometry`. It derives
`geom = make_geometry(rp)` internally. Two callers in `stages.py`
(`_load_forward_model`, `_load_inverse_model`) updated to pass `rp`
instead of `geom`. `training.py`'s `_forward_model_kwargs(config, rp)`
gained the `rp` argument so it can plumb `F_max=rp.F_max,
Mz_max=rp.Mz_max` to the model constructor.

**Verified end-to-end:**

| | before fix | after fix |
|---|---|---|
| F_fwd range at init | ±20 N (physical) | ±0.08 (normalized) |
| initial `state` loss | 3.5e+02 | 3.2e-02 |
| initial `grnd` loss | 2.7e+00 | 5.1e-03 |
| dec2_linear_F.weight movement / 20 steps | 0 (frozen) | 5.2e-03 (training) |

## Vectorization — per-wheel loops collapsed into batched ops

The four `for i in range(n_wheels)` loops in `forward_path` (encoder GRU,
dec1, dec3, dec_theta) and the one in `inverse_path` have been replaced
with vectorized operations that call each per-wheel head exactly once.
The trick: per-wheel inputs are built as `(B, L, n_wheels, feature_dim)`
tensors; MLP heads broadcast naturally over the n_wheels axis (Linear
operates on the last dim). For the GRU we reshape
`(B, L, n_wheels, in_dim) → (B·n_wheels, L, in_dim)`, call once, then
reshape back.

### Numerical equivalence

The output is **bit-identical** to the loop formulation for `F_fwd`, `H`,
and `F_inv`, and matches `S_pred` to 3.7e-9 absolute (float32 reduction-
order noise from `atan2`/`sin`/`cos` being applied to a batched tensor
rather than four separate slices — well below float32 epsilon of 1.2e-7).
This is purely a structural change; model semantics are preserved.

### Measured CPU speedup

| B   | L  | loop (ms) | vec (ms) | speedup |
|-----|----|-----------|----------|---------|
| 8   | 5  | 8.5       | 3.0      | 2.87×   |
| 64  | 5  | 9.6       | 4.3      | 2.26×   |
| 256 | 5  | 12.5      | 9.2      | 1.35×   |
| 256 | 25 | 45.1      | 40.9     | 1.10×   |

Speedup is largest at small batch / short sequence (where Python/kernel
overhead dominates). On GPU expect higher ratios since kernel-launch
overhead (~10 µs each) is far worse than on CPU; collapsing 12+ kernel
launches per forward into 3 has more leverage there. The default config
(`seq_len=5`, batch 256–1024) sits in the regime where vectorization
helps most.

At very long sequences the GRU's internal time loop (still hand-unrolled
for JVP compatibility) becomes the dominant cost, and further speedup
would require either CUDA-graph compilation or a parallel-scan GRU
implementation. Neither is worth the complexity at `seq_len=5`.

### Gradient flow under JVP

Verified after vectorization: 30/30 forward parameters and 7/7 inverse
parameters receive gradient under the full JVP physics residual loss.
JVP traces correctly through `reshape`, `permute`, `expand`, and `cat`
operations — none of these break the tangent propagation.

## Three launch-path bugs fixed alongside the architecture rewrite

`train.py` was calling `run_main(..., ood_enabled=OOD_ENABLED)`, but
`run_main()`'s signature is `(*, config_kwargs, prefix, suffix, p1_wheels,
argv)` — there is no `ood_enabled` parameter. Result: `TypeError` on first
launch. OOD evaluation is intentionally a separate workflow handled by
`plot_ood.py`, which reuses the trained checkpoints. The `OOD_ENABLED`
knob and the argument have been removed from `train.py`; the comment
points users to `plot_ood.py` for OOD runs.

### Bug 2 — pyarrow must import before torch on Windows

On Windows, importing torch before pyarrow can cause a hard crash inside
pyarrow's native loader. The package's `__init__.py` already did
`import pyarrow.feather` before any submodule import, but `train.py`
itself didn't, which meant the order was only correct for `python
train.py`. Notebook usage (`import torch; from mecanum_pinn import ...`)
could still hit the bug. `train.py` now does the pyarrow import at the
very top of the file, before anything else, with a comment explaining
why.

### Bug 3 — `import torch._dynamo` triggered UnboundLocalError

Inside `_set_global_torch_flags(device)` in `stages.py`:

```python
def _set_global_torch_flags(device: torch.device):
    if device.type != 'cuda':
        return
    torch.set_float32_matmul_precision('high')     # <-- uses `torch`
    torch.backends.cudnn.benchmark = True          # <-- uses `torch`
    try:
        import torch._dynamo                       # <-- rebinds `torch` LOCAL
        torch._dynamo.config.cache_size_limit = 16
```

The statement `import torch._dynamo` makes `torch` a *local* name in the
function scope. Python therefore treats every `torch.X` reference in the
function as a local-variable access — including the two lines above that
ran before the import. Result: `UnboundLocalError: cannot access local
variable 'torch' where it is not associated with a value` on every CUDA
launch. The fix is one line:

```python
from torch import _dynamo
_dynamo.config.cache_size_limit = 16
```

This only binds `_dynamo` locally, leaving the outer-scope `torch`
reference untouched. A comment on the line documents why future edits
shouldn't revert it.

## Verified end-to-end on CPU

50-step pinned-batch tests (on synthetic random data, B=8, L=5):

**Forward stage, physics ON (w_phys = 0.1):**
```
step  state     grnd      phys      total     fw_embnorm
0     2.75e+01  1.85e+02  1.08e+04  1.29e+03  0.0040
49    1.20e+01  1.05e+02  2.88e+03  4.05e+02  0.1653
```
Forward total drops 3.2×, physics residual drops 3.75×, all 30 parameter groups receive gradients on both grounding-only and JVP-physics-on backward.

**Inverse stage, physics ON:**
```
step  grnd      cons      phys      total     inv_embnorm
0     2.05e-02  1.02e+02  1.05e-02  2.15e-02  0.0040
49    2.65e-03  1.02e+02  1.05e-02  3.70e-03  0.0831
```
Inverse total drops 5.8×; both `use_H=True` and `use_H=False` variants work.

**Zero-init verified at construction:**
```
forward wheel_embed weight Frobenius norm: 0.000000e+00
inverse wheel_embed weight Frobenius norm: 0.000000e+00
```

## Migration

- v12 checkpoints will not load into v12.5 — state-dict keys differ for both models. Default `ckpt_dir` / `figure_dir` are renamed to `checkpoints_v12_5` / `figures_v12_5` to keep collateral separate.
- `physical_diagnostics()` adds two new keys (`wheel_embed`, `wheel_embed_norms`); old plotting code that iterates the dict will see them but won't break.
- Set `embed_dim=0` in config to fully reproduce a no-embedding ablation.

## Sim-to-real fine-tuning workflow (designed for later)

1. **Train on sim** with full physics residual at `w_phys_max = 1e-1`. Embeddings stay near zero (data is wheel-symmetric).
2. **Collect real-world data** for the trajectories of interest.
3. **Fine-tune on real data** with `w_phys` reduced ~10× (or to zero). Set `lr` ~10× lower than sim training. The architecture lets:
   - dec1 (large, ~2.7k params) absorb friction-law deviations
   - dec3 (small MLP) absorb wheel-shaft nonlinearities
   - wheel_embed (16 params per model) absorb per-wheel asymmetry from wear, mounting, manufacturing
4. **Don't reset embeddings.** They start at zero from sim training (assuming symmetric sim data) and grow only as much as real data forces them to.

The architecture supports all of this without requiring any model-surgery
code path or new model classes.
