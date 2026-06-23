# v12.5 — Factored forward + per-wheel inverse, with zero-init wheel embeddings

This release replaces both the forward and inverse model architectures
with structurally factored, per-wheel processing aligned to the body-frame
EOM. Per-wheel learnable embeddings are added everywhere there is per-wheel
processing, all zero-initialized — this provides a clean sim-to-real
adaptation path without requiring any retraining-from-scratch when wheels
later need to be modeled with individual differences (wear, manufacturing,
mounting).

## Combined parameter accounting

|  | v12 | v13 |
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
