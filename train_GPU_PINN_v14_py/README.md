# Mecanum-PINN training (modular)

Modular Python package for training a physics-informed neural network on
mecanum-wheeled robot trajectories with explicit (mu, chi) friction-spin
parameters. The training pipeline produces three model variants — a forward
model + two inverse-model ablations (with-H, without-H) — and a suite of
diagnostic plots.

For task-indexed commands and one-line examples, see `CLI_QUICKREF.md`.
This README explains how the pieces fit together.


## Project layout

```
.
├── train.py                   Top-level launcher. EDIT-ME knobs only.
├── plot_ood.py                Standalone OOD eval. EDIT-ME knobs only.
├── README.md                  This file.
├── CLI_QUICKREF.md            Task-indexed command sheet.
└── mecanum_pinn/              The package — implementation lives here.
    ├── __init__.py            Public API re-exports.
    ├── physics.py             RobotParams, Geometry, sawtooth_approx.
    ├── data.py                Scaling, Arrow loading, dataset, loaders.
    ├── models.py              GRU + forward + inverse + PINN coordinator.
    ├── losses.py              Physics residuals, autocast, compute_losses.
    ├── training.py            Epoch loop, phase runners, L-BFGS, top-level trainers.
    ├── evaluation.py          Test/OOD eval, body-frame plots, mu/chi recovery.
    ├── trajectory_eval.py     World-frame x-y rollout + RMSE-over-time plots.
    ├── plotting.py            Figure save infrastructure + history/comparison plots.
    ├── manifest.py            TOML manifest writer/reader.
    ├── config.py              build_config + dummy overrides + run-tag builder.
    └── stages.py              CLI parsing, ckpt classify/load, run_main entrypoint.
```

You only edit two files between runs: `train.py` for training, `plot_ood.py`
for OOD evaluation. Both are deliberately thin — knobs at the top, single
function call to the package below. Everything else lives inside the package.


## What gets produced

A successful training run drops three things into the project folder:

1. **Checkpoints** at `checkpoints_v12_1/<run_tag>/<stage>_<phase>.pth`.
   For inverse runs, two folders: `<run_tag>_invH/` and `<run_tag>_invNoH/`.
2. **Manifest** at `checkpoints_v12_1/<run_tag>/manifest.toml`. Records the
   exact training scope and the train/val/test split. See "Manifests" below.
3. **Figures** at `figures_v12_1/<run_tag>/figure_NN_<run_tag>_<label>.png`.


## Manifests

Every successful training stage writes a `manifest.toml` next to the .pth
files. It records:

```toml
schema_version = 1
created_at     = "2026-05-06T..."

[stages]
trained          = ["forward"]
forward_ckpt_ref = ""              # set to the forward .pth path for inverse runs

[scope]
motion_cases = ["circle", "infinity"]
mu_values    = [0.5, 0.6]
chi_values   = [0.0, 0.002, 0.005]

[data_source]
data_dir              = '...'
whitelist_path        = '...'
whitelist_total_count = 384
subsample_n           = 0

[split]
seed         = 42
train_count  = 326
val_count    = 38
test_count   = 38
train_names  = [list of trajectory file names]
val_names    = [...]
test_names   = [...]

[config_summary]
seq_len = 100
stride  = 20
batch_size = 128
hidden_dim = 128
tau     = 0.01
k_steps = 5
amp_enabled     = true
compile_enabled = true
lr = 0.001
```

This is the source of truth for "what was trained and on what data." The
figures stage and `plot_ood.py` both read it to reconstruct the
in-distribution test set without re-running `stratified_split` (which would
drift if the whitelist file changes between training and evaluation).

If a checkpoint folder doesn't have a manifest (older runs from before this
refactor), the figures stage falls back to seed-based stratified splitting
and prints a one-line note. OOD evaluation (`plot_ood.py`) requires a
manifest because the held-out scope only makes sense relative to a recorded
training scope.


## Run tags

The run tag is the folder name under `checkpoints_v12_1/`. It encodes the
training scope so different experiments don't collide. Format:

```
<prefix>_motion_<cases>_mu_<values>_chi_<values>_<suffix>
```

Decimal values use `p` instead of `.`, and trailing zeros are preserved at
the precision needed by the most-precise value in the list. Example:

```
6GBVRAM_motion_circle_infinity_mu_0p5_0p6_chi_0p000_0p002_0p005_run01seed42
```

This means motion = {circle, infinity}, mu = {0.5, 0.6}, chi = {0.000,
0.002, 0.005}. Tags can get long (75+ characters). Combined with checkpoint
folder + filename, full paths can approach Windows MAX_PATH (260 chars). If
this becomes an issue, move the project closer to the drive root or enable
long-path support in Windows.

Tags are derived automatically from `config['motion_cases']`,
`config['mu_values']`, `config['chi_values']` plus the `prefix`/`suffix`
strings you pass to `run_main`. You don't have to construct them by hand.


## The four CLI modes

```bash
python train.py forward                # train forward stage from scratch
python train.py inverse  --ckpt PATH   # train both inverse variants on a forward backbone
python train.py both                   # forward + inverse (default)
python train.py figures  --ckpt PATH   # load checkpoint(s), produce figures (no training)
```

`--ckpt` is repeatable. The kind (forward / inverse_H / inverse_NoH) is
auto-detected from each path's parent directory name (specifically the
`_invH` or `_invNoH` suffix).

For OOD evaluation, use the separate top-level script `plot_ood.py` instead
of a CLI flag. Edit the knobs at its top and run it.


## Workflows

**Train from scratch:**

```bash
# Edit VRAM_GB, MOTION_CASES, MU_VALUES, CHI_VALUES, PREFIX, SUFFIX in train.py
python train.py both
```

This produces forward + both inverse variants, plus history/comparison plots
and per-trajectory diagnostics. Three manifests are written.

**Generate figures from existing checkpoints:**

```bash
python train.py figures \
    --ckpt checkpoints_v12_1/<tag>/forward_lbfgs.pth \
    --ckpt checkpoints_v12_1/<tag>_invH/inverse_lbfgs.pth \
    --ckpt checkpoints_v12_1/<tag>_invNoH/inverse_lbfgs.pth
```

Loads all three, reads manifest from the first --ckpt to recover the test
split, and runs the same diagnostics that the training pipeline runs.

**Evaluate on out-of-distribution trajectories:**

```bash
# Edit CKPT_FORWARD, CKPT_INVERSE_H, CKPT_INVERSE_NH paths in plot_ood.py
# Edit HELD_OUT_CHI / HELD_OUT_MOTIONS / HELD_OUT_MU
python plot_ood.py
```

For each OOD axis, evaluates both ID (using the manifest's recorded test
set) and OOD pools, then plots id-vs-ood bar charts with degradation ratios.

**World-frame x-y trajectory rollout:**

This is the new evaluator that complements `plot_test_trajectory_predictions`
(which lives in evaluation.py and shows body-frame velocities + per-wheel
forces vs time). Use it programmatically:

```python
from mecanum_pinn import (RobotParams, make_geometry, build_config,
                          init_torch_globals, build_empty_pinn,
                          load_all_arrow_trajectories, parse_whitelist,
                          configure_figure_saving,
                          evaluate_and_plot_trajectory_window)
from mecanum_pinn.training import load_phase_checkpoint

# ... build cfg, rp, geom, load checkpoint into model ...

result = evaluate_and_plot_trajectory_window(
    model, traj, rp, cfg,
    start_idx=200,                  # initial condition at this index of traj
    window_seconds=6.0,
    initial_pose=(0.0, 0.0, 0.0),   # world-frame anchor (x0, y0, psi0)
)
# result has: true_x, true_y, true_psi, pred_x, pred_y, pred_psi,
#             cum_rmse_x_t, cum_rmse_y_t, cum_rmse_psi_t,
#             F_true, F_fwd, F_inv (all 12-dim per-step force vectors)
```

Two figures are saved: an x-y plane plot with cumulative-RMSE-over-time for
x, y, psi; and a forces plot overlaying sim, forward-model, and
inverse-model per-wheel Fx and Mz tracks.


## Smoke tests + parallel CPU diagnostics

For testing code paths quickly: set `DUMMY = True` in `train.py`. Tiny
seq_len/stride/batch, 2 epochs per phase, no L-BFGS, no `torch.compile`.
Useful when you've changed something in the package and want to verify the
training pipeline still runs end-to-end.

For running diagnostics on CPU while a real run holds the GPU:

```bash
# Linux / macOS:
CUDA_VISIBLE_DEVICES="" python train.py figures --ckpt <...>
CUDA_VISIBLE_DEVICES="" python plot_ood.py

# Windows PowerShell:
$env:CUDA_VISIBLE_DEVICES = ""; python train.py figures --ckpt <...>
$env:CUDA_VISIBLE_DEVICES = ""; python plot_ood.py
```

The empty `CUDA_VISIBLE_DEVICES` hides the GPU from the second process; it
falls back to CPU automatically. `torch.compile` is skipped on non-CUDA
devices, and checkpoints load with `map_location='cpu'`. The two processes
don't share GPU resources.


## Designed dependencies

- Python 3.10+ (3.11+ recommended; we use `tomllib` directly when available)
- PyTorch 2.0+ (`torch.compile`, `torch.func.jvp`)
- numpy, pandas, pyarrow, matplotlib, tqdm
- Optional: `tomli` if you run on Python <3.11 (`pip install tomli`)

No other special libraries. The TOML emitter is hand-rolled in
`mecanum_pinn/manifest.py` to avoid an extra dep on `tomli_w`.


## Where to make changes

- **Different training scope** → edit `train.py` knobs (`MOTION_CASES`,
  `MU_VALUES`, `CHI_VALUES`).
- **Different VRAM tier** → edit `VRAM_GB` in `train.py`.
- **Different epoch counts / loss weights / learning rates** → edit the
  `forward` / `inverse` blocks in `mecanum_pinn/config.py`.
- **Different architecture** → edit dims in `mecanum_pinn/config.py` and the
  modules in `mecanum_pinn/models.py`.
- **New evaluation plot** → add a function to `mecanum_pinn/evaluation.py`
  or `mecanum_pinn/trajectory_eval.py`, call it from `_stage_figures` in
  `mecanum_pinn/stages.py`.
- **New OOD axis** → already supported. Add the held-out values to
  `plot_ood.py`'s `HELD_OUT_*` knobs.
