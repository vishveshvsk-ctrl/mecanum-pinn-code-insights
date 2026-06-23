# CLI Quick Reference

Task-indexed command sheet. For background and design rationale, see
`README.md`.


## Training

| Goal | Command |
|---|---|
| Train forward stage from scratch | `python train.py forward` |
| Train inverse on saved forward | `python train.py inverse --ckpt PATH/forward_lbfgs.pth` |
| Full pipeline (forward + inverse with-H + inverse without-H) | `python train.py both` |

Defaults to `both` if no stage is given. Edit `VRAM_GB`, `MOTION_CASES`,
`MU_VALUES`, `CHI_VALUES`, `PREFIX`, `SUFFIX` at the top of `train.py`
between runs.


## Diagnostics on existing checkpoints (no training)

| Goal | Command |
|---|---|
| All standard plots from forward only | `python train.py figures --ckpt PATH/forward_lbfgs.pth` |
| All plots from one inverse variant | `python train.py figures --ckpt PATH_invH/inverse_lbfgs.pth` |
| Compare both inverse variants | `python train.py figures --ckpt PATH_invH/inverse_lbfgs.pth --ckpt PATH_invNoH/inverse_lbfgs.pth` |
| Forward + both inverse plots together | three `--ckpt` flags, one per checkpoint |

`--ckpt` is repeatable. The kind (forward / inverse_H / inverse_NoH) is
auto-detected from the parent folder's `_invH` / `_invNoH` suffix.

The figures stage produces (for each kind that's loaded):

- per-stage loss-history curves (train + val per component, log-y)
- train / val / test bar comparison
- per-trajectory body-frame velocity + force overlays (3 test trajectories,
  stratified across motion-case × chi cells)
- mu and chi rolling-mean recovery from inverse-path forces
- per-trajectory least-squares mu/chi recovery (printed table)


## Out-of-distribution evaluation

```bash
python plot_ood.py
```

This is a separate top-level script (not a CLI flag on `train.py`).
Edit at its top:

- `CKPT_FORWARD`, `CKPT_INVERSE_H`, `CKPT_INVERSE_NH` (set to `None` to skip a kind)
- `HELD_OUT_CHI`, `HELD_OUT_MOTIONS`, `HELD_OUT_MU` (empty list = skip that axis)
- `OOD_FRACTION`, `USE_WHITELIST_FOR_OOD`, `SEED`

Reads the manifest (`manifest.toml`) from any of the supplied checkpoint
folders to determine the training scope and the in-distribution test set.
Plots ID-vs-OOD bar charts with degradation ratios for every (axis, model)
combination configured.


## CPU diagnostics while GPU trains

Useful pattern: a long training run is on the GPU; you want to inspect last
night's checkpoints. Force the diagnostic to CPU.

| Shell | Command |
|---|---|
| Linux / macOS | `CUDA_VISIBLE_DEVICES="" python train.py figures --ckpt PATH` |
| Linux / macOS (OOD) | `CUDA_VISIBLE_DEVICES="" python plot_ood.py` |
| Windows PowerShell | `$env:CUDA_VISIBLE_DEVICES = ""; python train.py figures --ckpt PATH` |
| Windows cmd | `set CUDA_VISIBLE_DEVICES= && python train.py figures --ckpt PATH` |

Because the env var hides the GPU, the second process never touches it.
`torch.compile` is skipped on CPU, checkpoints load with `map_location='cpu'`,
and bf16 autocast is bypassed.


## Smoke testing the pipeline

Set `DUMMY = True` near the top of `train.py`, then run normally:

```bash
python train.py both
```

Effects:

- `seq_len = 20`, `stride = 5`, `batch_size = 32` (versus production 100/20/128)
- 2 epochs per phase, no L-BFGS
- 8 trajectories from the whitelist (stratified by motion case)
- `compile_enabled = False` (skips the 30-90 s torch.compile tracing)
- `min_epochs = 1`, `patience = 1` for the early stopper

A forward+inverse smoke run finishes in a few minutes on CPU.


## Where things land

| Artifact | Path |
|---|---|
| Forward checkpoints | `checkpoints_v12_1/<tag>/forward_<phase>.pth` |
| Inverse-with-H ckpts | `checkpoints_v12_1/<tag>_invH/inverse_<phase>.pth` |
| Inverse-without-H ckpts | `checkpoints_v12_1/<tag>_invNoH/inverse_<phase>.pth` |
| Manifest | `checkpoints_v12_1/<tag>/manifest.toml` (one per training run) |
| Figures | `figures_v12_1/<tag>/figure_NN_<tag>_<label>.png` |

`<phase>` is one of: `grounding`, `phys_rampup`, `overlap`, `grnd_rampdown`,
`physics`, `lbfgs`. The full sequence runs in order; `_lbfgs` is the final
checkpoint when L-BFGS refinement is enabled (default).


## Run-tag format reference

```
<prefix>_motion_<cases>_mu_<values>_chi_<values>_<suffix>
```

mu and chi values use `p` instead of `.`, with trailing zeros preserved at
the precision needed by the most-precise value in the list. So:

| Config | Run tag fragment |
|---|---|
| `mu_values=[0.5, 0.6]` | `mu_0p5_0p6` |
| `mu_values=[0.5, 0.55]` | `mu_0p50_0p55` |
| `chi_values=[0.0, 0.002, 0.005]` | `chi_0p000_0p002_0p005` |
| `chi_values=[0.008]` | `chi_0p008` |
| `motion_cases=['circle', 'infinity']` | `motion_circle_infinity` |
| `motion_cases=None` | `motion_all` |
| `motion_cases=[5+ entries]` | `motion_5cases` (collapsed) |

The full scope is also recorded in the manifest, so the tag is for human
inspection — disambiguation between runs always goes through the manifest.


## Common error symptoms and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'mecanum_pinn'` | Folder structure wrong | `mecanum_pinn/` must be a sibling of `train.py`, with `__init__.py` inside |
| `--ckpt: file not found` | Path typo or wrong working dir | Run from the folder containing `train.py`; use the path that includes the run-tag folder |
| `inverse mode needs a FORWARD checkpoint` | Passed an inverse ckpt to inverse mode | Inverse mode TRAINS the inverse stage on top of an existing forward; pass the forward `.pth` |
| `[fatal] no manifest.toml found` (in plot_ood.py) | Older checkpoint, pre-manifest | Re-train OR hand-write a manifest.toml using the schema in `mecanum_pinn/manifest.py` |
| Figures stage prints "no manifest, falling back to seed-based split" | Older checkpoint | Not an error — figures still work, but the test set comes from re-running stratified_split with the recorded seed |
| Tracing pause at first batch (~30-90 s) | Normal `torch.compile` startup cost | Wait it out; subsequent batches are fast. Set `'compile_enabled': False` in config to skip |
| `OneDrive sync conflicts` on .pth files | OneDrive trying to sync mid-write | Pause OneDrive sync during training: right-click cloud icon → Pause syncing → 2 hours |
