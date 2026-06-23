#!/usr/bin/env python3
"""Mecanum-PINN training entry point. THIS FILE IS THE LEVER PANEL.

This script intentionally contains no training logic, no model code, no
plotting code — only the knobs you change between runs. All implementation
lives in the `mecanum_pinn` package next to this file.

Layout of the package (`mecanum_pinn/`):

    physics.py          robot params + geometry + sawtooth helper
    data.py             scaling, Arrow loading, dataset, DataLoader factories
    models.py           GRU + forward + inverse + PINN + compile + builder
    losses.py           physics residuals + autocast + compute_losses
    training.py         epoch loop + phase runners + L-BFGS + top-level trainers
    plotting.py         figure save infra + history / comparison plots
    evaluation.py       test-set + OOD eval + body-frame state plots + mu/chi
    trajectory_eval.py  NEW: world-frame x-y rollout + RMSE-over-time plots
    config.py           build_config + dummy overrides + run-tag builder
    stages.py           CLI parsing + checkpoint helpers + stage runners + run_main

CLI usage:

    python train.py forward                           # train forward only
    python train.py inverse  --ckpt <forward.pth>     # train inverse on top
    python train.py both                              # full pipeline (default)
    python train.py figures  --ckpt <inv_H.pth> --ckpt <inv_NoH.pth>

For dummy/smoke tests: set DUMMY=True below. For CPU-only smoke runs while
the GPU is busy with a real run, set CUDA_VISIBLE_DEVICES="" in your shell
before launching this script.
"""
from __future__ import annotations

# Import pyarrow BEFORE anything that might pull in torch. On Windows,
# importing torch first can cause a hard crash inside pyarrow's native
# loader; this top-of-file import locks in the safe order regardless of
# how the script is launched (CLI, notebook, IDE, etc.).
import pyarrow.feather  # noqa: F401  (load order side effect)

# Editing knobs -------------------------------------------------------
# Set this to your GPU's VRAM in GB. Picks batch_size, seq_len, stride,
# num_workers, prefetch_factor, AND amp_enabled (True for Ampere/Ada, False
# for Quadro RTX 6000/8000 Turing which has no native bf16 path).
VRAM_GB = 24

# Smoke-test mode. True = tiny seq/batch + 2 epochs/phase, no LBFGS, no
# compile (the 30-90s tracing cost wastes most of a smoke run). For
# parallel smoke alongside a real GPU training job, prefer setting the
# env var CUDA_VISIBLE_DEVICES="" instead — runs on CPU, doesn't touch
# the busy GPU.
DUMMY = False

# Data scope. Anything that affects the LEARNED model goes into the
# run_tag automatically; (mu, chi) are external inputs to the architecture
# and so are deliberately NOT encoded in the tag.
MOTION_CASES = ['infinity', 'circle']        # subset of {straightline, sineline, infinity, circle}
MU_VALUES    = [0.5, 0.6]
CHI_VALUES   = [0.000, 0.002]

# Paths.
import os

HOME           = os.path.expanduser("~")
DATA_DIR       = os.path.join(HOME, "mecanum_pinn_main", "data", "SimulationDataSlipSpin_Julia_2")
WHITELIST_PATH = os.path.join(HOME, "mecanum_pinn_main", "code", "pinn_training_whitelist.txt")
CKPT_DIR       = os.path.join(HOME, "mecanum_pinn_main", "data", "checkpoints_v14")
FIGURE_DIR     = os.path.join(HOME, "mecanum_pinn_main", "data", "figures_v14")

# Run-tag pieces. The full tag is "<PREFIX>_motion_<...>_<SUFFIX>".
# Change SUFFIX between repeats of the same scope (run01 / run02 / ...)
# to avoid clobbering checkpoints.
PREFIX = '24GBVRAM'
SUFFIX = 'run01seed42'

# Frozen viscous bearing friction. Read by the physics residual; not
# trainable. Treat as a calibration constant for your wheel hardware.
P1_WHEELS = 0.11

# NOTE: OOD evaluation is NOT a parameter of train.py — it has its own
# entry point (`python plot_ood.py --ckpt <checkpoint>`) that reuses the
# trained checkpoints. This keeps the lever panel clean and decouples
# training from evaluation lifecycle.

# RNG seed — shared by numpy, torch, and the stratified split.
SEED = 42
# --------------------------------------------------------------------


def _main():
    from mecanum_pinn import run_main
    run_main(
        config_kwargs=dict(
            vram_gb=VRAM_GB,
            data_dir=DATA_DIR,
            whitelist_path=WHITELIST_PATH,
            motion_cases=MOTION_CASES,
            mu_values=MU_VALUES,
            chi_values=CHI_VALUES,
            ckpt_dir=CKPT_DIR,
            figure_dir=FIGURE_DIR,
            dummy=DUMMY,
            seed=SEED,
        ),
        prefix=PREFIX,
        suffix=SUFFIX,
        p1_wheels=P1_WHEELS,
    )


if __name__ == '__main__':
    _main()
