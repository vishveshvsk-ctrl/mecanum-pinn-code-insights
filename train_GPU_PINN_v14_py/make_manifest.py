#!/usr/bin/env python3
"""Recover manifest.toml for pre-manifest checkpoints.

Older checkpoints don't have a manifest.toml. This script reconstructs one
by re-running the same stratified_split with the recorded seed against the
same trajectory pool, then writing the result alongside the .pth files.

Run this ONCE per pre-manifest training run. After that, plot_ood.py and
the figures CLI mode work the same as for new checkpoints.

Usage:
    1. Edit the knobs below for one training run.
    2. python make_manifest.py
    3. (optional) verify with `cat checkpoints_v12_1/<tag>/manifest.toml`
    4. plot_ood.py / `python train.py figures --ckpt ...` now work normally.

The script auto-discovers the three sibling folders (<base>, <base>_invH,
<base>_invNoH) and writes a manifest into each one that exists. Missing
folders are silently skipped.

LIMITATION
----------
The split is deterministic from (trajectory pool, seed). The recovered
split matches the original ONLY if the data_dir contents AND whitelist
contents are identical to what they were at training time. If trajectories
have been added to or removed from the data dir, or the whitelist has
been edited, the recovered split will not match. There is no way to fix
this without retraining; the best you can do is treat the recovered
manifest as best-effort and note the caveat in any results that depend on
the ID/OOD partition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mecanum_pinn.data import (load_all_arrow_trajectories, parse_whitelist,
                                stratified_split)
from mecanum_pinn.manifest import save_training_manifest


# ============================================================
# KNOBS  --  edit these for one training run, then `python make_manifest.py`
# ============================================================

# The forward-folder name under CKPT_DIR (NO _invH / _invNoH suffix).
# Example: if your forward checkpoint is at
#   checkpoints_v12_1/myrun_motion_circle_run01seed42/forward_lbfgs.pth
# then BASE_RUN_TAG = 'myrun_motion_circle_run01seed42' and
# CKPT_DIR = Path('checkpoints_v12_1').
CKPT_DIR     = Path(r"checkpoints_v12_1")
BASE_RUN_TAG = "PUT_THE_FORWARD_FOLDER_NAME_HERE"

# Training scope. Required: old run_tags don't encode mu/chi.
MOTION_CASES = ['circle', 'infinity']
MU_VALUES    = [0.5, 0.6]
CHI_VALUES   = [0.000, 0.002, 0.005]

# Data source.
DATA_DIR       = r"G:\My Drive\SimulationDataSlipSpin_Julia_2"
WHITELIST_PATH = r"G:\My Drive\pinn_training_whitelist.txt"

# Split. Use the seed from the original training. 42 is the default.
SEED        = 42
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10

# Architectural knobs to record in config_summary. These are read by the
# figures stage and plot_ood.py to set seq_len/stride/etc when rebuilding
# the test loader. Match what was used in training.
SEQ_LEN         = 100
STRIDE          = 20
BATCH_SIZE      = 128
HIDDEN_DIM      = 128
TAU             = 0.01
K_STEPS         = 5
AMP_ENABLED     = True
COMPILE_ENABLED = True
LR              = 1e-3
# ============================================================


def main():
    np.random.seed(SEED)

    if BASE_RUN_TAG.startswith("PUT_THE"):
        sys.exit("[fatal] edit BASE_RUN_TAG (and other knobs) before running")

    base_dir = CKPT_DIR / BASE_RUN_TAG
    inv_h_dir  = CKPT_DIR / (BASE_RUN_TAG + '_invH')
    inv_nh_dir = CKPT_DIR / (BASE_RUN_TAG + '_invNoH')

    has_fwd    = base_dir.is_dir()    and any(base_dir.glob('forward_*.pth'))
    has_inv_h  = inv_h_dir.is_dir()   and any(inv_h_dir.glob('inverse_*.pth'))
    has_inv_nh = inv_nh_dir.is_dir()  and any(inv_nh_dir.glob('inverse_*.pth'))

    print(f"[discover] {base_dir}: "
          f"{'forward ckpt found' if has_fwd else 'no forward ckpt'}")
    print(f"[discover] {inv_h_dir}: "
          f"{'invH ckpt found' if has_inv_h else 'no invH ckpt'}")
    print(f"[discover] {inv_nh_dir}: "
          f"{'invNoH ckpt found' if has_inv_nh else 'no invNoH ckpt'}")
    if not (has_fwd or has_inv_h or has_inv_nh):
        sys.exit(f"[fatal] no checkpoint .pth files under {base_dir} or its "
                 f"_invH / _invNoH siblings")

    whitelist = parse_whitelist(Path(WHITELIST_PATH))
    whitelist_total = len(whitelist) if whitelist else 0
    print(f"[whitelist] {whitelist_total} approved trajectories")

    trajs = load_all_arrow_trajectories(
        Path(DATA_DIR), whitelist=whitelist,
        mu_values=MU_VALUES, chi_values=CHI_VALUES, motion_cases=MOTION_CASES,
    )
    if not trajs:
        sys.exit("[fatal] no trajectories matched the scope -- "
                 "check MOTION_CASES / MU_VALUES / CHI_VALUES / DATA_DIR")
    print(f"[load] {len(trajs)} trajectories matched scope")

    tr, va, te = stratified_split(trajs, TRAIN_RATIO, VAL_RATIO, seed=SEED)
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}  (seed={SEED})")

    common = dict(
        motion_cases=MOTION_CASES,
        mu_values=MU_VALUES,
        chi_values=CHI_VALUES,
        data_dir=Path(DATA_DIR),
        whitelist_path=Path(WHITELIST_PATH),
        whitelist_total_count=whitelist_total,
        subsample_n=0,
        seed=SEED,
        train_names=[t['name'] for t in tr],
        val_names  =[t['name'] for t in va],
        test_names =[t['name'] for t in te],
        config_summary={
            'seq_len':         SEQ_LEN,
            'stride':          STRIDE,
            'batch_size':      BATCH_SIZE,
            'hidden_dim':      HIDDEN_DIM,
            'tau':             TAU,
            'k_steps':         K_STEPS,
            'amp_enabled':     bool(AMP_ENABLED),
            'compile_enabled': bool(COMPILE_ENABLED),
            'lr':              float(LR),
        },
    )
    fwd_ref = str(base_dir / 'forward_lbfgs.pth') if has_fwd else ""

    if has_fwd:
        save_training_manifest(ckpt_dir=CKPT_DIR, run_tag=BASE_RUN_TAG,
                               stages_trained=['forward'],
                               forward_ckpt_ref="",
                               **common)
    if has_inv_h:
        save_training_manifest(ckpt_dir=CKPT_DIR,
                               run_tag=BASE_RUN_TAG + '_invH',
                               stages_trained=['inverse_H'],
                               forward_ckpt_ref=fwd_ref,
                               **common)
    if has_inv_nh:
        save_training_manifest(ckpt_dir=CKPT_DIR,
                               run_tag=BASE_RUN_TAG + '_invNoH',
                               stages_trained=['inverse_NoH'],
                               forward_ckpt_ref=fwd_ref,
                               **common)

    print("\n[done] manifests written. plot_ood.py will now work for these "
          "checkpoints.")
    print("\n[reminder] the recovered split is deterministic from "
          "(trajectory pool, seed). It matches the ORIGINAL split only if the "
          "data dir and whitelist haven't changed since training. If they "
          "have, treat the manifest as best-effort.")


if __name__ == '__main__':
    main()
