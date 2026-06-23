#!/usr/bin/env python3
"""Run OOD evaluation on trained checkpoint(s) and plot ID-vs-OOD comparisons.

Standalone -- safe to run on CPU while a real training job has the GPU.
The training scope (motion_cases, mu_values, chi_values) and the
in-distribution test trajectory list are read from the manifest.toml that
lives alongside each checkpoint .pth. You only specify checkpoint paths
and which OOD axes to evaluate.

Usage:
    1. Edit the knobs below.
    2. python plot_ood.py

CPU-only (e.g. while another GPU training is running):
    Linux/macOS:    CUDA_VISIBLE_DEVICES="" python plot_ood.py
    Windows PS:     $env:CUDA_VISIBLE_DEVICES = ""; python plot_ood.py
    Windows cmd:    set CUDA_VISIBLE_DEVICES= && python plot_ood.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mecanum_pinn import (
    RobotParams, build_config, configure_figure_saving,
    evaluate_on_test, evaluate_ood, init_torch_globals,
    load_all_arrow_trajectories, load_ood_test_trajectories,
    make_geometry, parse_whitelist, plot_id_vs_ood_comparison,
)
from mecanum_pinn.data import (build_loaders_from_lists,
                                filter_trajectories_by_name)
from mecanum_pinn.manifest import load_manifest_at
from mecanum_pinn.stages import (_load_forward_model, _load_inverse_model)


# ============================================================
# KNOBS  -- edit these, then python plot_ood.py
# ============================================================
# Checkpoints to evaluate. Set any to None to skip that kind. The training
# scope (motion / mu / chi) and ID test trajectory list are auto-read from
# the manifest.toml beside each .pth file.
CKPT_FORWARD: Optional[Path] = Path(
    r"/home/vishvesh/mecanum_pinn_main/data/checkpoints_v14/24GBVRAM_motion_circle_infinity_mu_0p5_0p6_chi_0p000_0p002_run01seed42/forward_lbfgs.pth"
)
CKPT_INVERSE_H: Optional[Path] = Path(
    r"/home/vishvesh/mecanum_pinn_main/data/checkpoints_v14/24GBVRAM_motion_circle_infinity_mu_0p5_0p6_chi_0p000_0p002_run01seed42_invH/inverse_lbfgs.pth"
)
CKPT_INVERSE_NH: Optional[Path] = Path(
    r"/home/vishvesh/mecanum_pinn_main/data/checkpoints_v14/24GBVRAM_motion_circle_infinity_mu_0p5_0p6_chi_0p000_0p002_run01seed42_invNoH/inverse_lbfgs.pth"
)

# OOD axes -- what was held OUT of training. Empty list = skip that axis.
HELD_OUT_CHI     = [0.008]                          # chi values absent from training
HELD_OUT_MOTIONS = ['straightline', 'sineline']     # motion cases absent
HELD_OUT_MU      = [0.4]                                # mu values absent (often empty)

OOD_FRACTION = 0.5            # fraction of held-out pool to subsample (1.0 = all)
USE_WHITELIST_FOR_OOD = True  # respect the training-time whitelist when finding OOD trajs

SEED          = 42
P1_WHEELS     = 0.11

# Optional override for figure dir (None -> use whatever build_config set)
FIGURE_DIR_OVERRIDE: Optional[str] = None
# ============================================================


def _find_manifest():
    """Walk through the candidate checkpoints and return the first one that
    has a manifest.toml beside it. We need exactly one manifest -- ID test
    set is the same across forward/_invH/_invNoH since they trained on the
    same split, so any manifest works."""
    for label, ck in (('forward',     CKPT_FORWARD),
                      ('inverse_H',   CKPT_INVERSE_H),
                      ('inverse_NoH', CKPT_INVERSE_NH)):
        if ck is None:
            continue
        if not ck.is_file():
            print(f"[warn] {label} ckpt not found: {ck}")
            continue
        m = load_manifest_at(ck)
        if m is not None:
            print(f"[manifest] read from {ck.parent / 'manifest.toml'}")
            return m, ck
    return None, None


def main():
    # 1) Device + flags
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")
    np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # 2) Manifest -- source of truth for scope and ID split
    manifest, manifest_source = _find_manifest()
    if manifest is None:
        sys.exit(
            "[fatal] no manifest.toml found beside any of the configured "
            "checkpoints. Either re-train with the new code (which writes "
            "manifests automatically) or hand-create a manifest matching the "
            "TOML schema in mecanum_pinn/manifest.py."
        )

    scope = manifest['scope']
    cs    = manifest.get('config_summary', {})
    sub_n = manifest['data_source'].get('subsample_n', 0)

    # 3) Build config from manifest. vram_gb just sets defaults that we
    #    immediately overwrite from the manifest's config_summary.
    cfg = build_config(
        vram_gb=6,
        data_dir=manifest['data_source']['data_dir'],
        whitelist_path=manifest['data_source']['whitelist_path'],
        motion_cases=scope['motion_cases'],
        mu_values=scope['mu_values'],
        chi_values=scope['chi_values'],
        seed=manifest['split']['seed'],
    )
    cfg['seq_len']    = cs.get('seq_len',    cfg['seq_len'])
    cfg['stride']     = cs.get('stride',     cfg['stride'])
    cfg['batch_size'] = cs.get('batch_size', cfg['batch_size'])
    cfg['hidden_dim'] = cs.get('hidden_dim', cfg['hidden_dim'])
    cfg['amp_enabled']     = bool(cs.get('amp_enabled',     cfg['amp_enabled']))
    cfg['compile_enabled'] = False         # never compile for one-shot eval
    cfg['device']          = device
    init_torch_globals(device)

    # 4) Run-tag = base tag (strip _invH/_invNoH from manifest source's parent)
    parent = manifest_source.parent.name
    for sfx in ('_invH', '_invNoH'):
        if parent.endswith(sfx):
            parent = parent[:-len(sfx)]
            break
    cfg['run_tag'] = parent
    fig_dir = FIGURE_DIR_OVERRIDE or cfg['figure_dir']
    configure_figure_saving(Path(fig_dir) / cfg['run_tag'], run_tag=cfg['run_tag'])

    # 5) Robot
    rp = RobotParams().finalize(p1_wheels=P1_WHEELS, device=device)
    geom = make_geometry(rp)

    # 6) ID trajectory pool (filtered by training scope) + ID test loader
    whitelist = parse_whitelist(cfg['whitelist_path'])
    id_pool = load_all_arrow_trajectories(
        cfg['data_dir'], whitelist=whitelist,
        mu_values=cfg['mu_values'], chi_values=cfg['chi_values'],
        motion_cases=cfg['motion_cases'],
    )
    if not id_pool:
        sys.exit("[fatal] no in-distribution trajectories found")

    test_names = manifest['split']['test_names']
    id_test_trajs = filter_trajectories_by_name(id_pool, test_names)
    if not id_test_trajs:
        sys.exit(f"[fatal] none of the {len(test_names)} manifest test "
                 f"trajectories are present in {cfg['data_dir']}")
    if len(id_test_trajs) < len(test_names):
        print(f"[warn] {len(id_test_trajs)}/{len(test_names)} manifest test "
              f"trajectories present -- some files may have moved")

    # build_loaders_from_lists wants three lists; we only need te_loader.
    placeholder = id_test_trajs[:1]
    _, _, te_loader = build_loaders_from_lists(
        placeholder, placeholder, id_test_trajs, cfg,
    )

    # 7) Load checkpoints
    model_fwd = model_H = model_NH = None
    if CKPT_FORWARD and CKPT_FORWARD.is_file():
        print(f"[load] forward: {CKPT_FORWARD}")
        model_fwd, _ = _load_forward_model(cfg, geom, CKPT_FORWARD)
    if CKPT_INVERSE_H and CKPT_INVERSE_H.is_file():
        print(f"[load] inverse_H: {CKPT_INVERSE_H}")
        model_H, _ = _load_inverse_model(cfg, geom, CKPT_INVERSE_H)
    if CKPT_INVERSE_NH and CKPT_INVERSE_NH.is_file():
        print(f"[load] inverse_NoH: {CKPT_INVERSE_NH}")
        model_NH, _ = _load_inverse_model(cfg, geom, CKPT_INVERSE_NH)

    # 8) For each OOD axis: load held-out trajs + evaluate ID + OOD + plot
    ood_axes = []
    if HELD_OUT_CHI:
        ood_axes.append(('chi', HELD_OUT_CHI, dict(
            ood_chi_values=HELD_OUT_CHI,
            ood_motion_cases=cfg['motion_cases'],
            ood_mu_values=cfg['mu_values'],
        )))
    if HELD_OUT_MOTIONS:
        ood_axes.append(('motion', HELD_OUT_MOTIONS, dict(
            ood_motion_cases=HELD_OUT_MOTIONS,
            ood_mu_values=cfg['mu_values'],
            ood_chi_values=cfg['chi_values'],
        )))
    if HELD_OUT_MU:
        ood_axes.append(('mu', HELD_OUT_MU, dict(
            ood_mu_values=HELD_OUT_MU,
            ood_motion_cases=cfg['motion_cases'],
            ood_chi_values=cfg['chi_values'],
        )))

    if not ood_axes:
        print("[ood] all HELD_OUT_* lists empty -- nothing to evaluate")
        return

    ood_whitelist = whitelist if USE_WHITELIST_FOR_OOD else None

    for axis_name, held_out, kw in ood_axes:
        print(f"\n=== OOD axis: {axis_name}  held_out={held_out} ===")
        ood_trajs = load_ood_test_trajectories(
            cfg['data_dir'], whitelist=ood_whitelist,
            fraction=OOD_FRACTION, seed=SEED, **kw,
        )
        if not ood_trajs:
            print(f"[skip] axis={axis_name}: no OOD trajectories returned")
            continue
        sample = ", ".join(t['name'] for t in ood_trajs[:6])
        more   = "..." if len(ood_trajs) > 6 else ""
        print(f"[ood/{axis_name}] {len(ood_trajs)} trajs: {sample}{more}")

        if model_fwd is not None:
            id_m  = evaluate_on_test(model_fwd, te_loader, rp, cfg,
                                     stage='forward', desc='ID/fwd')
            ood_m = evaluate_ood(model_fwd, ood_trajs, rp, cfg,
                                 stage='forward', desc=f'OOD-{axis_name}/fwd')
            plot_id_vs_ood_comparison(id_m, ood_m, stage='forward',
                                      label_ood=f'OOD {axis_name}')

        for tag, m in (('with_H', model_H), ('without_H', model_NH)):
            if m is None:
                continue
            id_m  = evaluate_on_test(m, te_loader, rp, cfg,
                                     stage='inverse', desc=f'ID/inv_{tag}')
            ood_m = evaluate_ood(m, ood_trajs, rp, cfg,
                                 stage='inverse', desc=f'OOD-{axis_name}/inv_{tag}')
            plot_id_vs_ood_comparison(id_m, ood_m, stage='inverse',
                                      label_ood=f'OOD {axis_name} ({tag})')

    print(f"\n[done] OOD figures saved under {Path(fig_dir) / cfg['run_tag']}")


if __name__ == '__main__':
    main()
