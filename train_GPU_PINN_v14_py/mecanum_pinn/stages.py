"""Run orchestration: CLI, checkpoint classification, stage runners, run_main.

Sits at the top of the import graph -- composes everything in the package
into the four CLI modes:

    forward   -- train forward stage only.
    inverse   -- load a forward checkpoint, train both inverse variants on it.
    both      -- full pipeline (default).
    figures   -- load saved checkpoints, regenerate figures, no training.

Run-tag handling:
    forward, both : run_tag built from build_run_tag() with prefix/suffix.
    inverse       : run_tag derived from --ckpt's parent dir name.
    figures       : run_tag derived from whichever --ckpt is provided first
                    (with-H / without-H suffixes stripped).

Every successful training stage writes a `manifest.toml` next to the .pth
files. The manifest records the exact trajectory split (train/val/test
file names), the training scope (motion/mu/chi), the seed, and a summary
of architectural knobs. The figures stage reads it back to reconstruct
the test set without depending on a re-run of stratified_split. plot_ood.py
(top-level, separate script) does the same thing for OOD evaluation.

`run_main(...)` is the single entry point a top-level train.py script calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import data as _data
from .config import apply_dummy_overrides, build_config, build_run_tag
from .data import (build_loaders_from_lists, build_loaders_with_split,
                   filter_trajectories_by_name, init_torch_globals,
                   load_all_arrow_trajectories, load_ood_test_trajectories,
                   parse_whitelist, stratified_split)
from .evaluation import (estimate_and_plot_mu_chi, estimate_mu_chi,
                         evaluate_on_test, evaluate_ood,
                         plot_test_trajectory_predictions)
from .manifest import (load_manifest_at, load_training_manifest,
                       save_training_manifest)
from .models import (MecanumPINN, build_empty_pinn, maybe_compile_pinn)
from .physics import RobotParams, make_geometry
from .plotting import (configure_figure_saving, plot_history,
                       plot_id_vs_ood_comparison,
                       plot_train_val_test_comparison)
from .training import (load_phase_checkpoint, train_forward,
                       train_inverse_ablation)


# ============================================================
# CLI
# ============================================================
def _parse_args(argv: Optional[List[str]] = None):
    """CLI: choose what to do, and pass checkpoint paths via --ckpt."""
    p = argparse.ArgumentParser(
        description="Mecanum PINN trainer (modular). Train, fine-tune, or "
                    "regenerate figures from saved checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "Train forward stage from scratch:\n"
            "    python train.py forward\n"
            "Train inverse on a saved forward checkpoint:\n"
            "    python train.py inverse --ckpt checkpoints_v12_1/<tag>/forward_lbfgs.pth\n"
            "Generate figures from a single inverse checkpoint:\n"
            "    python train.py figures --ckpt checkpoints_v12_1/<tag>_invH/inverse_lbfgs.pth\n"
            "Generate figures comparing both inverse variants:\n"
            "    python train.py figures \\\n"
            "        --ckpt checkpoints_v12_1/<tag>_invH/inverse_lbfgs.pth \\\n"
            "        --ckpt checkpoints_v12_1/<tag>_invNoH/inverse_lbfgs.pth\n"
            "\nFor OOD evaluation, see plot_ood.py at the top level.\n"
        ),
    )
    p.add_argument(
        "stage", nargs="?", default="both",
        choices=["forward", "inverse", "both", "figures"],
        help="forward: train forward only. "
             "inverse: train inverse on the forward checkpoint passed via --ckpt. "
             "both: full pipeline (default). "
             "figures: load checkpoints and produce figures only -- no training.",
    )
    p.add_argument(
        "--ckpt", type=Path, action="append", default=[], metavar="PATH",
        help="Path to a checkpoint .pth file. May be repeated. The kind "
             "(forward / inverse-with-H / inverse-without-H) is auto-detected "
             "from each path's parent directory suffix.",
    )
    args = p.parse_args(argv)

    if args.stage in ("forward", "both"):
        if args.ckpt:
            print(f"[args] warning: --ckpt args are ignored in '{args.stage}' mode")

    elif args.stage == "inverse":
        if len(args.ckpt) != 1:
            p.error("inverse mode requires exactly one --ckpt PATH "
                    "(a forward checkpoint to train inverse on top of)")
        kind, _ = _classify_checkpoint(args.ckpt[0])
        if kind != 'forward':
            p.error(f"inverse mode needs a FORWARD checkpoint; got '{kind}' "
                    f"from {args.ckpt[0]}.\n"
                    f"  (Forward checkpoints live at <tag>/forward_*.pth -- "
                    f"their parent dir does NOT end in _invH or _invNoH.)")

    elif args.stage == "figures":
        if not args.ckpt:
            p.error("figures mode requires at least one --ckpt PATH")

    for c in args.ckpt:
        if not c.is_file():
            p.error(f"--ckpt: file not found: {c}")

    return args


# ============================================================
# Checkpoint helpers
# ============================================================
def _classify_checkpoint(ckpt_path: Path) -> Tuple[str, Optional[bool]]:
    """Classify a checkpoint by its parent directory's suffix.

    Returns (kind, use_H):
      kind in {'forward', 'inverse_H', 'inverse_NoH'}
      use_H is True for inverse_H, False for inverse_NoH, None for forward.
    """
    parent = ckpt_path.parent.name
    if parent.endswith('_invH'):
        return 'inverse_H', True
    if parent.endswith('_invNoH'):
        return 'inverse_NoH', False
    return 'forward', None


def _derive_run_tag_from_ckpt(ckpt_path: Path) -> str:
    """Strip _invH / _invNoH suffix from parent dir to recover base run_tag."""
    parent = ckpt_path.parent.name
    for suffix in ('_invH', '_invNoH'):
        if parent.endswith(suffix):
            return parent[:-len(suffix)]
    return parent


def _detect_use_H_from_ckpt(ckpt_path: Path) -> bool:
    kind, use_H = _classify_checkpoint(ckpt_path)
    if use_H is None:
        raise ValueError(
            f"Cannot detect use_H from inverse checkpoint path: {ckpt_path}\n"
            f"  Parent dir name '{ckpt_path.parent.name}' does not end with "
            f"'_invH' or '_invNoH'.\n"
            f"  This looks like a {kind} checkpoint."
        )
    return use_H


def _load_forward_model(config: Dict[str, Any], geom,
                        ckpt_path: Path) -> Tuple[MecanumPINN, Dict]:
    """Load a forward-only checkpoint into a fresh PINN.

    Order is build -> load -> compile. Compiling AFTER load lets the saved
    state_dict (which has no _orig_mod prefix because we wrap bound methods,
    not the module) load cleanly into the uncompiled skeleton.
    """
    model = build_empty_pinn(config, geom, use_H=True)
    ckpt = load_phase_checkpoint(model, ckpt_path, map_location=config['device'])
    model = maybe_compile_pinn(model, config)
    return model, ckpt.get('history', {})


def _load_inverse_model(config: Dict[str, Any], geom,
                        ckpt_path: Path) -> Tuple[MecanumPINN, Dict]:
    """Load an inverse checkpoint. Auto-detects use_H from path suffix."""
    use_H = _detect_use_H_from_ckpt(ckpt_path)
    model = build_empty_pinn(config, geom, use_H=use_H)
    ckpt = load_phase_checkpoint(model, ckpt_path, map_location=config['device'])
    model = maybe_compile_pinn(model, config)
    return model, ckpt.get('history', {})


# ============================================================
# Manifest helper
# ============================================================
def _save_stage_manifest(config: Dict[str, Any],
                         run_tag_with_suffix: str,
                         stages_trained: Sequence[str],
                         tr_trajs, va_trajs, te_trajs,
                         whitelist_total_count: int,
                         subsample_n: int,
                         forward_ckpt_ref: str = "") -> None:
    """Bundle config snapshot + split into a manifest write."""
    save_training_manifest(
        ckpt_dir=config['ckpt_dir'],
        run_tag=run_tag_with_suffix,
        motion_cases=config['motion_cases'],
        mu_values=config['mu_values'],
        chi_values=config['chi_values'],
        data_dir=config['data_dir'],
        whitelist_path=config['whitelist_path'],
        whitelist_total_count=whitelist_total_count,
        subsample_n=subsample_n,
        seed=config['seed'],
        train_names=[t['name'] for t in tr_trajs],
        val_names  =[t['name'] for t in va_trajs],
        test_names =[t['name'] for t in te_trajs],
        config_summary={
            'seq_len':         config['seq_len'],
            'stride':          config['stride'],
            'batch_size':      config['batch_size'],
            'hidden_dim':      config['hidden_dim'],
            'tau':             config['tau'],
            'k_steps':         config['k_steps'],
            'amp_enabled':     bool(config['amp_enabled']),
            'compile_enabled': bool(config['compile_enabled']),
            'lr':              config['lr'],
        },
        stages_trained=stages_trained,
        forward_ckpt_ref=forward_ckpt_ref,
    )


# ============================================================
# Stage runners
# ============================================================
def _stage_forward(config, rp, geom, tr_loader, va_loader, te_loader):
    """Train forward end-to-end + evaluate. Returns (model, history)."""
    print("\n" + "=" * 60)
    print("STAGE: forward")
    print("=" * 60)

    model_fwd, hist_fwd = train_forward(rp, tr_loader, va_loader, config, geom=geom)
    plot_history(hist_fwd, stage='forward')

    test_fwd = evaluate_on_test(model_fwd, te_loader, rp, config,
                                stage='forward', desc='test/forward')
    plot_train_val_test_comparison(hist_fwd, test_fwd, stage='forward')
    return model_fwd, hist_fwd


def _stage_inverse(config, rp, geom, tr_loader, va_loader, te_loader,
                   te_trajs, model_fwd):
    """Train both inverse variants on a forward backbone, plus diagnostics.

    Returns (model_H, model_NH). The caller is responsible for saving
    manifests into the two inverse-variant checkpoint dirs.
    """
    print("\n" + "=" * 60)
    print("STAGE: inverse (with-H + without-H ablation)")
    print("=" * 60)

    model_H, hist_H, model_NH, hist_NH = train_inverse_ablation(
        rp, tr_loader, va_loader, config, model_fwd, geom,
    )
    plot_history(hist_H,  stage='inverse')
    plot_history(hist_NH, stage='inverse')

    print('--- inverse with H ---')
    plot_test_trajectory_predictions(model_H, te_trajs, rp, config, num_cases=3)
    test_inv_H = evaluate_on_test(model_H, te_loader, rp, config,
                                  stage='inverse', desc='test/inv_H')
    plot_train_val_test_comparison(hist_H, test_inv_H, stage='inverse')

    print('--- inverse without H ---')
    plot_test_trajectory_predictions(model_NH, te_trajs, rp, config, num_cases=3)
    test_inv_NH = evaluate_on_test(model_NH, te_loader, rp, config,
                                   stage='inverse', desc='test/inv_NH')
    plot_train_val_test_comparison(hist_NH, test_inv_NH, stage='inverse')

    print('--- mu/chi rolling estimates (with H) ---')
    estimate_and_plot_mu_chi(model_H, te_trajs, rp, config,
                             window=100, num_cases=3, seed=0,
                             tag='muchi_with_H')
    print('--- mu/chi rolling estimates (without H) ---')
    estimate_and_plot_mu_chi(model_NH, te_trajs, rp, config,
                             window=100, num_cases=3, seed=0,
                             tag='muchi_without_H')

    for traj in te_trajs[:5]:
        mu_h, chi_h = estimate_mu_chi(model_H,  traj, rp, config)
        mu_n, chi_n = estimate_mu_chi(model_NH, traj, rp, config)
        print(f"{traj['name']}")
        print(f"  label  : mu={traj['mu']:.3f} chi={traj['chi']:.4f}")
        print(f"  with H : mu={mu_h:.3f} chi={chi_h:.4f}")
        print(f"  no H   : mu={mu_n:.3f} chi={chi_n:.4f}")

    return model_H, model_NH


def _stage_figures(config, rp, geom, tr_loader, va_loader, te_loader,
                   te_trajs, ckpt_paths: List[Path]):
    """Load checkpoints (no training) and produce figures.

    The te_loader and te_trajs passed in are already manifest-aligned (or
    seed-aligned, if no manifest was found) by run_main.
    """
    print("\n" + "=" * 60)
    print("STAGE: figures (eval + plotting only, no training)")
    print("=" * 60)

    # Bucket checkpoints by kind. Last writer wins.
    buckets: Dict[str, Optional[Path]] = {
        'forward': None, 'inverse_H': None, 'inverse_NoH': None,
    }
    for path in ckpt_paths:
        kind, _ = _classify_checkpoint(path)
        if buckets[kind] is not None:
            print(f"[figs] warning: multiple '{kind}' checkpoints provided; "
                  f"using {path} (overrides {buckets[kind]})")
        buckets[kind] = path
        print(f"[figs] {path.name} -> classified as: {kind}")

    forward_ckpt = buckets['forward']
    inv_h_ckpt   = buckets['inverse_H']
    inv_nh_ckpt  = buckets['inverse_NoH']

    model_fwd = model_H = model_NH = None
    hist_fwd  = hist_H  = hist_NH  = None

    if forward_ckpt is not None:
        print(f"[figs] loading forward checkpoint: {forward_ckpt}")
        model_fwd, hist_fwd = _load_forward_model(config, geom, forward_ckpt)

        if hist_fwd and 'forward' in hist_fwd:
            plot_history(hist_fwd, stage='forward')
        else:
            print("[figs] no forward history in checkpoint -- skipping plot_history(forward)")

        test_fwd = evaluate_on_test(model_fwd, te_loader, rp, config,
                                    stage='forward', desc='test/forward')
        if hist_fwd and 'forward' in hist_fwd:
            plot_train_val_test_comparison(hist_fwd, test_fwd, stage='forward')

    if inv_h_ckpt is not None:
        print(f"[figs] loading inverse-with-H checkpoint: {inv_h_ckpt}")
        model_H, hist_H = _load_inverse_model(config, geom, inv_h_ckpt)
        if hist_H and 'inverse' in hist_H:
            plot_history(hist_H, stage='inverse')

    if inv_nh_ckpt is not None:
        print(f"[figs] loading inverse-without-H checkpoint: {inv_nh_ckpt}")
        model_NH, hist_NH = _load_inverse_model(config, geom, inv_nh_ckpt)
        if hist_NH and 'inverse' in hist_NH:
            plot_history(hist_NH, stage='inverse')

    if model_H is not None or model_NH is not None:
        if model_H is not None:
            print('--- inverse with H ---')
            plot_test_trajectory_predictions(model_H, te_trajs, rp, config, num_cases=3)
            test_inv_H = evaluate_on_test(model_H, te_loader, rp, config,
                                          stage='inverse', desc='test/inv_H')
            if hist_H and 'inverse' in hist_H:
                plot_train_val_test_comparison(hist_H, test_inv_H, stage='inverse')
            estimate_and_plot_mu_chi(model_H, te_trajs, rp, config,
                                     window=100, num_cases=3, seed=0,
                                     tag='muchi_with_H')

        if model_NH is not None:
            print('--- inverse without H ---')
            plot_test_trajectory_predictions(model_NH, te_trajs, rp, config, num_cases=3)
            test_inv_NH = evaluate_on_test(model_NH, te_loader, rp, config,
                                           stage='inverse', desc='test/inv_NH')
            if hist_NH and 'inverse' in hist_NH:
                plot_train_val_test_comparison(hist_NH, test_inv_NH, stage='inverse')
            estimate_and_plot_mu_chi(model_NH, te_trajs, rp, config,
                                     window=100, num_cases=3, seed=0,
                                     tag='muchi_without_H')

        for traj in te_trajs[:5]:
            print(f"{traj['name']}")
            print(f"  label  : mu={traj['mu']:.3f} chi={traj['chi']:.4f}")
            if model_H is not None:
                mu_h, chi_h = estimate_mu_chi(model_H,  traj, rp, config)
                print(f"  with H : mu={mu_h:.3f} chi={chi_h:.4f}")
            if model_NH is not None:
                mu_n, chi_n = estimate_mu_chi(model_NH, traj, rp, config)
                print(f"  no H   : mu={mu_n:.3f} chi={chi_n:.4f}")
    elif model_fwd is not None:
        print("[figs] no inverse checkpoints provided -- skipping inverse plots.")
    else:
        print("[figs] no usable checkpoints -- nothing to plot")

    return model_fwd, model_H, model_NH


# ============================================================
# Top-level entry
# ============================================================
def _set_global_torch_flags(device: torch.device):
    """One-time PyTorch knobs."""
    if device.type != 'cuda':
        return
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    try:
        # NOTE: use `from torch import _dynamo` rather than `import torch._dynamo`.
        # The latter rebinds `torch` to the local function scope, which makes
        # the earlier `torch.set_float32_matmul_precision` / `torch.backends`
        # lookups raise UnboundLocalError on some Python versions.
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 16
    except Exception as e:
        print(f"[torch flags] could not bump dynamo cache size: {e!r}")


def run_main(*, config_kwargs: Optional[Dict[str, Any]] = None,
             prefix: str = "",
             suffix: str = "",
             p1_wheels: float = 5e-4,
             argv: Optional[List[str]] = None) -> None:
    """The single entry point a top-level train.py script calls.

    Parameters
    ----------
    config_kwargs : kwargs forwarded to build_config(); use this for
        VRAM tier, paths, motion/mu/chi filters, dummy.
    prefix, suffix : strings concatenated into build_run_tag() for
        forward/both modes (ignored for inverse/figures, where the tag
        is derived from the checkpoint path).
    p1_wheels : viscous bearing friction coefficient frozen into RobotParams.
    argv : pass a list to override sys.argv (handy for notebooks).
    """
    args = _parse_args(argv)
    config_kwargs = dict(config_kwargs or {})
    config = build_config(**config_kwargs)

    # Device + global flags
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    print(f"[device] {device}")
    _set_global_torch_flags(device)

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])

    apply_dummy_overrides(config)
    init_torch_globals(device)

    # Run-tag resolution
    if args.stage in ('forward', 'both'):
        config['run_tag'] = build_run_tag(config, prefix=prefix, suffix=suffix)
    elif args.stage == 'inverse':
        config['run_tag'] = _derive_run_tag_from_ckpt(args.ckpt[0])
    elif args.stage == 'figures':
        config['run_tag'] = _derive_run_tag_from_ckpt(args.ckpt[0])
    print(f"[run] run_tag = {config['run_tag']}")
    configure_figure_saving(Path(config['figure_dir']) / config['run_tag'],
                            run_tag=config['run_tag'])

    # Robot + geometry
    rp = RobotParams().finalize(p1_wheels=p1_wheels, device=device)
    geom = make_geometry(rp)

    # Whitelist + trajectories (load_all_arrow_trajectories filters by scope)
    n_dummy = config['dummy_n_trajectories'] if config.get('dummy', False) else None
    whitelist = parse_whitelist(config['whitelist_path'],
                                subsample_n=n_dummy,
                                subsample_seed=config['seed'])
    whitelist_total_count = len(whitelist) if whitelist else 0
    trajectories = load_all_arrow_trajectories(
        config['data_dir'],
        whitelist=whitelist,
        mu_values=config['mu_values'],
        chi_values=config['chi_values'],
        motion_cases=config['motion_cases'],
    )
    if not trajectories:
        print("[fatal] no trajectories loaded -- check data_dir and filters")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Build split + loaders.
    # For training modes: stratified split, manifests get written from
    # the resulting train/val/test name lists.
    # For figures mode: try to read the manifest from the first --ckpt
    # and use its recorded test_names; fall back to seed-based split if
    # no manifest is present (older checkpoints stay loadable).
    # ----------------------------------------------------------------
    tr_loader, va_loader, te_loader, tr_trajs, va_trajs, te_trajs = \
        build_loaders_with_split(trajectories, config)

    if args.stage == 'figures':
        m = load_manifest_at(args.ckpt[0])
        if m is not None and 'split' in m and 'test_names' in m['split']:
            recorded_test_names = m['split']['test_names']
            te_trajs = filter_trajectories_by_name(trajectories, recorded_test_names)
            if te_trajs:
                # Rebuild the test loader from the recorded list. tr/va loaders
                # don't matter for figures mode, so we keep the original ones.
                _, _, te_loader = build_loaders_from_lists(
                    tr_trajs[:1], va_trajs[:1], te_trajs, config,
                )
                print(f"[figures] using manifest test set "
                      f"({len(te_trajs)} trajectories)")
            else:
                print(f"[figures] manifest test_names not present in "
                      f"trajectory pool; falling back to seed-based split")
        else:
            print(f"[figures] no manifest at {args.ckpt[0].parent} -- "
                  f"falling back to seed-based stratified_split")

    # ----------------------------------------------------------------
    # Dispatch
    # ----------------------------------------------------------------
    base_tag = config['run_tag']
    sub_n    = n_dummy if n_dummy is not None else 0

    if args.stage == 'forward':
        model_fwd, _ = _stage_forward(config, rp, geom,
                                      tr_loader, va_loader, te_loader)
        _save_stage_manifest(config, base_tag, ['forward'],
                             tr_trajs, va_trajs, te_trajs,
                             whitelist_total_count, sub_n)

    elif args.stage == 'inverse':
        # Load forward weights from the provided ckpt, then train both inverse variants.
        fwd_ckpt = args.ckpt[0]
        print(f"[inverse] loading forward weights from {fwd_ckpt}")
        model_fwd, _ = _load_forward_model(config, geom, fwd_ckpt)
        model_H, model_NH = _stage_inverse(config, rp, geom,
                                           tr_loader, va_loader, te_loader,
                                           te_trajs, model_fwd)
        # Two inverse manifests (different folders), both reference the forward
        for tag_suffix, st in (('_invH', 'inverse_H'),
                               ('_invNoH', 'inverse_NoH')):
            _save_stage_manifest(config, base_tag + tag_suffix, [st],
                                 tr_trajs, va_trajs, te_trajs,
                                 whitelist_total_count, sub_n,
                                 forward_ckpt_ref=str(fwd_ckpt))

    elif args.stage == 'both':
        model_fwd, _ = _stage_forward(config, rp, geom,
                                      tr_loader, va_loader, te_loader)
        # Save forward manifest immediately so a crash in inverse training
        # still leaves a manifest behind for the forward checkpoint.
        _save_stage_manifest(config, base_tag, ['forward'],
                             tr_trajs, va_trajs, te_trajs,
                             whitelist_total_count, sub_n)

        model_H, model_NH = _stage_inverse(config, rp, geom,
                                           tr_loader, va_loader, te_loader,
                                           te_trajs, model_fwd)
        # The forward-ckpt-ref for these inverse manifests is the lbfgs ckpt
        # we just wrote (or the last phase if lbfgs was disabled).
        forward_ref = str(Path(config['ckpt_dir']) / base_tag / 'forward_lbfgs.pth')
        for tag_suffix, st in (('_invH', 'inverse_H'),
                               ('_invNoH', 'inverse_NoH')):
            _save_stage_manifest(config, base_tag + tag_suffix, [st],
                                 tr_trajs, va_trajs, te_trajs,
                                 whitelist_total_count, sub_n,
                                 forward_ckpt_ref=forward_ref)

    elif args.stage == 'figures':
        _stage_figures(config, rp, geom,
                       tr_loader, va_loader, te_loader,
                       te_trajs, args.ckpt)

    print("\n[done] run complete.")
