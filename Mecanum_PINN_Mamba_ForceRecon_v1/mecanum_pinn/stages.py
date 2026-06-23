"""Run orchestration: CLI + run_main. Modes: forward | inverse | both | figures.

GPU settings retained from train_GPU_PINN_v14_py: float32_matmul_precision('high'),
cudnn.benchmark, dynamo cache bump, torch.compile (via maybe_compile_pinn), and
the VRAM-tier batch/worker sizing in config.build_config. Single inverse variant
(the v14 with-H / without-H ablation is dropped — our inverse has the (mu,chi)
readout instead).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import apply_dummy_overrides, build_config, build_run_tag
from .data import (build_loaders_from_lists, build_loaders_with_split,
                   init_torch_globals, load_all_arrow_trajectories,
                   load_regime_split, parse_whitelist, warm_cache)
from .evaluation import estimate_mu, evaluate_mu_id, evaluate_on_test
from .manifest import save_training_manifest
from .models import MecanumPINN, maybe_compile_pinn
from .physics import RobotParams
from .plotting import configure_figure_saving, plot_history
from .training import load_phase_checkpoint, train_forward, train_inverse


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Mecanum Mamba ForceRecon trainer.")
    p.add_argument("stage", nargs="?", default="both",
                   choices=["forward", "inverse", "both", "figures"])
    p.add_argument("--ckpt", type=Path, action="append", default=[],
                   help="checkpoint .pth (required for inverse/figures).")
    # --- overrides used by the shared parallel launcher (launch_parallel.py) ---
    p.add_argument("--vram", type=int, choices=[6, 12, 24], default=None,
                   help="VRAM tier (batch/workers/amp); explicit --batch-size still wins.")
    p.add_argument("--regime", type=Path, default=None,
                   help="regime TOML (observer_v1_py/regimes/*.toml); shared with A2.")
    p.add_argument("--test-chi", type=float, default=None,
                   help="S3: held-out chi for this k-fold run (overrides regime).")
    p.add_argument("--batch-size", "--per-run-batch", dest="batch_size", type=int,
                   default=None, help="override the VRAM-tier batch size.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="decimated-500Hz .npz cache dir ('' disables).")
    p.add_argument("--run-tag", type=str, default=None,
                   help="explicit run_tag (overrides build_run_tag); the launcher "
                        "sets this to the job label so the run dir is deterministic.")
    p.add_argument("--no-lbfgs", action="store_true",
                   help="disable the L-BFGS refinement (faster sweep runs).")
    p.add_argument("--epoch-scale", type=float, default=None,
                   help="scale EVERY forward+inverse phase epoch count (and the ES "
                        "min_epochs + lbfgs_max_iter) by this factor. 0.5 ~halves the "
                        "schedule (~220 total epochs); ~0.27 -> ~120 total, for tuning.")
    p.add_argument("--warm-cache-only", action="store_true",
                   help="single-process pre-build of the decimated cache, then exit "
                        "(covers all mu/chi so every regime+S3-fold run hits warm cache).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override any top-level config key (typed); repeatable. "
                        "e.g. --set ssm_d_model=48 --set lr=2e-3 --set force_four_term=false")
    args = p.parse_args(argv)
    if args.stage in ("inverse", "figures") and not args.ckpt:
        p.error(f"{args.stage} mode requires --ckpt")
    return args


def _coerce(v: str):
    """Type-infer a --set value: bool -> int -> float -> str."""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        if key not in cfg:
            print(f"[set] WARNING: '{key}' is not an existing config key (adding it)")
        cfg[key] = _coerce(val)
        print(f"[set] {key} = {cfg[key]!r}")


def _scale_epochs(cfg: Dict[str, Any], scale: float) -> None:
    """Scale every phase epoch count (+ ES min_epochs + L-BFGS iters) by `scale`,
    for both stages. The 5-phase shape (and its lr ramp) is preserved; only the
    epoch budget shrinks — the tuning lever the launcher's window ablation uses."""
    keys = ("grounding_epochs", "rampup_epochs", "overlap_epochs",
            "rampdown_epochs", "physics_epochs", "min_epochs", "lbfgs_max_iter")
    for stage in ("forward", "inverse"):
        for k in keys:
            if k in cfg[stage]:
                cfg[stage][k] = max(1, round(cfg[stage][k] * scale))
        tot = sum(cfg[stage][k] for k in keys[:5])
        print(f"[epoch-scale x{scale:g}] {stage}: {tot} phase-epochs "
              f"(min_ep={cfg[stage]['min_epochs']}, lbfgs={cfg[stage]['lbfgs_max_iter']})")


def _write_metrics(cfg: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Write run_dir/metrics.json — the completion marker the shared launcher
    uses for resume-skip and the ranking-CSV harvest (parallel_sweep.py)."""
    run_dir = Path(cfg['ckpt_dir']) / cfg['run_tag']
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_tag": cfg['run_tag'], "regime": str(cfg.get('regime_toml') or ''),
               "test_chi": cfg.get('test_chi'),
               "ssm_d_model": cfg.get('ssm_d_model'), "ssm_d_state": cfg.get('ssm_d_state'),
               "batch_size": cfg.get('batch_size'),
               **{k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in metrics.items()}}
    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(payload, fh, indent=0)
    print(f"[metrics] wrote {run_dir / 'metrics.json'}")


def _set_global_torch_flags(device: torch.device) -> None:
    if device.type != 'cuda':
        return
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    try:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 16
    except Exception as e:                                   # pragma: no cover
        print(f"[torch flags] dynamo cache bump failed: {e!r}")


def _summary(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: cfg[k] for k in ('seq_len', 'stride', 'batch_size', 'target_hz',
                                'ssm_d_model', 'ssm_d_state', 'inv_window', 'k_steps',
                                'lr', 'amp_enabled', 'compile_enabled') if k in cfg}


def _manifest(cfg, tag, stages, tr, va, te, fwd_ref=""):
    save_training_manifest(
        ckpt_dir=cfg['ckpt_dir'], run_tag=tag,
        profiles=cfg['profiles'], mu_values=cfg['mu_values'], chi_values=cfg['chi_values'],
        data_dir=cfg['data_dir'], whitelist_path=cfg['whitelist_path'], seed=cfg['seed'],
        train_names=[t['name'] for t in tr], val_names=[t['name'] for t in va],
        test_names=[t['name'] for t in te], config_summary=_summary(cfg),
        stages_trained=stages, forward_ckpt_ref=fwd_ref)


def _report_mu(model, va_loader, te_loader, te_trajs, rp, cfg, k=5) -> Dict[str, float]:
    # test-time mu identification (val + test) + a few per-trajectory estimates.
    # Returns the TEST mu-id dict (the deliverable metric) for the metrics harvest.
    evaluate_mu_id(model, va_loader, rp, cfg, 'val')
    test = evaluate_mu_id(model, te_loader, rp, cfg, 'test')
    for traj in te_trajs[:k]:
        mu_h = estimate_mu(model, traj, rp, cfg)
        print(f"{traj['name']}  label mu={traj['mu']:.3f}  |  est mu={mu_h:.3f}")
    return test


def run_main(*, config_kwargs: Optional[Dict[str, Any]] = None,
             prefix: str = "", suffix: str = "",
             p1_wheels: float = 0.11,           # drivetrain viscous, friction_case 1
             argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # CLI overrides (used by the shared launcher) win over the train.py defaults.
    ck = dict(config_kwargs or {})
    if args.vram is not None:
        ck['vram_gb'] = args.vram
    if args.regime is not None:
        ck['regime_toml'] = str(args.regime)
    if args.test_chi is not None:
        ck['test_chi'] = args.test_chi
    if args.cache_dir is not None:
        ck['cache_dir'] = args.cache_dir
    if args.run_tag is not None:
        ck['run_tag'] = args.run_tag
    cfg = build_config(**ck)
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.no_lbfgs:
        cfg['forward']['use_lbfgs'] = False
        cfg['inverse']['use_lbfgs'] = False
    if args.epoch_scale is not None:
        _scale_epochs(cfg, args.epoch_scale)
    _apply_overrides(cfg, args.overrides)

    # --warm-cache-only: single-process decimated-cache pre-build, then exit.
    # Broad mu/chi grid so every regime + S3 chi-fold run later hits warm cache.
    if args.warm_cache_only:
        from .regime_split import load_whitelist
        wl = load_whitelist(cfg['whitelist_csv'])
        warm_cache(cfg['data_dir'], whitelist=wl,
                   mu_values=[0.3, 0.5, 0.8], chi_values=[0.0, 0.002, 0.005, 0.008],
                   profiles=None, friction_models=cfg['friction_models'],
                   target_hz=cfg['target_hz'], cache_dir=cfg['cache_dir'])
        print("[warm-cache] done."); return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device
    print(f"[device] {device}")
    _set_global_torch_flags(device)
    np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    apply_dummy_overrides(cfg)
    init_torch_globals(device)

    if args.run_tag is not None:                       # launcher-supplied; deterministic run dir
        cfg['run_tag'] = args.run_tag
    elif args.stage in ('forward', 'both'):
        cfg['run_tag'] = build_run_tag(cfg, prefix=prefix, suffix=suffix)
    else:
        cfg['run_tag'] = args.ckpt[0].parent.name
    print(f"[run] run_tag = {cfg['run_tag']}")
    configure_figure_saving(Path(cfg['figure_dir']) / cfg['run_tag'], run_tag=cfg['run_tag'])

    rp = RobotParams().finalize(p1_wheels=p1_wheels, device=device)

    if cfg.get('regime_toml'):
        # shared regime split (observer_v1_py/regimes) -> same trajectories as A2
        tr, va, te = load_regime_split(cfg['regime_toml'], cfg)
        if not tr:
            print("[fatal] regime split produced no training trajectories"); sys.exit(1)
        tr_loader, va_loader, te_loader = build_loaders_from_lists(tr, va, te, cfg)
    else:
        n_dummy = cfg['dummy_n_trajectories'] if cfg.get('dummy') else None
        whitelist = parse_whitelist(cfg['whitelist_path'], subsample_n=n_dummy,
                                    subsample_seed=cfg['seed'])
        trajectories = load_all_arrow_trajectories(
            cfg['data_dir'], whitelist=whitelist, mu_values=cfg['mu_values'],
            chi_values=cfg['chi_values'], profiles=cfg['profiles'],
            friction_models=cfg['friction_models'], target_hz=cfg['target_hz'],
            cache_dir=cfg['cache_dir'])
        if not trajectories:
            print("[fatal] no trajectories loaded — check data_dir / filters"); sys.exit(1)
        tr_loader, va_loader, te_loader, tr, va, te = build_loaders_with_split(trajectories, cfg)

    if args.stage == 'forward':
        model, hist = train_forward(rp, tr_loader, va_loader, cfg)
        plot_history(hist, 'forward')
        fwd = evaluate_on_test(model, te_loader, rp, cfg, 'forward', 'test/forward')
        _manifest(cfg, cfg['run_tag'], ['forward'], tr, va, te)
        _write_metrics(cfg, {f"fwd_{k}": v for k, v in fwd.items()})

    elif args.stage == 'inverse':
        model = MecanumPINN(cfg, rp).to(device)
        load_phase_checkpoint(model, args.ckpt[0], map_location=device)
        model = maybe_compile_pinn(model, cfg)
        model, hist = train_inverse(rp, tr_loader, va_loader, cfg, model)
        plot_history(hist, 'inverse')
        inv = evaluate_on_test(model, te_loader, rp, cfg, 'inverse', 'test/inverse')
        muid = _report_mu(model, va_loader, te_loader, te, rp, cfg)
        _manifest(cfg, cfg['run_tag'], ['inverse'], tr, va, te, fwd_ref=str(args.ckpt[0]))
        _write_metrics(cfg, {**{f"inv_{k}": v for k, v in inv.items()}, **(muid or {})})

    elif args.stage == 'both':
        model, hist_f = train_forward(rp, tr_loader, va_loader, cfg)
        plot_history(hist_f, 'forward')
        fwd = evaluate_on_test(model, te_loader, rp, cfg, 'forward', 'test/forward')
        _manifest(cfg, cfg['run_tag'], ['forward'], tr, va, te)
        model, hist_i = train_inverse(rp, tr_loader, va_loader, cfg, model)
        plot_history(hist_i, 'inverse')
        inv = evaluate_on_test(model, te_loader, rp, cfg, 'inverse', 'test/inverse')
        muid = _report_mu(model, va_loader, te_loader, te, rp, cfg)
        _manifest(cfg, cfg['run_tag'], ['forward', 'inverse'], tr, va, te)
        _write_metrics(cfg, {**{f"fwd_{k}": v for k, v in fwd.items()},
                             **{f"inv_{k}": v for k, v in inv.items()},
                             **(muid or {})})

    elif args.stage == 'figures':
        model = MecanumPINN(cfg, rp).to(device)
        ckpt = load_phase_checkpoint(model, args.ckpt[0], map_location=device)
        model = maybe_compile_pinn(model, cfg)
        if 'history' in ckpt:
            for st in ('forward', 'inverse'):
                if st in ckpt['history']:
                    plot_history(ckpt['history'], st)
        evaluate_on_test(model, te_loader, rp, cfg, 'forward', 'test/forward')
        evaluate_on_test(model, te_loader, rp, cfg, 'inverse', 'test/inverse')
        _report_mu(model, va_loader, te_loader, te, rp, cfg)

    print("\n[done] run complete.")
