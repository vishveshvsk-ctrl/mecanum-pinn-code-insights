#!/usr/bin/env python
# =============================================================================
# cross_eval.py — Cross-subset evaluation for Mamba ForceRecon v1.
#
# Loads a finished checkpoint and evaluates it on the test split of another
# regime. Primary use case: S1-trained checkpoint on S2 data and vice-versa,
# giving an explicit same-vs-cross generalization signal.
#
# Run from code_insights/:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/cross_eval.py \
#       --ckpt Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/a1_S1_train_w16/inverse_lbfgs.pth
#   # explicit target regime:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/cross_eval.py \
#       --ckpt .../inverse_lbfgs.pth --regime observer_v1_py/regimes/S2_train.toml
# =============================================================================
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from mecanum_pinn.config import build_config
from mecanum_pinn.data import (build_loaders_from_lists, init_torch_globals,
                               load_regime_split)
from mecanum_pinn.evaluation import evaluate_cross_study, evaluate_mu_id_cross
from mecanum_pinn.manifest import load_manifest_at
from mecanum_pinn.models import MecanumPINN, maybe_compile_pinn
from mecanum_pinn.physics import RobotParams
from mecanum_pinn.training import load_phase_checkpoint


# Default opposite regime for the S1/S2 excitation-coverage 2-fold.
_OPPOSITE_REGIME = {
    "S1_train": "observer_v1_py/regimes/S2_train.toml",
    "S2_train": "observer_v1_py/regimes/S1_train.toml",
}


def _infer_regime_from_run_tag(run_tag: str) -> str:
    """Best-effort regime inference from the run tag for legacy manifests."""
    for name in ("S1_train", "S2_train", "S3_chi_kfold"):
        if name in run_tag:
            return f"observer_v1_py/regimes/{name}.toml"
    return ""


def _config_from_checkpoint(ckpt: dict, ckpt_path: Path) -> dict:
    """Recover the training config from the checkpoint payload if present,
    otherwise fall back to the checkpoint's manifest + build_config defaults."""
    cfg = ckpt.get("config")
    if cfg is not None:
        return dict(cfg)

    manifest = load_manifest_at(ckpt_path) or {}
    scope = manifest.get("scope", {})
    summary = manifest.get("config_summary", {})
    run_dir_name = ckpt_path.parent.name

    cfg = build_config()
    cfg["profiles"] = scope.get("profiles", cfg["profiles"])
    cfg["mu_values"] = scope.get("mu_values", cfg["mu_values"])
    cfg["chi_values"] = scope.get("chi_values", cfg["chi_values"])
    cfg["data_dir"] = Path(scope.get("data_dir", cfg["data_dir"]))
    cfg["whitelist_path"] = Path(scope.get("whitelist_path", cfg["whitelist_path"]))
    cfg["seed"] = scope.get("seed", cfg["seed"])

    # regime: manifest (new) -> metrics.json (old) -> run-tag/dir-name inference
    regime = manifest.get("regime", "")
    run_tag = manifest.get("run_tag") or run_dir_name
    if not regime:
        metrics_path = ckpt_path.parent / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                regime = metrics.get("regime", "")
                run_tag = metrics.get("run_tag") or run_tag
            except Exception:
                pass
    if not regime:
        regime = _infer_regime_from_run_tag(run_tag)
        if not regime:
            regime = _infer_regime_from_run_tag(run_dir_name)
    cfg["regime_toml"] = regime

    cfg.update({k: v for k, v in summary.items() if k in cfg})
    return cfg


def _resolve_target_regime(train_regime: str, override: Path | None) -> str:
    if override is not None:
        return str(override)
    name = Path(train_regime).stem if train_regime else ""
    opp = _OPPOSITE_REGIME.get(name)
    if not opp:
        raise SystemExit(
            f"Cannot infer opposite regime for {train_regime!r}; pass --regime explicitly")
    return opp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-subset evaluation for Mamba ForceRecon v1.")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="trained checkpoint .pth to evaluate")
    ap.add_argument("--regime", type=Path, default=None,
                    help="target regime TOML (default: opposite of training regime)")
    ap.add_argument("--stage", choices=["forward", "inverse", "both"], default="both")
    ap.add_argument("--run-tag", default=None,
                    help="output run-tag (default: <train_run_tag>_on_<target_regime>)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(ckpt, args.ckpt)
    cfg["device"] = device
    cfg.pop("dummy", None)                 # never run cross-eval in dummy mode

    # Windows spawn + PyArrow/CUDA context in DataLoader workers can trigger native
    # crashes; force single-process loading for evaluation.
    if platform.system() == "Windows":
        cfg["num_workers"] = 0
        cfg["persistent_workers"] = False
        cfg["pin_memory"] = False

    init_torch_globals(device)

    train_regime = cfg.get("regime_toml", "")
    target_regime = _resolve_target_regime(train_regime, args.regime)
    cfg["regime_toml"] = target_regime

    target_name = Path(target_regime).stem
    cfg["run_tag"] = args.run_tag or f"{cfg['run_tag']}_on_{target_name}"

    rp = RobotParams().finalize(p1_wheels=0.11, device=device)
    tr, va, te = load_regime_split(cfg["regime_toml"], cfg)
    if not tr:
        print("[fatal] target regime produced no training trajectories")
        sys.exit(1)
    tr_loader, va_loader, te_loader = build_loaders_from_lists(tr, va, te, cfg)

    model = MecanumPINN(cfg, rp).to(device)
    load_phase_checkpoint(model, args.ckpt, map_location=device)
    model = maybe_compile_pinn(model, cfg)

    metrics: dict = {}
    if args.stage in ("forward", "both"):
        fwd = evaluate_cross_study(model, va_loader, te_loader, rp, cfg, "forward")
        metrics.update({f"fwd_{k}": v for k, v in fwd.items()})
    if args.stage in ("inverse", "both"):
        inv = evaluate_cross_study(model, va_loader, te_loader, rp, cfg, "inverse")
        metrics.update({f"inv_{k}": v for k, v in inv.items()})
        muid = evaluate_mu_id_cross(model, va_loader, te_loader, rp, cfg)
        metrics.update(muid)

    out_path = args.ckpt.parent / "cross_metrics.json"
    payload = {
        "run_tag": cfg["run_tag"],
        "train_regime": train_regime,
        "eval_regime": target_regime,
        "stage": args.stage,
        **{k: (float(v) if isinstance(v, (int, float)) else v)
           for k, v in metrics.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[cross-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
