#!/usr/bin/env python
# =============================================================================
# physics_ablation_eval.py — Physics-ablation study for Mamba ForceRecon v1.
#
# Compares the FINAL trained model (Adam + L-BFGS physics refinement) against the
# ADAM-ONLY model (last Adam phase, e.g. inverse_physics.pth / forward_physics.pth)
# on both the same-subset (val) and cross-subset (test) splits.
#
# Run from code_insights/:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/physics_ablation_eval.py \
#       --final-ckpt Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/a1_S1_train_w16/inverse_lbfgs.pth
#   # explicit Adam-only checkpoint:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/physics_ablation_eval.py \
#       --final-ckpt .../inverse_lbfgs.pth --adam-ckpt .../inverse_physics.pth --stage inverse
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


def _infer_regime_from_run_tag(run_tag: str) -> str:
    """Best-effort regime inference from the run tag for legacy manifests."""
    for name in ("S1_train", "S2_train", "S3_chi_kfold"):
        if name in run_tag:
            return f"observer_v1_py/regimes/{name}.toml"
    return ""


def _config_from_checkpoint(ckpt: dict, ckpt_path: Path) -> dict:
    """Recover training config from checkpoint payload or manifest fallback."""
    cfg = ckpt.get("config")
    if cfg is not None:
        return dict(cfg)

    print("[physics-ablation] checkpoint lacks config; reconstructing from manifest/run dir")
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
                import json as _json
                metrics = _json.loads(metrics_path.read_text())
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


def _resolve_ckpt_pair(final_ckpt: Path, adam_ckpt: Path | None, stage: str):
    """Map (final, adam) checkpoint files for the requested stage(s)."""
    final_dir = final_ckpt.parent
    pairs = {}
    for st in ("forward", "inverse"):
        if stage not in ("both", st):
            continue
        final = final_ckpt if st in final_ckpt.name else final_dir / f"{st}_lbfgs.pth"
        if adam_ckpt is not None and st in adam_ckpt.name:
            adam = adam_ckpt
        else:
            adam = (adam_ckpt if adam_ckpt is not None
                    else final_dir / f"{st}_physics.pth")
        pairs[st] = (final, adam)
    return pairs


def _evaluate_stage(model_final, model_adam, va_loader, te_loader, rp, cfg, stage: str
                    ) -> dict:
    """Evaluate final and Adam-only models for one stage and return prefixed metrics."""
    final = evaluate_cross_study(model_final, va_loader, te_loader, rp, cfg, stage)
    adam = evaluate_cross_study(model_adam, va_loader, te_loader, rp, cfg, stage)
    metrics = {}
    for k, vf in final.items():
        va = adam[k]
        metrics[f"final_{stage}_{k}"] = vf
        metrics[f"adam_{stage}_{k}"] = va
        metrics[f"delta_{stage}_{k}"] = vf - va
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Physics-ablation: final model vs Adam-only model for Mamba ForceRecon.")
    ap.add_argument("--final-ckpt", type=Path, required=True,
                    help="final checkpoint .pth (or run dir; stage counterparts auto-derived)")
    ap.add_argument("--adam-ckpt", type=Path, default=None,
                    help="Adam-only checkpoint .pth (auto-derived if omitted)")
    ap.add_argument("--stage", choices=["forward", "inverse", "both"], default="both")
    args = ap.parse_args()

    final_ckpt = args.final_ckpt
    if final_ckpt.is_dir():
        final_ckpt = final_ckpt / "inverse_lbfgs.pth"
    if not final_ckpt.exists():
        raise SystemExit(f"final checkpoint not found: {final_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_stt = torch.load(final_ckpt, map_location=device, weights_only=False)
    cfg = _config_from_checkpoint(final_stt, final_ckpt)
    cfg["device"] = device
    cfg.pop("dummy", None)

    print(f"[physics-ablation] effective cfg: seq_len={cfg.get('seq_len')} stride={cfg.get('stride')} "
          f"ssm_d_model={cfg.get('ssm_d_model')} ssm_d_state={cfg.get('ssm_d_state')} "
          f"inv_window={cfg.get('inv_window')} batch={cfg.get('batch_size')}")

    if platform.system() == "Windows":
        print("[physics-ablation] Windows detected: using num_workers=0 for DataLoader stability")
        cfg["num_workers"] = 0
        cfg["persistent_workers"] = False
        cfg["pin_memory"] = False

    init_torch_globals(device)

    rp = RobotParams().finalize(p1_wheels=0.11, device=device)
    print("[physics-ablation] loading regime split...")
    tr, va, te = load_regime_split(cfg["regime_toml"], cfg)
    if not tr:
        print("[fatal] regime produced no training trajectories")
        sys.exit(1)
    print(f"[physics-ablation] building loaders (tr={len(tr)} va={len(va)} te={len(te)} trajectories)...")
    tr_loader, va_loader, te_loader = build_loaders_from_lists(tr, va, te, cfg)

    pairs = _resolve_ckpt_pair(final_ckpt, args.adam_ckpt, args.stage)
    all_metrics: dict = {}
    for st, (final_path, adam_path) in pairs.items():
        if not final_path.exists():
            print(f"[physics-ablation] skipping {st}: final checkpoint missing {final_path}")
            continue
        if not adam_path.exists():
            print(f"[physics-ablation] skipping {st}: Adam-only checkpoint missing {adam_path}")
            continue

        model_final = MecanumPINN(cfg, rp).to(device)
        load_phase_checkpoint(model_final, final_path, map_location=device)
        model_final = maybe_compile_pinn(model_final, cfg)

        model_adam = MecanumPINN(cfg, rp).to(device)
        load_phase_checkpoint(model_adam, adam_path, map_location=device)
        model_adam = maybe_compile_pinn(model_adam, cfg)

        all_metrics.update(_evaluate_stage(model_final, model_adam, va_loader,
                                           te_loader, rp, cfg, st))
        if st == "inverse":
            final_muid = evaluate_mu_id_cross(model_final, va_loader, te_loader, rp, cfg)
            adam_muid = evaluate_mu_id_cross(model_adam, va_loader, te_loader, rp, cfg)
            for k, vf in final_muid.items():
                va = adam_muid[k]
                all_metrics[f"final_inv_{k}"] = vf
                all_metrics[f"adam_inv_{k}"] = va
                all_metrics[f"delta_inv_{k}"] = vf - va

    if not all_metrics:
        print("[physics-ablation] no metrics collected; exiting")
        sys.exit(1)

    out_path = final_ckpt.parent / "physics_ablation_metrics.json"
    payload = {
        "run_tag": cfg["run_tag"],
        "regime": cfg.get("regime_toml", ""),
        "stage": args.stage,
        "final_ckpt": str(final_ckpt),
        "adam_ckpt": str(args.adam_ckpt) if args.adam_ckpt else "auto",
        **{k: (float(v) if isinstance(v, (int, float)) else v)
           for k, v in all_metrics.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[physics-ablation] wrote {out_path}")


if __name__ == "__main__":
    main()
