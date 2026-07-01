#!/usr/bin/env python
# =============================================================================
# physics_ablation_eval.py — Physics-ablation study for the state observer (A2).
#
# Compares the FINAL trained model (checkpoint.pt) against the ADAM-ONLY model
# (last fully-Adam phase snapshot, by default phase_ckpts/physics_*.pt) on both
# the same-subset (val) and cross-subset (test) splits.
#
# Run from code_insights/:
#   python observer_v1_py/physics_ablation_eval.py \
#       --final-ckpt observer_v1_py/runs/S1_train_w32/checkpoint.pt
#   # explicit Adam-only snapshot:
#   python observer_v1_py/physics_ablation_eval.py \
#       --final-ckpt observer_v1_py/runs/S1_train_w32/checkpoint.pt \
#       --adam-ckpt observer_v1_py/runs/S1_train_w32/phase_ckpts/physics_epXXX.pt
# =============================================================================
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.feather  # noqa: F401  (Windows load-order lock)
import torch

from mecanum_observer import data as D
from mecanum_observer.config import ObserverConfig
from mecanum_observer.evaluation import evaluate_observer_model
from mecanum_observer.models import build_model


def _latest_phase_snapshot(run_dir: Path, pattern: str = "physics_*.pt") -> Path | None:
    snap_dir = run_dir / "phase_ckpts"
    if not snap_dir.exists():
        return None
    snaps = sorted(snap_dir.glob(pattern))
    return snaps[-1] if snaps else None


def _aggregate(df):
    """Return overall and per-state normalised RMSE for same/cross splits."""
    ov = df[df.bin_kind == "overall"]
    metrics = {}
    for split in ("same_subset", "cross_subset"):
        sub = ov[ov.split == split]
        if sub.empty:
            continue
        key = "same" if split == "same_subset" else "cross"
        metrics[f"overall_rmse_norm_{key}"] = float(sub.rmse_norm.mean())
        metrics[f"overall_rmse_phys_{key}"] = float(sub.rmse_phys.mean())
        for state, grp in sub.groupby("state"):
            metrics[f"{state}_rmse_norm_{key}"] = float(grp.rmse_norm.mean())
            metrics[f"{state}_rmse_phys_{key}"] = float(grp.rmse_phys.mean())
    return metrics


def _evaluate_pair(model_final, model_adam, cfg, nrm, device):
    df_final = evaluate_observer_model(model_final, cfg, nrm, device)
    df_adam = evaluate_observer_model(model_adam, cfg, nrm, device)
    final = _aggregate(df_final)
    adam = _aggregate(df_adam)
    metrics = {}
    for k in sorted(final.keys()):
        vf, va = final[k], adam[k]
        metrics[f"final_{k}"] = vf
        metrics[f"adam_{k}"] = va
        metrics[f"delta_{k}"] = vf - va
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Physics-ablation: final observer vs Adam-only observer.")
    ap.add_argument("--final-ckpt", type=Path, required=True,
                    help="final checkpoint.pt (or run dir)")
    ap.add_argument("--adam-ckpt", type=Path, default=None,
                    help="Adam-only phase snapshot (auto-derived if omitted)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for physics_ablation_metrics.json")
    args = ap.parse_args()

    final_ckpt = args.final_ckpt
    if final_ckpt.is_dir():
        final_ckpt = final_ckpt / "checkpoint.pt"
    if not final_ckpt.exists():
        raise SystemExit(f"final checkpoint not found: {final_ckpt}")

    run_dir = final_ckpt.parent
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_stt = torch.load(final_ckpt, map_location=device, weights_only=False)
    cfg = ObserverConfig(**final_stt["cfg"]).resolved()

    nrm_path = run_dir / "norm.npz"
    if not nrm_path.exists():
        raise SystemExit(f"normalizer not found: {nrm_path}")
    nrm = D.Normalizer.from_npz(nrm_path)

    adam_ckpt = args.adam_ckpt or _latest_phase_snapshot(run_dir)
    if adam_ckpt is None or not adam_ckpt.exists():
        raise SystemExit(f"Adam-only checkpoint not found; pass --adam-ckpt")

    model_final = build_model(cfg).to(device)
    model_final.load_state_dict(final_stt["model"]); model_final.eval()

    adam_stt = torch.load(adam_ckpt, map_location=device, weights_only=False)
    model_adam = build_model(cfg).to(device)
    model_adam.load_state_dict(adam_stt["model"]); model_adam.eval()

    metrics = _evaluate_pair(model_final, model_adam, cfg, nrm, device)

    out_path = out_dir / "physics_ablation_metrics.json"
    payload = {
        "run_tag": cfg.run_tag,
        "regime": cfg.regime_name,
        "final_ckpt": str(final_ckpt),
        "adam_ckpt": str(adam_ckpt),
        **{k: (float(v) if isinstance(v, (int, float)) else v)
           for k, v in metrics.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[physics-ablation] wrote {out_path}")


if __name__ == "__main__":
    main()
