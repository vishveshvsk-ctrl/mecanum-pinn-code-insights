#!/usr/bin/env python
# =============================================================================
# physics_ablation_report.py — Aggregate physics-ablation metrics for A1.
#
# Discovers run dirs under Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/,
# reads each run's physics_ablation_metrics.json (written by physics_ablation_eval.py),
# and produces a single CSV plus a delta table.
#
# Run from code_insights/:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/physics_ablation_report.py
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


PKG_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PKG_ROOT / "runs" / "checkpoints"
REPORT_CSV = PKG_ROOT / "runs" / "physics_ablation_report.csv"


def _get(d: dict, key: str, default=None):
    return d.get(key, default)


def _fmt(v) -> str:
    return f"{v:.4e}" if isinstance(v, (int, float)) else str(v)


def _headline_metrics(metrics: dict, stage: str) -> Dict[str, Optional[float]]:
    """Extract the headline same/cross totals and mu-ID MAE for a stage."""
    prefix = f"{stage}_"
    return {
        f"{stage}_same_total_final": _get(metrics, f"final_{prefix}same_total"),
        f"{stage}_same_total_adam":  _get(metrics, f"adam_{prefix}same_total"),
        f"{stage}_same_total_delta": _get(metrics, f"delta_{prefix}same_total"),
        f"{stage}_cross_total_final": _get(metrics, f"final_{prefix}cross_total"),
        f"{stage}_cross_total_adam":  _get(metrics, f"adam_{prefix}cross_total"),
        f"{stage}_cross_total_delta": _get(metrics, f"delta_{prefix}cross_total"),
        f"{stage}_mu_mae_inv_same_final": _get(metrics, "final_inv_same_mu_mae_inv"),
        f"{stage}_mu_mae_inv_same_adam":  _get(metrics, "adam_inv_same_mu_mae_inv"),
        f"{stage}_mu_mae_inv_same_delta": _get(metrics, "delta_inv_same_mu_mae_inv"),
        f"{stage}_mu_mae_inv_cross_final": _get(metrics, "final_inv_cross_mu_mae_inv"),
        f"{stage}_mu_mae_inv_cross_adam":  _get(metrics, "adam_inv_cross_mu_mae_inv"),
        f"{stage}_mu_mae_inv_cross_delta": _get(metrics, "delta_inv_cross_mu_mae_inv"),
    }


def collect_rows(ckpt_dir: Path) -> List[dict]:
    rows: List[dict] = []
    if not ckpt_dir.exists():
        return rows
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        path = run_dir / "physics_ablation_metrics.json"
        if not path.exists():
            continue
        try:
            with open(path) as fh:
                metrics = json.load(fh)
        except Exception as e:
            print(f"[report] skipping bad physics_ablation_metrics.json in {run_dir.name}: {e}")
            continue
        base = {
            "run_tag": metrics.get("run_tag", run_dir.name),
            "regime": metrics.get("regime", ""),
            "stage": metrics.get("stage", ""),
        }
        row = dict(base)
        for st in ("forward", "inverse"):
            if any(k.startswith(f"final_{st}_") for k in metrics):
                row.update(_headline_metrics(metrics, st))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate Mamba ForceRecon physics-ablation report.")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    args = ap.parse_args()

    rows = collect_rows(args.ckpt_dir)
    if not rows:
        print(f"[report] no physics-ablation records found under {args.ckpt_dir}")
        return

    fieldnames = ["run_tag", "regime", "stage"]
    # Use first row to infer available columns
    for k in rows[0]:
        if k not in fieldnames:
            fieldnames.append(k)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fieldnames})
    print(f"[report] wrote {len(rows)} rows -> {args.report_csv}")

    print("\n[physics-ablation] headline deltas (final - adam; negative = final is better)")
    hdr = f"  {'run_tag':<50} {'stage':<8} {'fwd_same':>12} {'fwd_cross':>12} " \
          f"{'inv_same':>12} {'inv_cross':>12} {'mu_same':>12} {'mu_cross':>12}"
    print(hdr)
    for r in rows:
        print(f"  {r['run_tag']:<50} {r['stage']:<8} "
              f"{_fmt(r.get('forward_same_total_delta')):>12} "
              f"{_fmt(r.get('forward_cross_total_delta')):>12} "
              f"{_fmt(r.get('inverse_same_total_delta')):>12} "
              f"{_fmt(r.get('inverse_cross_total_delta')):>12} "
              f"{_fmt(r.get('inverse_mu_mae_inv_same_delta')):>12} "
              f"{_fmt(r.get('inverse_mu_mae_inv_cross_delta')):>12}")


if __name__ == "__main__":
    main()
