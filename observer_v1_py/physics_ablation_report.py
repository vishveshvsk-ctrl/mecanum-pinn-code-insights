#!/usr/bin/env python
# =============================================================================
# physics_ablation_report.py — Aggregate physics-ablation metrics for A2.
#
# Discovers run dirs under observer_v1_py/runs/, reads each run's
# physics_ablation_metrics.json (written by physics_ablation_eval.py), and
# produces a single CSV plus a delta table.
#
# Run from code_insights/:
#   python observer_v1_py/physics_ablation_report.py
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


PKG_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PKG_ROOT / "runs"
REPORT_CSV = PKG_ROOT / "report" / "physics_ablation_report.csv"


def _fmt(v) -> str:
    return f"{v:.4e}" if isinstance(v, (int, float)) else str(v)


def collect_rows(runs_dir: Path) -> List[dict]:
    rows: List[dict] = []
    if not runs_dir.exists():
        return rows
    for run_dir in sorted(runs_dir.iterdir()):
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
        row: Dict[str, object] = {
            "run_tag": metrics.get("run_tag", run_dir.name),
            "regime": metrics.get("regime", ""),
        }
        # Collect all final/adam/delta keys as-is
        for k, v in metrics.items():
            if k in ("run_tag", "regime", "final_ckpt", "adam_ckpt"):
                continue
            row[k] = v
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate observer physics-ablation report.")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    args = ap.parse_args()

    rows = collect_rows(args.runs_dir)
    if not rows:
        print(f"[report] no physics-ablation records found under {args.runs_dir}")
        return

    fieldnames = ["run_tag", "regime"]
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
    print(f"  {'run_tag':<50} {'overall_same':>14} {'overall_cross':>14}")
    for r in rows:
        print(f"  {r['run_tag']:<50} {_fmt(r.get('delta_overall_rmse_norm_same')):>14} "
              f"{_fmt(r.get('delta_overall_rmse_norm_cross')):>14}")


if __name__ == "__main__":
    main()
