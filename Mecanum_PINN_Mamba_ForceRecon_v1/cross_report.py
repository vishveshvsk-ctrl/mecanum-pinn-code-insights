#!/usr/bin/env python
# =============================================================================
# cross_report.py — Aggregate same-vs-cross generalization metrics.
#
# Discovers trained runs under Mecanum_PINN_Mamba_ForceRecon_v1/runs/checkpoints/,
# reads each run's metrics.json (same/cross from training) and any cross_metrics.json
# produced by cross_eval.py, and writes a single CSV plus a same-vs-cross gap table.
#
# Run from code_insights/:
#   python Mecanum_PINN_Mamba_ForceRecon_v1/cross_report.py
#   python Mecanum_PINN_Mamba_ForceRecon_v1/cross_report.py --out-dir ... --report-csv ...
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT_DIRS = [PKG_ROOT / "runs" / "checkpoints",
                     PKG_ROOT / "checkpoints_mamba_v1"]
REPORT_CSV = PKG_ROOT / "runs" / "cross_report.csv"


def _get(d: dict, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _gap(cross: Optional[float], same: Optional[float]) -> Optional[float]:
    if cross is None or same is None:
        return None
    return float(cross) - float(same)


def _extract_core(metrics: dict) -> dict:
    """Pull the headline same/cross numbers from a metrics dict."""
    return {
        "fwd_same_total":  _get(metrics, ["fwd_same_total"]),
        "fwd_cross_total": _get(metrics, ["fwd_cross_total"]),
        "inv_same_total":  _get(metrics, ["inv_same_total"]),
        "inv_cross_total": _get(metrics, ["inv_cross_total"]),
        "mu_mae_inv_same":  _get(metrics, ["same_mu_mae_inv"]),
        "mu_mae_inv_cross": _get(metrics, ["cross_mu_mae_inv"]),
    }


def _row_from_metrics(run_tag: str, regime: str, metrics: dict,
                      source: str = "training") -> dict:
    core = _extract_core(metrics)
    return {
        "run_tag": run_tag,
        "source": source,
        "train_regime": regime,
        "eval_regime": regime,
        **core,
        "fwd_gap": _gap(core["fwd_cross_total"], core["fwd_same_total"]),
        "inv_gap": _gap(core["inv_cross_total"], core["inv_same_total"]),
        "mu_mae_inv_gap": _gap(core["mu_mae_inv_cross"], core["mu_mae_inv_same"]),
    }


def collect_rows(ckpt_dir: Path) -> List[dict]:
    rows: List[dict] = []
    if not ckpt_dir.exists():
        return rows
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as fh:
                    metrics = json.load(fh)
            except Exception as e:
                print(f"[report] skipping bad metrics.json in {run_dir.name}: {e}")
                continue
            rows.append(_row_from_metrics(
                metrics.get("run_tag", run_dir.name),
                metrics.get("regime", ""), metrics, source="training"))

        cross_path = run_dir / "cross_metrics.json"
        if cross_path.exists():
            try:
                with open(cross_path) as fh:
                    cross = json.load(fh)
            except Exception as e:
                print(f"[report] skipping bad cross_metrics.json in {run_dir.name}: {e}")
                continue
            core = _extract_core(cross)
            rows.append({
                "run_tag": cross.get("run_tag", run_dir.name),
                "source": "cross_eval",
                "train_regime": cross.get("train_regime", ""),
                "eval_regime": cross.get("eval_regime", ""),
                **core,
                "fwd_gap": _gap(core["fwd_cross_total"], core["fwd_same_total"]),
                "inv_gap": _gap(core["inv_cross_total"], core["inv_same_total"]),
                "mu_mae_inv_gap": _gap(core["mu_mae_inv_cross"], core["mu_mae_inv_same"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate Mamba ForceRecon cross-subset generalization report.")
    ap.add_argument("--ckpt-dir", type=Path, action="append", default=[],
                    help="checkpoint root(s) to scan (default: runs/checkpoints + checkpoints_mamba_v1)")
    ap.add_argument("--report-csv", type=Path, default=REPORT_CSV)
    args = ap.parse_args()

    ckpt_dirs = args.ckpt_dir if args.ckpt_dir else DEFAULT_CKPT_DIRS
    rows: List[dict] = []
    for d in ckpt_dirs:
        rows.extend(collect_rows(d))
    if not rows:
        print(f"[report] no runs found under {[str(d) for d in ckpt_dirs]}")
        return

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = ["run_tag", "source", "train_regime", "eval_regime",
                  "fwd_same_total", "fwd_cross_total", "fwd_gap",
                  "inv_same_total", "inv_cross_total", "inv_gap",
                  "mu_mae_inv_same", "mu_mae_inv_cross", "mu_mae_inv_gap"]
    with open(args.report_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"[report] wrote {len(rows)} rows -> {args.report_csv}")

    print("\n[cross-report] same-vs-cross gap (cross - same; positive = worse on cross)")
    print(f"  {'run_tag':<50} {'source':<12} {'fwd_gap':>12} {'inv_gap':>12} {'mu_gap':>12}")
    for r in rows:
        print(f"  {r['run_tag']:<50} {r['source']:<12} "
              f"{_fmt(r['fwd_gap']):>12} {_fmt(r['inv_gap']):>12} {_fmt(r['mu_mae_inv_gap']):>12}")


def _fmt(v) -> str:
    return f"{v:.4e}" if isinstance(v, (int, float)) else str(v)


if __name__ == "__main__":
    main()
