#!/usr/bin/env python
# =============================================================================
# make_observability_report_v2hy3.py — γ + ΔV + ω_z observability report.
#
# Thin report over v2-Hy3 run dirs. Loads each run's evaluation CSV (or re-runs
# evaluation_v2hy3 if missing), writes summary tables, and emits
# `gamma_error_by_slip.csv` for A1-v2 consumption.
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer.evaluation_v2hy3 import evaluate_observer_v2hy3


def load_eval(run_dir: Path, cfg: ObserverConfigV2Hy3, device: torch.device) -> pd.DataFrame:
    csv_path = run_dir / "eval_v2hy3.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    df = evaluate_observer_v2hy3(cfg, run_dir, device)
    df.to_csv(csv_path, index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    overall = df[(df["bin_kind"] == "overall") & (df["state"].isin(["gamma", "deltaV", "V_used"]))]
    return overall.groupby(["model", "window", "regime", "split", "state", "wheel"])[
        ["rmse_norm", "rmse_phys", "n"]].first().reset_index()


def plot_slip_binned(df: pd.DataFrame, out_path: Path, state: str):
    slip = df[(df["state"] == state) & (df["bin_kind"] == "slip")]
    if slip.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (split,), grp in slip.groupby(["split"]):
        meaned = grp.groupby("bin_center")["rmse_norm"].mean().reset_index()
        ax.plot(meaned["bin_center"], meaned["rmse_norm"], marker="o", label=split)
    ax.set_xlabel("slip speed bin center (m/s)")
    ax.set_ylabel(f"{state} RMSE (normalised)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate v2-Hy3 observability report.")
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="v2hy3 run directories (observer_v1_py/runs/*_v2hy3_*)")
    ap.add_argument("--out-dir", type=Path, default=Path("observer_v1_py/report_v2hy3"))
    ap.add_argument("--re-eval", action="store_true",
                    help="re-run evaluation even if eval_v2hy3.csv exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    for run_dir in args.runs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"[report-v2hy3] skipping {run_dir}: no metrics.json")
            continue
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        cfg = ObserverConfigV2Hy3(**metrics.get("cfg", {}))
        cfg = cfg.resolved()
        if args.re_eval and (run_dir / "eval_v2hy3.csv").exists():
            (run_dir / "eval_v2hy3.csv").unlink()
        df = load_eval(run_dir, cfg, device)
        all_rows.append(df)
        print(f"[report-v2hy3] {run_dir.name}: {len(df)} eval rows")

    if not all_rows:
        print("[report-v2hy3] no runs to report")
        return

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(args.out_dir / "eval_v2hy3_all.csv", index=False)

    summary = summarize(full)
    summary.to_csv(args.out_dir / "summary_overall.csv", index=False)
    print("\nOverall γ RMSE (normalised):")
    print(full[(full["state"] == "gamma") & (full["bin_kind"] == "overall")]
          .groupby(["split"])["rmse_norm"].mean().to_string())
    print("\nOverall V_used RMSE (physical):")
    print(full[(full["state"] == "V_used") & (full["bin_kind"] == "overall")]
          .groupby(["split"])["rmse_phys"].mean().to_string())

    slip = full[(full["state"] == "gamma") & (full["bin_kind"] == "slip")]
    if not slip.empty:
        slip.to_csv(args.out_dir / "gamma_error_by_slip.csv", index=False)

    plot_slip_binned(full, args.out_dir / "gamma_rmse_by_slip.png", "gamma")
    plot_slip_binned(full, args.out_dir / "vused_rmse_by_slip.png", "V_used")
    print(f"[report-v2hy3] report -> {args.out_dir}")


if __name__ == "__main__":
    main()
