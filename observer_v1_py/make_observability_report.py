#!/usr/bin/env python
# =============================================================================
# make_observability_report.py — aggregate trained runs into the observability
# signature: state_observability.csv + static matplotlib figures.
#
# Discovers run dirs (<out-dir>/{ssm,gru}_w<W>/checkpoint.pt), evaluates each on
# its held-out test split, and writes:
#   * state_observability.csv  — per (model, window, wheel, state, bin) RMSE
#   * fig_overall_rmse.png     — normalised RMSE per state, SSM vs GRU
#   * fig_omega_z_vs_spin.png  — derived omega_z error vs |omega_z| (chi bound)
#   * fig_z_vs_slip.png        — bristle-z error vs slip (the non-uniqueness floor)
#
# Run on the WSL machine (needs torch + the Arrow data). Figures are static
# matplotlib (no widgets), per project convention.
# =============================================================================
from __future__ import annotations

# pyarrow before torch (Windows load-order lock; see v14 train.py).
import pyarrow.feather  # noqa: F401

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mecanum_observer.config import ObserverConfig, TARGET_STATES

# torch is imported lazily (only collect()/_load_cfg need it) so the figure /
# table logic can run in a torch-free env.


def _load_cfg(run_dir: Path) -> ObserverConfig:
    import torch
    st = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    return ObserverConfig(**st["cfg"]).resolved()


def collect(out_dir: Path, pattern: str = "*_w*") -> pd.DataFrame:
    import torch
    from mecanum_observer.evaluation import evaluate_observer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames: List[pd.DataFrame] = []
    for run_dir in sorted(out_dir.glob(pattern)):
        if not (run_dir / "checkpoint.pt").exists():
            continue
        cfg = _load_cfg(run_dir)
        print(f"[report] evaluating {run_dir.name}")
        frames.append(evaluate_observer(cfg, run_dir, device))
    if not frames:
        raise SystemExit(f"no trained runs under {out_dir}")
    return pd.concat(frames, ignore_index=True)


def fig_overall(df: pd.DataFrame, out: Path) -> None:
    # headline = cross-subset (the transfer metric); same-subset has its own fig
    d = df[(df.bin_kind == "overall") & (df.split == "cross_subset")].groupby(
        ["model", "window", "state"], as_index=False)["rmse_norm"].mean()
    states = list(TARGET_STATES) + ["omega_z_derived"]
    combos = sorted({(m, w) for m, w in zip(d.model, d.window)})
    x = np.arange(len(states)); width = 0.8 / max(len(combos), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, (m, w) in enumerate(combos):
        sub = d[(d.model == m) & (d.window == w)].set_index("state")
        vals = [sub.loc[s, "rmse_norm"] if s in sub.index else np.nan for s in states]
        ax.bar(x + j * width, vals, width, label=f"{m} w{w}")
    ax.set_xticks(x + width * (len(combos) - 1) / 2)
    ax.set_xticklabels(states)
    ax.set_ylabel("normalised RMSE (std-relative)")
    ax.set_title("Per-state observability — lower is more observable")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_same_vs_cross(df: pd.DataFrame, out: Path) -> None:
    """Per-state same-subset (val) vs cross-subset (test) RMSE — the headline
    generalization/anti-fingerprinting signal. Averaged over wheels & runs; one
    panel, two bars per state. Uses the SSM (deployed) model."""
    d = df[(df.bin_kind == "overall") & (df.model == "ssm")]
    if d.empty or d.split.nunique() < 2:
        print("[report] need both splits for same-vs-cross, skipping")
        return
    g = d.groupby(["state", "split"], as_index=False)["rmse_norm"].mean()
    states = list(TARGET_STATES) + ["omega_z_derived"]
    x = np.arange(len(states)); width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, (split, lab) in enumerate([("same_subset", "same (val)"),
                                      ("cross_subset", "cross (test)")]):
        sub = g[g.split == split].set_index("state")
        vals = [sub.loc[s, "rmse_norm"] if s in sub.index else np.nan for s in states]
        ax.bar(x + j * width, vals, width, label=lab)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(states)
    ax.set_ylabel("normalised RMSE")
    ax.set_title("Same- vs cross-subset reconstruction (gap = fingerprinting)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_binned(df: pd.DataFrame, state: str, bin_kind: str, xlabel: str,
               out: Path) -> None:
    d = df[(df.state == state) & (df.bin_kind == bin_kind)
           & (df.split == "cross_subset")].groupby(
        ["model", "window", "bin_center"], as_index=False)["rmse_norm"].mean()
    if d.empty:
        print(f"[report] no data for {state}/{bin_kind}, skipping {out.name}")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (m, w), sub in d.groupby(["model", "window"]):
        sub = sub.sort_values("bin_center")
        ax.plot(sub.bin_center, sub.rmse_norm, marker="o", label=f"{m} w{w}")
    ax.set_xlabel(xlabel); ax.set_ylabel("normalised RMSE")
    ax.set_title(f"{state} reconstruction vs {xlabel}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the observability report.")
    ap.add_argument("--out-dir", type=Path, default=Path("observer_v1_py/runs"))
    ap.add_argument("--report-dir", type=Path, default=Path("observer_v1_py/report"))
    ap.add_argument("--pattern", default="*_w*",
                    help="run-dir glob (e.g. '*_non_phys_max_norm' to isolate one norm)")
    args = ap.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = collect(args.out_dir, args.pattern)
    csv = args.report_dir / "state_observability.csv"
    df.to_csv(csv, index=False)
    print(f"[report] wrote {len(df)} rows -> {csv}")

    fig_overall(df, args.report_dir / "fig_overall_rmse.png")
    fig_same_vs_cross(df, args.report_dir / "fig_same_vs_cross.png")
    fig_binned(df, "omega_z_derived", "wz", "|omega_z| (rad/s)",
               args.report_dir / "fig_omega_z_vs_spin.png")
    fig_binned(df, "gamma", "wz", "|omega_z| (rad/s)",
               args.report_dir / "fig_gamma_vs_spin.png")
    fig_binned(df, "zx", "slip", "|Vp| slip (m/s)",
               args.report_dir / "fig_zx_vs_slip.png")
    fig_binned(df, "zs", "wz", "|omega_z| (rad/s)",
               args.report_dir / "fig_zs_vs_spin.png")

    # Headline: per-state same vs cross + gap (overall, mean over wheels & runs).
    ov = df[df.bin_kind == "overall"]
    piv = ov.pivot_table(index="state", columns="split", values="rmse_norm", aggfunc="mean")
    for col in ("same_subset", "cross_subset"):
        if col not in piv.columns:
            piv[col] = np.nan
    piv["gap"] = piv["cross_subset"] - piv["same_subset"]
    piv = piv.sort_values("cross_subset")
    print("\n[observability] normalised RMSE (low = observable)")
    print(f"  {'state':<18} {'same(val)':>10} {'cross(test)':>12} {'gap':>8}")
    for s, r in piv.iterrows():
        print(f"  {s:<18} {r['same_subset']:>10.4f} {r['cross_subset']:>12.4f} {r['gap']:>8.4f}")


if __name__ == "__main__":
    main()
