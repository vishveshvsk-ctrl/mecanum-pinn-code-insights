#!/usr/bin/env python3
"""Compare FBDF vs TRBDF2 evaluation summaries (stdlib only)."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / "reports"

def load_summary(folder):
    p = ROOT / folder / "summary.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def compare(name, fbdf_folder, trbdf2_folder, keys):
    f_rows = load_summary(fbdf_folder)
    t_rows = load_summary(trbdf2_folder)
    if f_rows is None or t_rows is None:
        print(f"[{name}] missing summary; skipping")
        return

    # Index TRBDF2 rows by (controller, trajectory, sensor_noise)
    t_idx = {}
    for r in t_rows:
        key = (r["controller"], r["trajectory"], r["sensor_noise"])
        t_idx[key] = r

    print(f"\n=== {name} ===")
    header = ["controller", "trajectory", "sensor_noise"] + [
        f"{k}_trbdf2" for k in keys] + [
        f"{k}_fbdf" for k in keys] + [
        f"{k}_diff" for k in keys] + [
        f"{k}_pct" for k in keys]
    print(",".join(header))
    max_pct = {k: 0.0 for k in keys}
    for r in f_rows:
        key = (r["controller"], r["trajectory"], r["sensor_noise"])
        t = t_idx.get(key)
        if t is None:
            continue
        out = list(key)
        for k in keys:
            fv = float(r[k])
            tv = float(t[k])
            diff = fv - tv
            pct = 100 * diff / (abs(tv) if tv != 0 else 1e-12)
            out += [f"{tv:.6g}", f"{fv:.6g}", f"{diff:.6g}", f"{pct:.2f}"]
            max_pct[k] = max(max_pct[k], abs(pct))
        print(",".join(out))

    print("\nMax absolute pct changes:")
    for k in keys:
        print(f"  {k}: {max_pct[k]:.2f}%")

def main():
    compare("Clean ESKF feedback",
            "controller_eskf_pose_fbdf",
            "controller_eskf_pose",
            ["tracking_mean", "ce_mean", "chatter_mean",
             "ctrl_rmse_pos_mean", "ctrl_rmse_heading_mean"])

    compare("Realistic ESKF feedback",
            "controller_eskf_pose_realistic_fbdf",
            "controller_eskf_pose_realistic",
            ["tracking_mean", "tracking_std", "ce_mean", "chatter_mean",
             "ctrl_rmse_pos_mean", "ctrl_rmse_heading_mean"])

if __name__ == "__main__":
    main()
