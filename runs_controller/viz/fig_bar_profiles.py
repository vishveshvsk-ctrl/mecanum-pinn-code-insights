#!/usr/bin/env python
"""Grouped bar: ASMC vs PID tracking error per profile, at 1x realistic noise.
y = tolerance-normalized tracking error (dimensionless, 'x over target'), so all
profiles share one axis. Velocity: (rms_vx+rms_vy over 1mm/s tol + w over 1mrad/s)/3
(exact from CSV). Pose: (max_pos/10cm-tol + max_head/0.1rad-tol)/2 (max-term proxy)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = 1.0  # realistic noise
df = pd.read_csv("runs_controller/noise_eval_10seed.csv")
df = df[df.scale == SCALE].copy()

def norm_track(r):
    if r["mode"] == "vel":   # rms in mm/s & mrad/s, tol=1mm/s & 1mrad/s -> value == number
        return (r["rms_vx_mm"] + r["rms_vy_mm"] + r["rms_w_mrad"]) / 3.0
    else:                    # pose: max_pos(cm)/10cm + max_head(rad)/0.1rad, /2
        return (r["max_pos_cm"] / 10.0 + r["max_head_rad"] / 0.1) / 2.0
df["track"] = df.apply(norm_track, axis=1)

PROFILES = ["octagon", "spin_creep", "coupled_vomega", "spiral_orbit",
            "ellipse_tangent", "ellipse_crab"]
agg = df.groupby(["controller", "trajectory"])["track"].agg(["mean", "std"])

x = np.arange(len(PROFILES)); w = 0.38
STYLE = {"asmc": ("#1f77b4", "ASMC", -w/2), "pid": ("#d62728", "PID", +w/2)}

fig, ax = plt.subplots(figsize=(11, 6))
for ctrl, (c, lbl, off) in STYLE.items():
    means = [agg.loc[(ctrl, p), "mean"] for p in PROFILES]
    stds  = [agg.loc[(ctrl, p), "std"]  for p in PROFILES]
    bars = ax.bar(x + off, means, w, yerr=stds, capsize=3, color=c, label=lbl,
                  edgecolor="black", linewidth=0.4, error_kw=dict(lw=1))
    for xi, m, s in zip(x + off, means, stds):   # value labels above the error bar cap
        ax.text(xi, (m + s) * 1.18, f"{m:.0f}" if m >= 10 else f"{m:.1f}",
                ha="center", va="bottom", fontsize=8.5, color=c, fontweight="bold")

ax.set_yscale("log")
ax.set_ylabel("tracking error  ( x over tolerance )", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(PROFILES, rotation=20, ha="right")
ax.set_title("ASMC vs PID tracking error per profile  —  at 1x realistic sensor noise\n"
             "(tolerance-normalized, mean $\\pm$ std over 10 seeds; lower = better)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, frameon=False)
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax.margins(x=0.02)
fig.tight_layout()
fig.savefig("runs_controller/viz/fig_bar_profiles.png", dpi=160, bbox_inches="tight")
print("saved runs_controller/viz/fig_bar_profiles.png")
