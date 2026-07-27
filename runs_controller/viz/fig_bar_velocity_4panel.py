#!/usr/bin/env python
"""2x2 panel (one per noise level): ASMC vs PID velocity-tracking error per
profile, ellipse (pose) EXCLUDED. Linear y-axis + error bars (now viable since
all 4 remaining profiles share a scale). Tolerance-normalized velocity tracking
= (rms_vx + rms_vy + rms_w)/3 over the 1mm/s & 1mrad/s tolerances."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("runs_controller/noise_eval_10seed.csv")
df = df[df["mode"] == "vel"].copy()                       # velocity profiles only
df["track"] = (df.rms_vx_mm + df.rms_vy_mm + df.rms_w_mrad) / 3.0

PROFILES = ["octagon", "spin_creep", "coupled_vomega", "spiral_orbit"]
SCALES = [(0.0, "clean"), (1.0, "1x  (realistic)"), (2.0, "2x"), (5.0, "5x  (stress)")]
STYLE = {"asmc": ("#1f77b4", "ASMC", -0.19), "pid": ("#d62728", "PID", +0.19)}
x = np.arange(len(PROFILES)); w = 0.38
hi = df.groupby(["controller", "trajectory", "scale"])["track"].mean().max() * 1.15

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
for ax, (sc, title) in zip(axes.ravel(), SCALES):
    g = df[df.scale == sc].groupby(["controller", "trajectory"])["track"].agg(["mean", "std"])
    for ctrl, (c, lbl, off) in STYLE.items():
        means = [g.loc[(ctrl, p), "mean"] for p in PROFILES]
        stds  = [(g.loc[(ctrl, p), "std"] if not np.isnan(g.loc[(ctrl, p), "std"]) else 0.0)
                 for p in PROFILES]
        ax.bar(x + off, means, w, yerr=stds, capsize=3.5, color=c, label=lbl,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=1))
        for xi, m, s in zip(x + off, means, stds):
            ax.text(xi, m + s + hi*0.015, f"{m:.0f}" if m >= 10 else f"{m:.1f}",
                    ha="center", va="bottom", fontsize=8, color=c, fontweight="bold")
    ax.set_ylim(0, hi)
    ax.set_title(f"noise: {title}", fontsize=11.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(PROFILES, rotation=12, ha="center", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
for ax in axes[:, 0]:
    ax.set_ylabel("velocity tracking error\n( x over tolerance )", fontsize=10)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, 0.985))
fig.suptitle("ASMC vs PID velocity-tracking error per profile, across noise levels  "
             "(ellipse excluded; mean $\\pm$ std / 10 seeds; lower = better)",
             y=1.02, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("runs_controller/viz/fig_bar_velocity_4panel.png", dpi=155, bbox_inches="tight")
print("saved runs_controller/viz/fig_bar_velocity_4panel.png")
