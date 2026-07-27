#!/usr/bin/env python
"""2x2 panel: ASMC vs PID grouped-bar tracking error per profile, one panel per
noise level (clean, 1x, 2x, 5x). Shared log-y. Tolerance-normalized tracking
(velocity exact; pose = max-term proxy). Error bars = std over 10 seeds (clean=1 seed)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("runs_controller/noise_eval_10seed.csv")
def norm_track(r):
    if r["mode"] == "vel":
        return (r["rms_vx_mm"] + r["rms_vy_mm"] + r["rms_w_mrad"]) / 3.0
    return (r["max_pos_cm"] / 10.0 + r["max_head_rad"] / 0.1) / 2.0
df["track"] = df.apply(norm_track, axis=1)

PROFILES = ["octagon", "spin_creep", "coupled_vomega", "spiral_orbit",
            "ellipse_tangent", "ellipse_crab"]
SCALES = [(0.0, "clean"), (1.0, "1x  (realistic)"), (2.0, "2x"), (5.0, "5x  (stress)")]
STYLE = {"asmc": ("#1f77b4", "ASMC", -0.19), "pid": ("#d62728", "PID", +0.19)}
x = np.arange(len(PROFILES)); w = 0.38

# shared y-limits across all panels
hi_lin = (df.groupby(["controller","trajectory","scale"])["track"].mean().max()) * 1.12

plt.rcParams.update({"font.size": 9.5})
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
for ax, (sc, title) in zip(axes.ravel(), SCALES):
    sub = df[df.scale == sc]
    g = sub.groupby(["controller", "trajectory"])["track"].agg(["mean", "std"])
    for ctrl, (c, lbl, off) in STYLE.items():
        means = [g.loc[(ctrl, p), "mean"] for p in PROFILES]
        stds  = [(g.loc[(ctrl, p), "std"] if not np.isnan(g.loc[(ctrl, p), "std"]) else 0.0)
                 for p in PROFILES]
        ax.bar(x + off, means, w, yerr=stds, capsize=2.5, color=c, label=lbl,
               edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.9))
    ax.set_ylim(0, hi_lin)
    ax.set_title(f"noise: {title}", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(PROFILES, rotation=25, ha="right", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)  # tolerance line
for ax in axes[:, 0]:
    ax.set_ylabel("tracking error ( x over tol )", fontsize=10)

h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, 0.985))
fig.suptitle("ASMC vs PID tracking error per profile, across noise levels  "
             "(tolerance-normalized, mean $\\pm$ std / 10 seeds; dashed = at-target; lower = better)",
             y=1.03, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("runs_controller/viz/fig_bar_profiles_4panel_linear.png", dpi=155, bbox_inches="tight")
print("saved runs_controller/viz/fig_bar_profiles_4panel_linear.png")
