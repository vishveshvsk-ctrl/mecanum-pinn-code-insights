#!/usr/bin/env python
"""Performance degradation under sensor noise: ASMC vs PID, per trajectory,
clean -> 1x -> 2x -> 5x, mean +/- std over 10 seeds."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("runs_controller/noise_eval_10seed.csv")

# panel spec: (trajectory, metric column, y-label, pretty title)
PANELS = [
    ("octagon",        "rms_w_mrad", "yaw $\\omega_{rms}$ (mrad/s)", "octagon (translation)"),
    ("spin_creep",     "rms_w_mrad", "yaw $\\omega_{rms}$ (mrad/s)", "spin_creep (pure yaw)"),
    ("coupled_vomega", "rms_w_mrad", "yaw $\\omega_{rms}$ (mrad/s)", "coupled_vomega (transl+yaw)"),
    ("spiral_orbit",   "rms_w_mrad", "yaw $\\omega_{rms}$ (mrad/s)", "spiral_orbit"),
    ("ellipse_tangent","max_pos_cm", "pos error max (cm)",           "ellipse_tangent (pose)"),
    ("ellipse_crab",   "max_pos_cm", "pos error max (cm)",           "ellipse_crab (pose, omnidir.)"),
]
SCALES = [0.0, 1.0, 2.0, 5.0]
XLAB   = ["clean", "1x", "2x", "5x"]
X      = np.arange(len(SCALES))
STYLE  = {"asmc": ("#1f77b4", "o", "ASMC"), "pid": ("#d62728", "s", "PID")}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

for ax, (traj, col, ylab, title) in zip(axes.ravel(), PANELS):
    for ctrl, (c, mk, lbl) in STYLE.items():
        means, stds = [], []
        for sc in SCALES:
            v = df[(df.controller == ctrl) & (df.trajectory == traj) & (df.scale == sc)][col].dropna()
            means.append(v.mean()); stds.append(v.std() if len(v) > 1 else 0.0)
        ax.errorbar(X, means, yerr=stds, color=c, marker=mk, ms=6, lw=2,
                    capsize=3, label=lbl, zorder=3)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylab); ax.set_xticks(X); ax.set_xticklabels(XLAB)
    ax.margins(x=0.08)

# shared legend + suptitle
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, 0.995))
fig.suptitle("Performance degradation under sensor noise  (mean $\\pm$ std, 10 seeds)",
             y=1.05, fontsize=13, fontweight="bold")
fig.text(0.5, -0.01, "noise scale  (1x = realistic indoor-AMR suite: gyro 3 mrad/s, odo 10 mm/s, fix 2 cm)",
         ha="center", fontsize=9, style="italic", color="#555")
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
fig.savefig("runs_controller/viz/fig_noise_degradation.png", dpi=160, bbox_inches="tight")
print("saved runs_controller/viz/fig_noise_degradation.png")
