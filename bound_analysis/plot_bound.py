"""
bound_analysis/plot_bound.py — figures from bound_results.npz / bound_summary.csv
(brief §7.5):

  Figure 1 — sigma_pos(t) bound curves for B0/B1/B2/B3 (default sensor grade)
             with the achieved ESKF error overlaid, one panel per trajectory,
             LOADED and UNLOADED groups in separate rows.
  Figure 2 — headline: achieved-to-bound ratio (B0, default grade) versus
             u_peak across all 10 trajectories.
  Figure 3 — sigma_bg(t), the gyro-bias bound (B0, default grade), overlaid
             across all 10 trajectories, colour-coded by group.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_bound import load_trace, TRACE_DIR, REPORT_DIR, SELECTION_PATH

VARIANTS = ("B0", "B1", "B2", "B3")
VARIANT_COLORS = {"B0": "#1b1b1b", "B1": "#4c72b0", "B2": "#dd8452", "B3": "#55a868"}
GROUP_COLORS = {"loaded": "#c44e52", "unloaded": "#4c72b0"}


def _load():
    summary = pd.read_csv(REPORT_DIR / "bound_summary.csv")
    results = np.load(REPORT_DIR / "bound_results.npz")
    return summary, results


def figure1(summary: pd.DataFrame, results, grade: str = "default"):
    loaded = sorted(summary[summary.group == "loaded"].combo_idx.unique())
    unloaded = sorted(summary[summary.group == "unloaded"].combo_idx.unique())
    ncols = max(len(loaded), len(unloaded))
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4), sharey=False)

    for row, (group_name, combo_list) in enumerate((("loaded", loaded), ("unloaded", unloaded))):
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(combo_list):
                ax.axis("off")
                continue
            combo_idx = combo_list[col]
            key0 = f"c{combo_idx:03d}_B0_{grade}"
            t = results[f"{key0}_t"]

            for variant in VARIANTS:
                key = f"c{combo_idx:03d}_{variant}_{grade}"
                if f"{key}_sigma_pos" not in results:
                    continue
                ax.plot(t, results[f"{key}_sigma_pos"], label=variant,
                        color=VARIANT_COLORS[variant], lw=1.3)

            trace = load_trace(TRACE_DIR / f"ellipse_c{combo_idx:03d}.arrow")
            achieved_err = np.hypot(trace["eskf_X"] - trace["X"], trace["eskf_Y"] - trace["Y"])
            ax.plot(trace["t"], achieved_err, label="achieved (ESKF)",
                    color="gray", lw=1.0, ls="--")

            ax.set_title(f"{group_name} c{combo_idx:03d}", fontsize=9)
            ax.set_xlabel("t [s]", fontsize=8)
            if col == 0:
                ax.set_ylabel("$\\sigma_{pos}$ / err [m]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle(f"PCRLB position-error bound vs. achieved ESKF ({grade} sensor grade)", y=0.99, fontsize=13)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, fontsize=9, frameon=False)
    out = REPORT_DIR / "figure1_sigma_pos_bounds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure2(summary: pd.DataFrame, grade: str = "default", variant: str = "B0"):
    sub = summary[(summary.variant == variant) & (summary.sensor_grade == grade)].copy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # Combos come in +/- heading-warp pairs sharing near-identical u_peak (the
    # ellipse builder's worbit sign flip): stagger duplicate-position labels
    # vertically so they don't render on top of each other.
    label_offsets = {}
    for group, color in GROUP_COLORS.items():
        g = sub[sub.group == group]
        ax.scatter(g.u_peak, g.ratio_achieved_to_bound, color=color, label=group, s=60, zorder=3)
        for _, r in g.iterrows():
            pos_key = (round(r.u_peak, 2), round(r.ratio_achieved_to_bound, 2))
            n_prior = label_offsets.get(pos_key, 0)
            label_offsets[pos_key] = n_prior + 1
            ax.annotate(f"c{int(r.combo_idx):03d}", (r.u_peak, r.ratio_achieved_to_bound),
                        fontsize=7, xytext=(4, 3 + 11 * n_prior), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="achieved == bound")
    ax.set_xlabel("friction-circle utilization $u_{peak}$")
    ax.set_ylabel(f"achieved / bound position RMS  ({variant}, {grade})")
    ax.set_title("Achieved-to-bound ratio vs. friction-circle utilization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = REPORT_DIR / "figure2_ratio_vs_upeak.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Explicit finding, whatever its sign (brief §10).
    if len(sub) >= 2:
        corr = np.corrcoef(sub.u_peak, sub.ratio_achieved_to_bound)[0, 1]
        slope = np.polyfit(sub.u_peak, sub.ratio_achieved_to_bound, 1)[0]
        print(f"Figure 2 finding: corr(u_peak, ratio)={corr:.3f}, slope={slope:.3f} "
              f"({'ratio grows with utilization -> slip-driven deficit supported' if slope > 0 else 'ratio flat/falling -> slip-driven-deficit hypothesis NOT supported'})")


def figure3(summary: pd.DataFrame, results, grade: str = "default", variant: str = "B0"):
    # NOTE: the per-trajectory sigma_bg(t) curves land almost exactly on top of
    # each other regardless of group (loaded/unloaded) -- gyro-bias
    # observability here is driven by the pose-fix cadence, not the specific
    # trajectory's loading. Different linestyles per group make this
    # near-total overlap visible rather than one colour silently masking the
    # other.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    combos = summary[["combo_idx", "group"]].drop_duplicates().sort_values("combo_idx")
    group_ls = {"loaded": "-", "unloaded": "--"}
    seen = set()
    for _, row in combos.iterrows():
        combo_idx, group = int(row.combo_idx), row.group
        key = f"c{combo_idx:03d}_{variant}_{grade}"
        if f"{key}_t" not in results:
            continue
        ax.plot(results[f"{key}_t"], results[f"{key}_sigma_bg"],
                color=GROUP_COLORS[group], alpha=0.8, lw=1.3, ls=group_ls[group],
                label=group if group not in seen else None)
        seen.add(group)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$\\sigma_{bg}$  [rad/s]  (gyro-bias bound)")
    ax.set_title(f"Gyro-bias observability under the 100 Hz docking pose fix ({variant}, {grade})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = REPORT_DIR / "figure3_sigma_bg.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    summary, results = _load()
    figure1(summary, results)
    figure2(summary)
    figure3(summary, results)


if __name__ == "__main__":
    main()
