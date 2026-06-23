#!/usr/bin/env python
# =============================================================================
# make_pipeline_flowchart_state_recon.py - static information-flow diagram for
# the Approach-2 causal state-reconstruction observer (data -> training ->
# results). SSM is the deployed model; GRU is a dashed CONTROL branch (cross-
# architecture check that per-state error floors are intrinsic, not SSM-capacity
# artifacts). Pure matplotlib (no widgets); writes PNG + SVG to images_and_plots/.
# Unicode glyphs (DejaVu Sans), not mathtext.
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_SRC   = ("#DCE6F2", "#2E5984")
C_PIPE  = ("#D6ECEC", "#2A7B7B")
C_IN    = ("#DDEFD6", "#4A7A3A")
C_TGT   = ("#FCEBD2", "#B5792A")
C_MODEL = ("#E4DCF0", "#5B3F8C")
C_CTRL  = ("#EFEAF6", "#8E79B4")   # control branch (lighter purple)
C_EVAL  = ("#E8E8EC", "#555555")
C_RES   = ("#2E4A6B", "#1B2D44")
C_HAND  = ("#F5D9D9", "#9C3B3B")

fig, ax = plt.subplots(figsize=(12.5, 16.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc_ec, fs=9.5, tc="black", weight="normal", ls="-"):
    fc, ec = fc_ec
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.4,rounding_size=1.6",
                 facecolor=fc, edgecolor=ec, linewidth=1.6, linestyle=ls, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, weight=weight, linespacing=1.35)


def arrow(x1, y1, x2, y2, color="#33373D", lw=1.7, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))


ax.text(50, 98.3, "Approach 2 - Causal State-Reconstruction Observer:  Information Flow",
        ha="center", va="center", fontsize=14.5, weight="bold", color="#1B2D44")
ax.text(50, 95.7, "Julia sim data  →  measurable-only training  →  "
        "per-state observability results",
        ha="center", va="center", fontsize=10.5, color="#444", style="italic")

box(50, 90, 60, 6.2,
    "Julia high-fidelity sim  →  Arrow files\n"
    "one run per (profile × combo × μ × χ)   ·   "
    "measurables + hidden-state labels", C_SRC, fs=9.5)
box(89, 90, 19, 5.2, "diagnostics_combined.csv\n(whitelist)", C_SRC, fs=8.2)
arrow(89, 87.4, 70, 84.6)

box(50, 82, 60, 6.0,
    "Discover & gate   (data.py)\n"
    "μ ∈ {0.3, 0.5, 0.8}   ·   χ ∈ {0, 0.002, 0.005, 0.008}   ·   "
    "whitelist (combined_reco ≠ reject)", C_PIPE, fs=9.0)
arrow(50, 86.9, 50, 85.1); arrow(50, 78.9, 50, 77.1)

box(50, 74, 60, 5.4,
    "Grouped split by (profile, combo)\n"
    "train / val / test  -  no trajectory (or χ-sibling) leakage", C_PIPE, fs=9.0)
arrow(50, 71.2, 50, 69.4)

box(50, 66.5, 60, 5.2,
    "Per-file stream (read → build → drop)\n"
    "decimate  2000 Hz → 500 Hz  (stride-4)", C_PIPE, fs=9.0)
arrow(43, 63.9, 30, 60.4); arrow(57, 63.9, 70, 60.4)

box(28, 56, 40, 8.6,
    "INPUTS  (measurable-only)   features.py\n"
    "globals:  Vx, Vy, ψ̇\n"
    "per wheel:  Msat, ω, sin θ̃, cos θ̃\n"
    "(θ̃ = sawtooth_tanh ; independent set)",
    C_IN, fs=8.6)
box(72, 56, 40, 8.6,
    "TARGETS  (hidden states, Arrow cols)\n"
    "γ  (roller rate, [13:16])\n"
    "zx, zy  (linear bristle)\n"
    "× 4 wheels  ·  NEVER inputs  (zs/Mz dropped)", C_TGT, fs=8.4)
arrow(28, 51.7, 40, 48.0); arrow(72, 51.7, 60, 48.0)

box(50, 45, 66, 5.4,
    "Normalize (streaming train-only stats)  +  causal windows  W ∈ {8, 32}\n"
    "window  [t-W+1 .. t]  →  hidden state at  t   (past-only filter)",
    C_PIPE, fs=9.0)
arrow(50, 42.3, 50, 40.6)

box(50, 38, 56, 4.8,
    "Wheel-shared encoder  +  wheel embedding (zero/frozen)   (models.py)", C_MODEL, fs=9.0)
# branch: SSM (deployed, solid) and GRU (control, dashed)
arrow(43, 35.6, 33, 32.9)
arrow(57, 35.6, 67, 32.9, color="#8E79B4", ls="--")
box(33, 30.3, 30, 5.4, "SSM encoder  (deployed)\nMamba-lite selective scan · SiLU",
    C_MODEL, fs=8.4)
box(67, 30.3, 30, 5.4, "GRU baseline  —  CONTROL BRANCH\ncross-architecture floor check",
    C_CTRL, fs=8.3, ls="--")
arrow(33, 27.6, 43, 25.6)
arrow(67, 27.6, 57, 25.6, color="#8E79B4", ls="--")

box(50, 23, 66, 5.6,
    "12 heads  →  {γ, zx, zy} × 4   ·   "
    "per-state normalized MSE  (losses.py)\n"
    "train loop (training.py):  GPU bf16 autocast · AdamW · streaming · resumable",
    C_MODEL, fs=9.0)
arrow(50, 20.2, 50, 18.4)

box(50, 15.3, 66, 5.6,
    "Evaluation: val (same-subset) + test (cross-subset)   (evaluation.py)\n"
    "per-state / -wheel / -bin RMSE  ·  derive  "
    "ω_z = ψ̇ + γ̂ · sin θ̃ · cos δ  ·  bin by |Vp|, |ω_z|", C_EVAL, fs=8.6)
arrow(34, 12.5, 20, 9.7); arrow(50, 12.5, 50, 9.7); arrow(66, 12.5, 80, 9.7)

box(20, 7.0, 30, 4.6, "state_observability.csv\n(per-state RMSE table)", C_RES, fs=8.2, tc="white")
box(50, 7.0, 34, 4.6, "figures:  same-vs-cross gap\n(SSM vs GRU control) · ω_z, z bins",
    C_RES, fs=7.8, tc="white")
box(80, 7.0, 30, 4.6, "observability ranking\n{γ,  ω_z,  z}", C_RES, fs=8.2, tc="white")
arrow(80, 4.7, 55, 2.6); arrow(50, 4.7, 50, 2.6)

box(50, 1.4, 62, 3.2,
    "→  hand-back to Approach 1:   ω_z fidelity bounds the "
    "χ-channel   (c_t = (8/3π)·|ω_z|·χ)", C_HAND, fs=8.8)

leg = [("Data source", C_SRC), ("Data pipeline", C_PIPE),
       ("Measurable inputs", C_IN), ("Hidden-state targets", C_TGT),
       ("Model / training", C_MODEL), ("Control branch (GRU)", C_CTRL),
       ("Evaluation", C_EVAL), ("Results / hand-back", C_RES)]
for i, (lab, (fc, ec)) in enumerate(leg):
    yy = 90 - i * 2.4
    ls = "--" if "Control" in lab else "-"
    ax.add_patch(FancyBboxPatch((1.0, yy - 0.8), 2.0, 1.6,
                 boxstyle="round,pad=0.1,rounding_size=0.4",
                 facecolor=fc, edgecolor=ec, linewidth=1.0, linestyle=ls))
    ax.text(3.5, yy, lab, ha="left", va="center", fontsize=7.4, color="#333")
ax.text(1.0, 90 - len(leg) * 2.4 - 0.6,
        "dashed = control branch (not deployed)", ha="left", va="center",
        fontsize=6.8, color="#666", style="italic")

out = Path(__file__).resolve().parents[1] / "images_and_plots"
out.mkdir(exist_ok=True)
fig.savefig(out / "observer_pipeline_flowchart.png", dpi=200, bbox_inches="tight")
fig.savefig(out / "observer_pipeline_flowchart.svg", bbox_inches="tight")
print("wrote", out / "observer_pipeline_flowchart.png")
print("wrote", out / "observer_pipeline_flowchart.svg")
