"""Reusable generator for the Mamba ForceRecon PINN information-flow flowchart.

Static matplotlib figure (no interactive widgets). Saves PNG + SVG to
images_and_plots/. Run with the claude-venv python:

    python make_forcerecon_flowchart.py

Bands (top -> bottom): Data -> Forward path -> Inverse path -> Training/Losses
-> Results. Band labels are HORIZONTAL in a left gutter. All connectors are
strictly orthogonal (vertical/horizontal segments only); where two connectors
cross, the secondary one hops over with a small circular bump.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent / "images_and_plots"

# ---- palette (flat, <=2 colored ramps + neutrals) ----
C_DATA     = "#5B6B7F"
C_FORWARD  = "#2D5BA0"
C_INVERSE  = "#1F7A6B"
C_TRAIN    = "#C0822A"
C_RESULT   = "#1B2A4A"
C_ARROW    = "#44505F"
C_FEED     = "#B23A48"
BAND_BG    = {"data": "#EDF0F4", "fwd": "#EAF0F8", "inv": "#E7F3F0",
              "train": "#FAF2E2", "res": "#EAECF2"}
TXT_DARK   = "#1F2A37"


def _box(ax, cx, cy, w, h, title, detail=None, fc=C_FORWARD, tc="white",
         tfs=10.5, dfs=8.3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 fc=fc, ec=fc, lw=1.3, zorder=3))
    if detail:
        ax.text(cx, cy + h * 0.21, title, ha="center", va="center", color=tc,
                fontsize=tfs, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.17, detail, ha="center", va="center", color=tc,
                fontsize=dfs, zorder=4, linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha="center", va="center", color=tc,
                fontsize=tfs, fontweight="bold", zorder=4)
    return dict(cx=cx, cy=cy, w=w, h=h)


def _b(p):  # bottom / top / left / right anchor points
    return (p["cx"], p["cy"] - p["h"] / 2)
def _t(p):
    return (p["cx"], p["cy"] + p["h"] / 2)
def _l(p):
    return (p["cx"] - p["w"] / 2, p["cy"])
def _r(p):
    return (p["cx"] + p["w"] / 2, p["cy"])


def _arrow(ax, p0, p1, color=C_ARROW, lw=1.6, dashed=False, label=None,
           lfs=8.0, lcol=None):
    """Straight (already-orthogonal) connector with an arrowhead."""
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle="--" if dashed else "-",
                                shrinkA=3, shrinkB=3))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=lfs,
                color=lcol or color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


def _bump(cx, cy, r, sgn, axis):
    """Semicircular hop points; horizontal seg bumps +y, vertical seg bumps +x."""
    phi = np.linspace(0, np.pi, 14)
    if axis == "h":
        xs = cx - sgn * r * np.cos(phi)
        ys = cy + r * np.sin(phi)
    else:
        xs = cx + r * np.sin(phi)
        ys = cy - sgn * r * np.cos(phi)
    return list(zip(xs, ys))


def _on_seg(a, b, h):
    if abs(a[1] - b[1]) < 1e-6:                       # horizontal
        return abs(h[1] - a[1]) < 0.06 and min(a[0], b[0]) < h[0] < max(a[0], b[0])
    return abs(h[0] - a[0]) < 0.06 and min(a[1], b[1]) < h[1] < max(a[1], b[1])


def _route(ax, wpts, color=C_ARROW, lw=1.6, dashed=False, hops=(), r=0.14,
           label=None, lpos=None, lfs=8.0, lcol=None):
    """Orthogonal polyline through wpts (right-angle bends only), with a small
    circular hop over every point in `hops` that lies on a segment."""
    pts = [wpts[0]]
    for a, b in zip(wpts, wpts[1:]):
        seg_hops = [h for h in hops if _on_seg(a, b, h)]
        if abs(a[1] - b[1]) < 1e-6:                   # horizontal segment
            sgn = 1 if b[0] >= a[0] else -1
            for hx, hy in sorted(seg_hops, key=lambda h: sgn * h[0]):
                pts.append((hx - sgn * r, a[1]))
                pts += _bump(hx, a[1], r, sgn, "h")
                pts.append((hx + sgn * r, a[1]))
        else:                                          # vertical segment
            sgn = 1 if b[1] >= a[1] else -1
            for hx, hy in sorted(seg_hops, key=lambda h: sgn * h[1]):
                pts.append((a[0], hy - sgn * r))
                pts += _bump(a[0], hy, r, sgn, "v")
                pts.append((a[0], hy + sgn * r))
        pts.append(b)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls="--" if dashed else "-", zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))
    if label:
        lx, ly = lpos if lpos else ((wpts[0][0] + wpts[-1][0]) / 2,
                                    (wpts[0][1] + wpts[-1][1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=lfs,
                color=lcol or color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


def _band(ax, y0, y1, name, key, qual=None):
    ax.add_patch(Rectangle((-2.2, y0), 18.0, y1 - y0, fc=BAND_BG[key], ec="none",
                           alpha=0.7, zorder=0))
    cy = (y0 + y1) / 2
    ax.text(-1.15, cy + (0.16 if qual else 0), name, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#3A4757", zorder=1)
    if qual:
        ax.text(-1.15, cy - 0.24, qual, ha="center", va="center",
                fontsize=6.6, color="#5A6573", zorder=1)


def build():
    fig, ax = plt.subplots(figsize=(16.6, 11.6))
    ax.set_xlim(-2.4, 16); ax.set_ylim(0, 11.7); ax.axis("off")

    ax.text(8, 11.42, "Forward–Inverse Force-Reconstruction PINN — Information Flow",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=TXT_DARK)
    ax.text(8, 11.08, "Mecanum_PINN_Mamba_ForceRecon_v1   (measurable inputs only · "
            "roller-frame Fpar/Fperp · Heun O(dt³) integrator)",
            ha="center", va="center", fontsize=9.5, color="#5A6573")

    # ---- bands (horizontal labels in the left gutter) ----
    _band(ax, 9.55, 10.85, "DATA", "data")
    _band(ax, 7.35, 8.75, "FORWARD", "fwd", "causal in time")
    _band(ax, 5.30, 6.70, "INVERSE", "inv", "causal in Δstate")
    _band(ax, 2.95, 4.65, "TRAINING", "train", "/ losses")
    _band(ax, 0.55, 2.25, "RESULTS", "res", "/ presentation")

    # ---- DATA ----
    yD = 10.2
    a1 = _box(ax, 3.2, yD, 4.0, 1.05, "Julia plant simulation",
              "LuGre+Adamov · ASMC+DOB\n→ Arrow @ 2000 Hz", fc=C_DATA)
    a2 = _box(ax, 8.0, yD, 5.4, 1.05, "Arrow dataset",
              "measurable: V, yaw-rate, ω, θ, Msat\nlabels: Fpar, Fperp · probes: ω_z, util",
              fc=C_DATA)
    a3 = _box(ax, 13.2, yD, 4.0, 1.05, "Loader",
              "whitelist 1568 · 500 Hz\nnormalize · sliding windows", fc=C_DATA)
    _arrow(ax, _r(a1), _l(a2)); _arrow(ax, _r(a2), _l(a3))

    hub = _box(ax, 8.0, 9.18, 5.0, 0.42, "measurable inputs  (sensor-only)", None,
               fc="#3A4757", tfs=9.5)
    _arrow(ax, _b(a2), _t(hub))

    # ---- FORWARD ----
    yF = 8.05
    b1 = _box(ax, 2.95, yF, 3.0, 1.05, "Selective SSM encoder",
              "reconstruct hidden\nγ, z, ω_z", fc=C_FORWARD)
    b2 = _box(ax, 6.45, yF, 3.6, 1.05, "Structured force head",
              "[μ(A + Cχ) + B](slip)\n+ D (bristle)", fc=C_FORWARD)
    b3 = _box(ax, 9.65, yF, 2.4, 1.05, "Forces @k", "F_par, F_perp", fc=C_FORWARD)
    b4 = _box(ax, 12.4, yF, 2.8, 1.05, "Analytical NE integrator",
              "Heun · O(dt³)", fc=C_FORWARD)
    b5 = _box(ax, 14.85, yF, 1.9, 1.05, "State @k+1", "V, ω, θ", fc=C_FORWARD)
    for x, y in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        _arrow(ax, _r(x), _l(y))
    # measurable inputs -> SSM encoder (orthogonal: down · left · down)
    _route(ax, [(8.0, 8.97), (8.0, 8.80), (2.95, 8.80), (2.95, 8.575)],
           label="inputs @k", lpos=(5.4, 8.80), lfs=7.6)

    # ---- INVERSE ----
    yI = 6.0
    c1 = _box(ax, 3.2, yI, 3.4, 1.05, "Δ-state window", "inputs @ {k-2 … k}", fc=C_INVERSE)
    c2 = _box(ax, 7.2, yI, 3.2, 1.05, "Inverse net", "causal in Δstate", fc=C_INVERSE)
    c3 = _box(ax, 10.7, yI, 2.6, 1.05, "Forces @k", "F_inv  (μ-agnostic)", fc=C_INVERSE)
    c4 = _box(ax, 13.85, yI, 3.0, 1.05, "(μ, χ) estimate",
              "linear readout\nχ slip-spin-gated", fc=C_INVERSE)
    for x, y in ((c1, c2), (c2, c3), (c3, c4)):
        _arrow(ax, _r(x), _l(y))
    # inverse consumes the SAME measurable stream as forward -> straight vertical feed
    _route(ax, [(3.0, 7.525), (3.0, 6.525)],
           label="same inputs · Δ-window", lpos=(3.0, 7.02), lfs=7.0)

    # closed loop: (μ,χ) estimate -> forward force head, routed in the fwd/inv gap.
    # hops over the F_fwd down-feed at (9.1, 7.0).
    _route(ax, [(13.85, 6.525), (13.85, 7.0), (6.45, 7.0), (6.45, 7.525)],
           color=C_FEED, dashed=True, hops=[(9.1, 7.0)],
           label="(μ, χ) → force head   (closed loop)", lpos=(11.7, 7.0),
           lcol=C_FEED, lfs=7.6)

    # ---- TRAINING ----
    yT = 3.78
    d1 = _box(ax, 3.0, yT, 3.2, 1.35, "Grounding", "force labels\n(warm-up only)", fc=C_TRAIN)
    d2 = _box(ax, 6.7, yT, 3.2, 1.35, "Physics", "discrete NE residual\n(measurable-only)", fc=C_TRAIN)
    d3 = _box(ax, 10.2, yT, 3.2, 1.35, "Consistency (monitor)",
              "F_fwd ↔ F_inv\nplotted · NOT trained", fc=C_DATA)
    d4 = _box(ax, 13.7, yT, 3.2, 1.35, "Parameter ID", "(μ, χ) vs labels\nslip-spin-gated χ", fc=C_TRAIN)
    ax.text(6.3, 2.60, "curriculum:  force warm-up  →  physics-only final phase",
            ha="center", va="center", fontsize=8.6, style="italic", color="#7A5A1E", zorder=5)

    # forward / inverse -> training (orthogonal; F_fwd threads the c2|c3 channel
    # and hops over the inverse-row arrow at (9.1, 6.0))
    _route(ax, [(9.65, 7.525), (9.65, 7.2), (9.1, 7.2), (9.1, 4.66)],
           hops=[(9.1, 6.0)], label="F_fwd", lpos=(8.72, 5.25), lfs=7.8)
    _arrow(ax, _b(c3), (10.7, 4.66), label="F_inv")
    _route(ax, [(13.78, 5.475), (13.78, 4.455)], label="(μ, χ)", lpos=(13.78, 4.95), lfs=7.8)

    # ---- RESULTS ----
    yR = 1.42
    e1 = _box(ax, 3.0, yR, 3.3, 1.4, "Force reconstruction",
              "Fpar/Fperp error\nmulti-step rollout", fc=C_RESULT)
    e2 = _box(ax, 6.8, yR, 3.4, 1.4, "μ, χ identification",
              "vs grid 0.3/0.5/0.8 × χ\nconfidence gating", fc=C_RESULT)
    e3 = _box(ax, 10.4, yR, 3.4, 1.4, "Closed-loop twin",
              "(μ,χ) → forward\nsudden-change adapt", fc=C_RESULT)
    e4 = _box(ax, 13.9, yR, 3.0, 1.4, "Figures + manifest",
              "loss history\nOOD / test", fc=C_RESULT)
    _arrow(ax, (12.0, 3.10), (12.0, 2.26), label="trained model")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "mamba_forcerecon_flow.png"
    svg = OUT_DIR / "mamba_forcerecon_flow.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[flowchart] wrote {png}")
    print(f"[flowchart] wrote {svg}")


if __name__ == "__main__":
    build()
