"""Mecanum trajectory visualizer widget.

Flow
----
Dropdown selection → catalog lookup → ``mecanum_widget.data_extract.extract``
→ matplotlib payload → render in an ``ipywidgets.Output``/``Image`` widget.

Geometry
--------
The chassis footprint is a rectangle anchored at local corner ``(-h, -l)`` with
width ``2*h`` (body-x / heading axis) and height ``2*l`` (body-y).  A heading
pointer extends from the plant centre along body +x, i.e. direction
``(cos psi, sin psi)``.  This makes the long axis of the rectangle parallel to
the pointer.

Redraw strategy
---------------
The render core is free of ``ipywidgets``/``IPython`` so it can run headless
under Agg.  The interactive layer keeps one persistent ``Figure`` and pushes
pixels to an ``ipywidgets.Image`` after each frame by reading the canvas buffer.
This avoids a dependency on ``ipympl``.

Headless entry points
---------------------
``setup_axes`` and ``update_frame`` build the figure; ``export_gif`` produces an
animation file using only matplotlib.
"""

from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D

# Add code_insights/ to path so we can import mecanum_widget.
_CODE_INSIGHTS = Path(__file__).resolve().parent.parent
if str(_CODE_INSIGHTS) not in sys.path:
    sys.path.insert(0, str(_CODE_INSIGHTS))

from mecanum_widget.data_extract import extract  # noqa: E402


# -----------------------------------------------------------------------------
# Catalog helpers
# -----------------------------------------------------------------------------

_CATALOG_PATH_DEFAULT = Path(__file__).resolve().parent / "catalog.json"


def _catalog_path(path: Path | str | None) -> Path:
    if path is None:
        return _CATALOG_PATH_DEFAULT
    return Path(path)


def load_catalog(path: Path | str | None = None) -> dict[str, Any]:
    """Load catalog.json written by build_catalog.py."""
    import json

    p = _catalog_path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def profiles(catalog: dict[str, Any] | None = None) -> list[str]:
    cat = load_catalog() if catalog is None else catalog
    return list(cat.get("profiles", []))


def mus(catalog: dict[str, Any] | None = None) -> list[float]:
    cat = load_catalog() if catalog is None else catalog
    return [float(v) for v in cat.get("mus", [])]


def chis(catalog: dict[str, Any] | None = None) -> list[float]:
    cat = load_catalog() if catalog is None else catalog
    return [float(v) for v in cat.get("chis", [])]


def valid_combos(profile: str, catalog: dict[str, Any] | None = None) -> list[int]:
    cat = load_catalog() if catalog is None else catalog
    return list(cat.get("valid_combos_per_profile", {}).get(profile, []))


def _float_close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) < tol


def resolve_filename(
    profile: str,
    combo: int,
    mu: float,
    chi: float,
    catalog: dict[str, Any] | None = None,
) -> str | None:
    cat = load_catalog() if catalog is None else catalog
    for entry in cat.get("entries", []):
        if (
            entry["profile"] == profile
            and entry["combo"] == combo
            and _float_close(entry["mu"], mu)
            and _float_close(entry["chi"], chi)
        ):
            return entry["filename"]
    return None


def has_entry(
    profile: str,
    combo: int,
    mu: float,
    chi: float,
    catalog: dict[str, Any] | None = None,
) -> bool:
    return resolve_filename(profile, combo, mu, chi, catalog) is not None


# -----------------------------------------------------------------------------
# Headless render core
# -----------------------------------------------------------------------------

_ARTISTS = dict[str, Any]


def setup_axes(ax: plt.Axes, payload: dict[str, Any]) -> _ARTISTS:
    """Prepare axes for animation; return per-frame artists."""
    ax.clear()

    ref = payload["reference"]
    act = payload["actual"]
    geom = payload["geometry"]
    h = geom["h"]
    l = geom["l"]
    margin = 2.0 * h

    # Dotted black reference path.
    ax.plot(
        ref["x"],
        ref["y"],
        linestyle=":",
        color="k",
        linewidth=1.5,
        label="reference",
    )

    # Faint full actual path for context.
    ax.plot(
        act["x"],
        act["y"],
        color="lightgray",
        linewidth=0.8,
        zorder=0,
    )

    # Traced actual path (updated per frame).
    (trace,) = ax.plot([], [], color="tab:blue", linewidth=1.8, label="actual")

    # Chassis rectangle anchored at (-h, -l) in body frame.
    rect = patches.Rectangle(
        (-h, -l),
        2.0 * h,
        2.0 * l,
        linewidth=2,
        edgecolor="tab:blue",
        facecolor="cornflowerblue",
        alpha=0.5,
        zorder=3,
    )
    ax.add_patch(rect)

    # Heading pointer.
    (pointer,) = ax.plot([], [], color="tab:red", linewidth=2.2, zorder=4)

    # Current position dot.
    (dot,) = ax.plot([], [], "o", color="tab:blue", markersize=5, zorder=5)

    # Time annotation.
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Equal aspect + limits + labels.
    all_x = np.asarray(ref["x"]).tolist() + np.asarray(act["x"]).tolist()
    all_y = np.asarray(ref["y"]).tolist() + np.asarray(act["y"]).tolist()
    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        dx = max(xmax - xmin, 0.1)
        dy = max(ymax - ymin, 0.1)
        ax.set_xlim(xmin - margin - 0.1 * dx, xmax + margin + 0.1 * dx)
        ax.set_ylim(ymin - margin - 0.1 * dy, ymax + margin + 0.1 * dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{payload['meta']['profile']} c{payload['meta']['combo']} "
        f"mu={payload['meta']['mu']} chi={payload['meta']['chi']}"
    )
    ax.grid(True)
    ax.legend(loc="upper right")

    return {
        "trace": trace,
        "rect": rect,
        "pointer": pointer,
        "dot": dot,
        "text": text,
        "ax": ax,
    }


def update_frame(artists: _ARTISTS, payload: dict[str, Any], idx: int) -> list[Any]:
    """Update artists to frame ``idx``; return changed artists."""
    act = payload["actual"]
    geom = payload["geometry"]
    h = geom["h"]
    time_arr = payload["time"]

    idx = max(0, min(idx, len(time_arr) - 1))
    x = float(act["x"][idx])
    y = float(act["y"][idx])
    psi = float(act["psi"][idx])

    # Traced path up to current index.
    artists["trace"].set_data(act["x"][: idx + 1], act["y"][: idx + 1])

    # Rectangle transform: rotate about body origin, then translate to (x, y).
    transform = (
        Affine2D().rotate(psi).translate(x, y) + artists["ax"].transData
    )
    artists["rect"].set_transform(transform)

    # Heading pointer from centre along body +x.
    pointer_len = 1.3 * h
    px = x + pointer_len * math.cos(psi)
    py = y + pointer_len * math.sin(psi)
    artists["pointer"].set_data([x, px], [y, py])

    # Current position dot.
    artists["dot"].set_data([x], [y])

    # Time text.
    artists["text"].set_text(f"t = {float(time_arr[idx]):.2f}s")

    return [
        artists["trace"],
        artists["rect"],
        artists["pointer"],
        artists["dot"],
        artists["text"],
    ]


# -----------------------------------------------------------------------------
# GIF export (headless reusable)
# -----------------------------------------------------------------------------


def export_gif(
    payload: dict[str, Any],
    out_path: str | Path,
    fps: int = 15,
    max_frames: int = 200,
) -> Path:
    """Export a playback GIF from ``payload`` to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(payload["time"])
    if n <= 1:
        raise ValueError("Need at least two frames to export")

    # Cap frames while preserving start and end.
    if n > max_frames:
        stride = max(1, math.ceil((n - 1) / (max_frames - 1)))
        frame_indices = list(range(0, n, stride))
        if frame_indices[-1] != n - 1:
            frame_indices.append(n - 1)
    else:
        frame_indices = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 7))
    artists = setup_axes(ax, payload)

    def init() -> list[Any]:
        return update_frame(artists, payload, 0)

    def update(i: int) -> list[Any]:
        return update_frame(artists, payload, frame_indices[i])

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )

    try:
        writer = PillowWriter(fps=fps)
        anim.save(str(out_path), writer=writer)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(f"Failed to write GIF with PillowWriter: {exc}")

    plt.close(fig)
    return out_path


def render_frame_png(
    payload: dict[str, Any],
    idx: int | None = None,
    dpi: int = 100,
) -> bytes:
    """Render a single frame to PNG bytes (headless)."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    artists = setup_axes(ax, payload)
    if idx is None:
        idx = len(payload["time"]) // 2
    update_frame(artists, payload, idx)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Interactive widget
# -----------------------------------------------------------------------------


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_widget(
    catalog_path: str | Path | None = None,
    max_points: int = 1500,
    data_dir: str | Path | None = None,
):
    """Build and return the interactive ipywidgets widget."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    if data_dir is not None:
        os.environ["MECANUM_DATA_DIR"] = str(Path(data_dir).expanduser().resolve())

    catalog = load_catalog(catalog_path)

    profile_options = profiles(catalog)
    mu_options = mus(catalog)
    chi_options = chis(catalog)

    profile_dd = widgets.Dropdown(options=profile_options, description="profile")
    mu_dd = widgets.Dropdown(
        options=[(f"{v:g}", v) for v in mu_options], description="mu"
    )
    chi_dd = widgets.Dropdown(
        options=[(f"{v:.3f}", v) for v in chi_options], description="chi"
    )
    combo_dd = widgets.Dropdown(options=[], description="combo")

    def refresh_combo_options(*_):
        combo_dd.options = valid_combos(profile_dd.value, catalog)
        if combo_dd.options:
            combo_dd.value = combo_dd.options[0]

    profile_dd.observe(refresh_combo_options, names="value")
    refresh_combo_options()

    slider = widgets.IntSlider(min=0, max=0, value=0, description="frame")
    play = widgets.Play(
        value=0,
        min=0,
        max=0,
        step=1,
        interval=60,
        description="play",
    )
    widgets.jslink((play, "value"), (slider, "value"))

    export_btn = widgets.Button(description="Export GIF")
    status_lbl = widgets.Label(value="Ready")
    img_widget = widgets.Image(value=b"", format="png")

    fig: plt.Figure | None = None
    ax: plt.Axes | None = None
    artists: _ARTISTS | None = None
    payload: dict[str, Any] | None = None

    def redraw():
        nonlocal fig, ax, artists, payload
        if payload is None or fig is None or ax is None or artists is None:
            return
        update_frame(artists, payload, slider.value)
        fig.canvas.draw()
        img_widget.value = _fig_to_png_bytes(fig)

    def on_selection_change(change=None):
        nonlocal fig, ax, artists, payload
        p = profile_dd.value
        c = combo_dd.value
        m = mu_dd.value
        ch = chi_dd.value
        if c is None or not has_entry(p, c, m, ch, catalog):
            status_lbl.value = f"No data for {p} c{c} mu={m} chi={ch}"
            return
        try:
            payload = extract(p, c, m, ch, max_points=max_points)
        except Exception as exc:
            status_lbl.value = f"Extract failed: {exc}"
            return

        matplotlib.use("Agg")
        if fig is not None:
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 7))
        artists = setup_axes(ax, payload)
        slider.max = len(payload["time"]) - 1
        play.max = len(payload["time"]) - 1
        slider.value = 0
        play.value = 0
        update_frame(artists, payload, 0)
        fig.canvas.draw()
        img_widget.value = _fig_to_png_bytes(fig)
        status_lbl.value = f"Loaded {payload['meta']['filename']} ({payload['meta']['n_points']} pts)"

    def on_slider_change(change):
        redraw()

    def on_export(_):
        if payload is None:
            status_lbl.value = "Nothing to export"
            return
        out_dir = Path(__file__).resolve().parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{payload['meta']['profile']}_c{payload['meta']['combo']:03d}_mu_{payload['meta']['mu']:g}_chi_{payload['meta']['chi']:.3f}.gif"
        try:
            export_gif(payload, out_path)
            status_lbl.value = f"Saved {out_path}"
        except Exception as exc:
            status_lbl.value = f"Export failed: {exc}"

    profile_dd.observe(on_selection_change, names="value")
    combo_dd.observe(on_selection_change, names="value")
    mu_dd.observe(on_selection_change, names="value")
    chi_dd.observe(on_selection_change, names="value")
    slider.observe(on_slider_change, names="value")
    export_btn.on_click(on_export)

    controls = widgets.HBox([profile_dd, mu_dd, chi_dd, combo_dd])
    playback = widgets.HBox([play, slider])
    export_row = widgets.HBox([export_btn, status_lbl])
    ui = widgets.VBox([controls, playback, img_widget, export_row])

    on_selection_change()
    return ui


if __name__ == "__main__":
    # Headless smoke test: render a mid-frame to PNG.
    import argparse

    parser = argparse.ArgumentParser(description="Headless widget smoke test")
    parser.add_argument("--profile", default="coupled_vomega")
    parser.add_argument("--combo", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--chi", type=float, default=0.005)
    parser.add_argument("--out", default="trajectory_widget/_tmp/smoke_frame.png")
    args = parser.parse_args()

    matplotlib.use("Agg")
    payload = extract(args.profile, args.combo, args.mu, args.chi)
    png = render_frame_png(payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(png)
    print(f"Wrote {out} ({len(png)} bytes)")
