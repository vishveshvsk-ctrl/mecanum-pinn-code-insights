"""Extract a JSON-serializable payload from one Mecanum simulation Arrow file.

Reference reconstruction
------------------------
VelRef profiles (everything except ellipse) do NOT store a world-frame
reference path.  We reconstruct it by integrating the desired body-frame
velocity through the desired heading on the full-resolution time grid:

    Xref += (Vx_des*cos(psi_des) - Vy_des*sin(psi_des)) * dt
    Yref += (Vx_des*sin(psi_des) + Vy_des*cos(psi_des)) * dt

starting from the actual initial position (Xo[0], Yo[0]).

For the ellipse profile (PosRef), xo_des/yo_des are present and are used
directly as the reference XY.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.feather as feather
import tomllib


def resolve_repo_root(start: Path | None = None) -> Path:
    """Walk up until we find a directory that contains both code_insights/ and data/."""
    if start is None:
        start = Path(__file__).resolve().parent
    here = start
    for _ in range(8):
        if (here / "code_insights").is_dir() and (here / "data").is_dir():
            return here
        parent = here.parent
        if parent == here:
            break
        here = parent
    raise FileNotFoundError(
        f"Could not locate repo root containing code_insights/ and data/ starting from {start}"
    )


def resolve_data_dir() -> Path:
    """Return the authoritative simulation-data directory.

    Order of precedence:
      1. MECANUM_DATA_DIR environment variable (absolute path).
      2. Repo-root sibling data/Simulation_Data_MecanumSlipSpin_LugreAdamov/.

    PROJECT_LAYOUT.md authority: repo root is the directory containing both
    ``code_insights/`` and ``data/``.  On Windows this is typically
    ``C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\``.
    """
    env = os.environ.get("MECANUM_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        # Do not hard-fail; the env var may point to a stale per-machine path.
        # Fall through to repo-root auto-resolution.

    repo_root = resolve_repo_root()
    data_dir = repo_root / "data" / "Simulation_Data_MecanumSlipSpin_LugreAdamov"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found at {data_dir}. Set MECANUM_DATA_DIR to override."
        )
    return data_dir


def build_filename(profile: str, combo: int, mu: float, chi: float) -> str:
    """Return the basename obeying the project filename contract."""
    return (
        f"{profile}_c{combo:03d}_mu_{mu:g}_case1_lugre_adamov_chi_{chi:.3f}.arrow"
    )


def load_geometry() -> dict[str, float]:
    """Read platform geometry from base.toml with documented safe fallback."""
    defaults = {"l": 0.15, "h": 0.235, "R": 0.05}
    try:
        repo_root = resolve_repo_root()
        base_toml = repo_root / "code_insights" / "trajectory_files_run_0p3_main" / "base.toml"
        if not base_toml.is_file():
            base_toml = repo_root / "trajectory_files_run_0p3_main" / "base.toml"
        if not base_toml.is_file():
            return defaults
        with open(base_toml, "rb") as f:
            cfg = tomllib.load(f)
        geom = cfg.get("platform", {}).get("geometry", {})
        return {
            "l": float(geom.get("l", defaults["l"])),
            "h": float(geom.get("h", defaults["h"])),
            "R": float(geom.get("R", defaults["R"])),
        }
    except Exception:
        return defaults


def _read_column(table: Any, name: str) -> np.ndarray:
    return table[name].to_numpy()


def _decimate_stride(n_rows: int, max_points: int) -> int:
    if n_rows <= max_points:
        return 1
    return max(1, math.ceil(n_rows / max_points))


def _decimation_indices(n_rows: int, stride: int) -> np.ndarray:
    """Return index set that subsamples every ``stride`` rows and always includes the final row."""
    idx = np.arange(0, n_rows, stride)
    if n_rows > 0 and idx[-1] != n_rows - 1:
        idx = np.concatenate([idx, np.array([n_rows - 1])])
    return idx


def _decimate(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride == 1:
        return arr
    idx = _decimation_indices(len(arr), stride)
    return arr[idx].copy()


def _reconstruct_reference(
    time: np.ndarray,
    vx_des: np.ndarray,
    vy_des: np.ndarray,
    psi_des: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate desired body velocity into the world frame (full resolution)."""
    n = len(time)
    xref = np.empty(n)
    yref = np.empty(n)
    xref[0] = x0
    yref[0] = y0
    for i in range(1, n):
        dt = time[i] - time[i - 1]
        c = math.cos(psi_des[i - 1])
        s = math.sin(psi_des[i - 1])
        xref[i] = xref[i - 1] + (vx_des[i - 1] * c - vy_des[i - 1] * s) * dt
        yref[i] = yref[i - 1] + (vx_des[i - 1] * s + vy_des[i - 1] * c) * dt
    return xref, yref


def extract(
    profile: str,
    combo: int,
    mu: float,
    chi: float,
    max_points: int = 1500,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable payload for the widget."""
    if data_dir is None:
        data_dir = resolve_data_dir()

    filename = build_filename(profile, combo, mu, chi)
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Arrow file not found: {path}")

    table = feather.read_table(path)
    columns = table.column_names

    # Columns we always need for the actual pose.
    required = ["time", "Xo", "Yo", "psi"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {filename}: {missing}")

    time = _read_column(table, "time")
    xo = _read_column(table, "Xo")
    yo = _read_column(table, "Yo")
    psi = _read_column(table, "psi")

    n_rows = len(time)
    stride = _decimate_stride(n_rows, max_points)

    if "xo_des" in columns and "yo_des" in columns:
        ref_source = "posref"
        xo_des = _read_column(table, "xo_des")
        yo_des = _read_column(table, "yo_des")
        xref_full, yref_full = xo_des.copy(), yo_des.copy()
    else:
        ref_source = "integrated"
        for col in ("Vx_des", "Vy_des", "psi_des"):
            if col not in columns:
                raise ValueError(
                    f"Cannot integrate reference for {filename}; "
                    f"missing {col} (and no xo_des/yo_des)"
                )
        vx_des = _read_column(table, "Vx_des")
        vy_des = _read_column(table, "Vy_des")
        psi_des = _read_column(table, "psi_des")
        xref_full, yref_full = _reconstruct_reference(
            time, vx_des, vy_des, psi_des, float(xo[0]), float(yo[0])
        )

    time_dec = _decimate(time, stride)
    xo_dec = _decimate(xo, stride)
    yo_dec = _decimate(yo, stride)
    psi_dec = _decimate(psi, stride)
    xref_dec = _decimate(xref_full, stride)
    yref_dec = _decimate(yref_full, stride)

    n_points = len(time_dec)
    dt = float(time[1] - time[0]) if n_rows > 1 else 0.0

    geometry = load_geometry()

    return {
        "meta": {
            "profile": profile,
            "combo": combo,
            "mu": mu,
            "chi": chi,
            "filename": filename,
            "n_rows_full": n_rows,
            "n_points": n_points,
            "dt": dt,
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "ref_source": ref_source,
        },
        "geometry": {
            "l": geometry["l"],
            "h": geometry["h"],
            "R": geometry["R"],
        },
        "time": time_dec.tolist(),
        "actual": {
            "x": xo_dec.tolist(),
            "y": yo_dec.tolist(),
            "psi": psi_dec.tolist(),
        },
        "reference": {
            "x": xref_dec.tolist(),
            "y": yref_dec.tolist(),
        },
    }


def to_json(payload: dict[str, Any]) -> str:
    """Serialize payload to JSON."""
    return json.dumps(payload)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Extract widget payload from an Arrow file")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--combo", type=int, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--chi", type=float, required=True)
    parser.add_argument("--max-points", type=int, default=1500)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    payload = extract(
        args.profile,
        args.combo,
        args.mu,
        args.chi,
        max_points=args.max_points,
    )

    summary = {
        "meta": payload["meta"],
        "geometry": payload["geometry"],
        "array_lengths": {
            "time": len(payload["time"]),
            "actual_x": len(payload["actual"]["x"]),
            "actual_y": len(payload["actual"]["y"]),
            "reference_x": len(payload["reference"]["x"]),
            "reference_y": len(payload["reference"]["y"]),
        },
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(to_json(payload))
        print(f"Wrote full payload to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
