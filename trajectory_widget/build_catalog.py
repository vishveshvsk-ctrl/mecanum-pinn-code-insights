"""Build a data-driven catalog of Mecanum simulation Arrow files.

Part 1 of the trajectory-widget build.  Filename parsing only — no Arrow
contents are read during the catalog build.  A separate --verify path opens
one representative file and writes contract.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

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
    r"""Return the authoritative simulation-data directory.

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
    data_dir = (
        repo_root
        / "data"
        / "Simulation_Data_MecanumSlipSpin_LugreAdamov"
    )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found at {data_dir}. "
            "Set MECANUM_DATA_DIR to override."
        )
    return data_dir


# Anchored regex for the authoritative filename contract.
# Example: octagon_c001_mu_0.3_case1_lugre_adamov_chi_0.005.arrow
FILENAME_RE = re.compile(
    r"^(?P<profile>.+)_c(?P<combo>\d{3})_mu_(?P<mu>[^_]+)_case1_lugre_adamov_chi_(?P<chi>[^_]+)\.arrow$"
)


def parse_filename(filename: str) -> dict[str, Any] | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    return {
        "profile": m.group("profile"),
        "combo": int(m.group("combo")),
        "mu": float(m.group("mu")),
        "chi": float(m.group("chi")),
        "filename": filename,
    }


def build_catalog(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = resolve_data_dir()

    entries: list[dict[str, Any]] = []
    skipped: list[str] = []

    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".arrow"):
            continue
        parsed = parse_filename(name)
        if parsed is None:
            skipped.append(name)
            continue
        entries.append(parsed)

    profiles = sorted({e["profile"] for e in entries})
    mus = sorted({e["mu"] for e in entries})
    chis = sorted({e["chi"] for e in entries})
    combos = sorted({e["combo"] for e in entries})

    valid_combos_per_profile: dict[str, list[int]] = {}
    for profile in profiles:
        valid_combos_per_profile[profile] = sorted(
            {e["combo"] for e in entries if e["profile"] == profile}
        )

    return {
        "data_dir": str(data_dir),
        "n_files": len(entries),
        "profiles": profiles,
        "mus": mus,
        "chis": chis,
        "combos": combos,
        "valid_combos_per_profile": valid_combos_per_profile,
        "entries": entries,
        "skipped": skipped,
        "n_skipped": len(skipped),
    }


def load_base_geometry(data_dir: Path | None = None) -> dict[str, float]:
    """Read [platform.geometry] from a base.toml next to the data tree.

    We look for trajectory_files_run_0p3_main/base.toml in the repo root.
    If anything fails, we fall back to the documented defaults and note it.
    """
    defaults = {"R": 0.05, "Ra": 0.0355, "h": 0.235, "l": 0.15}
    try:
        repo_root = resolve_repo_root()
        base_toml = repo_root / "code_insights" / "trajectory_files_run_0p3_main" / "base.toml"
        if not base_toml.is_file():
            base_toml = repo_root / "trajectory_files_run_0p3_main" / "base.toml"
        if not base_toml.is_file():
            return {**defaults, "_fallback": True}
        with open(base_toml, "rb") as f:
            cfg = tomllib.load(f)
        geom = cfg.get("platform", {}).get("geometry", {})
        return {
            "R": float(geom.get("R", defaults["R"])),
            "Ra": float(geom.get("Ra", defaults["Ra"])),
            "h": float(geom.get("h", defaults["h"])),
            "l": float(geom.get("l", defaults["l"])),
            "_source": str(base_toml),
        }
    except Exception:
        return {**defaults, "_fallback": True}


def verify_contract(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = resolve_data_dir()

    catalog = build_catalog(data_dir)
    if not catalog["entries"]:
        raise RuntimeError("No catalog entries; cannot verify contract")

    sample = catalog["entries"][0]
    sample_path = data_dir / sample["filename"]
    table = feather.read_table(sample_path)
    columns = table.column_names

    required = [
        "time",
        "Xo",
        "Yo",
        "psi",
        "Vx_des",
        "Vy_des",
        "psi_des",
        "omega_des",
    ]
    missing = [c for c in required if c not in columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    time_arr = table["time"].to_numpy()
    row_count = len(time_arr)
    t_min = float(time_arr.min())
    t_max = float(time_arr.max())

    has_posref = ("xo_des" in columns) and ("yo_des" in columns)
    geom = load_base_geometry(data_dir)

    contract: dict[str, Any] = {
        "required_columns": required,
        "sample_file": str(sample_path),
        "sample_meta": {
            "profile": sample["profile"],
            "combo": sample["combo"],
            "mu": sample["mu"],
            "chi": sample["chi"],
        },
        "row_count": row_count,
        "time_range": {"min": t_min, "max": t_max},
        "geometry": {
            "R": geom["R"],
            "Ra": geom["Ra"],
            "h": geom["h"],
            "l": geom["l"],
            "chassis_length_x": 2.0 * geom["h"],
            "chassis_width_y": 2.0 * geom["l"],
        },
        "reference_path": {
            "stored_posref": has_posref,
            "notes": (
                "World-frame reference XY is NOT stored for VelRef profiles; "
                "it must be reconstructed by integrating desired body velocity. "
                "Exception: ellipse (PosRef) provides xo_des/yo_des and can be used directly."
            ),
        },
    }
    if "_source" in geom:
        contract["geometry"]["_source"] = geom["_source"]
    if geom.get("_fallback"):
        contract["geometry"]["_fallback"] = True
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Mecanum trajectory catalog")
    parser.add_argument("--verify", action="store_true", help="Also verify contract against one Arrow file")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--catalog", type=str, default=None, help="Output catalog JSON path")
    parser.add_argument("--contract", type=str, default=None, help="Output contract JSON path")
    args = parser.parse_args()

    data_dir: Path | None = Path(args.data_dir) if args.data_dir else None

    catalog = build_catalog(data_dir)
    print(f"Matched {catalog['n_files']} files, skipped {catalog['n_skipped']}")
    if catalog["skipped"]:
        print(f"First 10 skipped: {catalog['skipped'][:10]}")

    here = Path(__file__).resolve().parent
    catalog_path = Path(args.catalog) if args.catalog else here / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"Wrote {catalog_path}")

    if args.verify:
        contract = verify_contract(data_dir)
        contract_path = Path(args.contract) if args.contract else here / "contract.json"
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2)
        print(f"Wrote {contract_path}")
        print(
            f"Sample {contract['sample_meta']} -> rows={contract['row_count']}, "
            f"t∈[{contract['time_range']['min']:.4f}, {contract['time_range']['max']:.4f}]"
        )
        print(
            f"Geometry: h={contract['geometry']['h']}, l={contract['geometry']['l']}, "
            f"R={contract['geometry']['R']}, chassis={contract['geometry']['chassis_length_x']}×"
            f"{contract['geometry']['chassis_width_y']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
