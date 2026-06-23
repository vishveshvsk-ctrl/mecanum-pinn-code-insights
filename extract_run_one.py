#!/usr/bin/env python3
"""
extract_run_one.py
==================

Convert the active Mecanum ASMC+DOB Julia notebook into a runnable
``run_one.jl`` script that the profile-based parallel sweep driver
(``Data_Generation_Julia.jl``) includes.

This is the PROFILE-PIPELINE extractor. It superseded the old (beta, amplitude)
version: the reference trajectory now lives entirely in ``Profiles``
(``profiles.jl``), so there is no Symbolics ``build_function`` trajectory cell to
wrap, and no ``run_one(params)`` wrapper to synthesize — the driver builds each
reference via ``Profiles.build_job`` / ``publish!`` and calls the lower-level
physics functions (``PlatformParams(BASE; …)``, ``dynamics_full_mf_asmc!``,
``DataStore.compute_labels``, …) directly.

What this does, cell by cell:

  parameters cell        → STRIPPED, replaced by BANNER stand-ins (the driver
                           overrides physics per job; the stand-ins only let the
                           definition cells compile and skip the notebook-only
                           diagnostic plot gated on `!write_data`).
  imports cell           → kept, but `includet(` → `include(` (drop the Revise
                           dependency) and any bare `using Revise` line removed.
                           `include("profiles.jl")` / `include("datastore.jl")`
                           are what put `Profiles` / `DataStore` in `Main` scope.
  definition cells       → kept verbatim (structs, friction model, controllers,
                           the ODE, ASMC/DOB helpers, build_initial_state,
                           run_one_chi, sawtooth_approx, diagnostics).
  trajectory-build cell  → STRIPPED (it consumes the stripped parameter globals
                           PROFILE_FILE/COMBO_IDX/PICK_SEED and publishes a ref).
  main sweep cell        → STRIPPED (interactive; the driver replaces it).
  plotting cells         → STRIPPED (everything downstream of the sweep touches
                           the `all_sols`/`all_labels`/`all_paths` dicts).
  reload cell            → STRIPPED.

Strip detectors (in priority order):
  1. cell carries the `parameters` tag.
  2. trajectory build  — `pick_and_build` present, or both `build_job(` and
     `publish!(` present at the top level.
  3. sweep / plots / reload — references any of `all_sols`, `all_labels`,
     `all_paths`, or calls `reload_run(`. (The single robust signal: every
     consumer of sweep output is interactive.)
  A cell that DEFINES `run_one_chi` (cell with `function run_one_chi`) is kept
  even though it mentions `run_one_chi(`; the sweep cell only CALLS it.

Usage:
    python extract_run_one.py \
        --notebook Mecanum_SlipSpinLuGre_ASMC_DOB_full_supertwist_v4.ipynb \
        --out      run_one.jl
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


BANNER = """# ============================================================================
# run_one.jl  —  AUTO-GENERATED from the Mecanum ASMC+DOB Julia notebook.
#
# Regenerate with:
#   python extract_run_one.py --notebook <notebook>.ipynb --out run_one.jl
#
# Provides (consumed by Data_Generation_Julia.jl after `include`):
#   - Profiles, DataStore            (via include("profiles.jl"/"datastore.jl"))
#   - PlatformParams(base; mu_friction), ASMCParams, ESOParams, LuGreParams
#   - dynamics_full_mf_asmc!, build_initial_state, run_one_chi
#   - asmc_torques, asmc_torques_vel, lugre / lugre_dyn_rates / coupling_of,
#     sawtooth_approx
# DO NOT hand-edit — changes are overwritten on the next extraction. Edit the
# notebook instead.
# ============================================================================

# --- Stand-in defaults for the stripped `parameters` cell. -------------------
# The sweep driver overrides the physics point per job; these exist only so the
# definition cells compile, and so the notebook-only sawtooth diagnostic plot
# (gated on `!write_data`) is SKIPPED at include time.
write_data     = true            # true ⇒ skip the notebook-only diagnostic plot
mu_friction    = 0.5
friction_case  = 1
friction_model = :lugre_adamov
use_dob        = false           # ASMCParams' gamma_y/gamma_psi defaults read this;
                                 # the sweep driver overrides it from base.toml [dob].enable.

"""


def load_cells(nb_path: Path):
    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)
    return nb["cells"]


def cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def is_parameters_cell(cell) -> bool:
    return "parameters" in cell.get("metadata", {}).get("tags", [])


def is_trajectory_build_cell(src: str) -> bool:
    """Cell 14: builds + publishes the reference from the (stripped) parameter
    globals. Identified by the profile-library build/publish calls."""
    return ("pick_and_build" in src) or ("build_job(" in src and "publish!(" in src)


def is_sweep_or_plot_or_reload_cell(src: str) -> bool:
    """Sweep / plotting / reload cells: every consumer of the sweep output
    touches one of the result dicts, and the reload cell calls reload_run."""
    return any(tok in src for tok in ("all_sols", "all_labels", "all_paths", "reload_run("))


def transform_kept_cell(src: str) -> str:
    """Make a kept cell batch-safe: load modules without Revise."""
    out_lines = []
    for line in src.splitlines():
        if line.strip() == "using Revise":
            out_lines.append("# [extract] dropped `using Revise` (batch script, no live reload)")
            continue
        # Revise's includet -> plain include so Profiles/DataStore land in Main.
        out_lines.append(line.replace("includet(", "include("))
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", required=True, type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    args = ap.parse_args()

    if not args.notebook.is_file():
        sys.exit(f"Notebook not found: {args.notebook}")

    cells = load_cells(args.notebook)
    pieces = [BANNER]
    kept, stripped = 0, []

    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue

        src = cell_source(cell)

        if is_parameters_cell(cell):
            pieces.append(f"# --- [skipped: parameters cell {i} — replaced by BANNER stand-ins] ---\n\n")
            stripped.append((i, "parameters"))
            continue
        if is_trajectory_build_cell(src):
            pieces.append(f"# --- [skipped: trajectory-build cell {i} — driver builds refs per job] ---\n\n")
            stripped.append((i, "trajectory-build"))
            continue
        if is_sweep_or_plot_or_reload_cell(src):
            pieces.append(f"# --- [skipped: sweep/plot/reload cell {i}] ---\n\n")
            stripped.append((i, "sweep/plot/reload"))
            continue

        pieces.append(transform_kept_cell(src.rstrip()) + "\n\n")
        kept += 1

    args.out.write_text("".join(pieces), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)  kept={kept} code cells")
    for i, why in stripped:
        print(f"  stripped cell {i}: {why}")


if __name__ == "__main__":
    main()
