#!/usr/bin/env python
"""Compat gate for accel/ sidecars — run BEFORE the fleet.

MUST be run with the conda myenv Python (`mecanum_pinn` imports torch at
package init: `C:\\Users\\vishv\\miniforge3\\envs\\myenv\\python.exe`).
`mecanum_observer` is torch-free and would also work under the venv, but this
script tests both packages together.

Checks, on copies in code_insights/_tmp/ only (never touches the real fleet):
  (a) the A1 (`mecanum_pinn.data.load_all_arrow_trajectories`) and A2
      (`mecanum_observer.data.discover`) loaders see IDENTICAL output with the
      accel/ subfolder present vs absent;
  (b) neither package, nor datastore.jl / Data_Generation_Julia.jl / run_one.jl,
      contains a recursive Arrow glob (`**/*.arrow`, `rglob`, `os.walk`,
      Julia `walkdir`) that could descend into accel/;
  (c) `load_with_accel` joins an original + its sidecar correctly and raises on
      a stale (fingerprint-mismatched) pairing.

Exits 0 iff every check passes. The fleet run is blocked until this exits 0.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_INSIGHTS = REPO_ROOT / "code_insights"
TOOLS_DIR = CODE_INSIGHTS / "tools_accel"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(CODE_INSIGHTS / "Mecanum_PINN_Mamba_ForceRecon_v1"))
sys.path.insert(0, str(CODE_INSIGHTS / "observer_v1_py"))

import make_accel_sidecars as MS  # noqa: E402

REAL_DATA_DIR = REPO_ROOT / "data" / "Simulation_Data_MecanumSlipSpin_LugreAdamov"
FIXTURE_NAMES = [
    "coupled_vomega_c001_mu_0.3_case1_lugre_adamov_chi_0.000.arrow",
    "spin_creep_c001_mu_0.8_case1_lugre_adamov_chi_0.005.arrow",
]

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def audit_recursive_globs() -> list[str]:
    targets = list((CODE_INSIGHTS / "Mecanum_PINN_Mamba_ForceRecon_v1" / "mecanum_pinn").glob("*.py"))
    targets += list((CODE_INSIGHTS / "observer_v1_py" / "mecanum_observer").glob("*.py"))
    for jl in ("datastore.jl", "Data_Generation_Julia.jl", "run_one.jl"):
        p = CODE_INSIGHTS / jl
        if p.exists():
            targets.append(p)
    pattern = re.compile(r"\*\*/\*\.arrow|\.rglob\(|os\.walk\(|walkdir\(")
    hits = []
    for f in targets:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in pattern.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            hits.append(f"{f.relative_to(CODE_INSIGHTS)}:{line_no}: {m.group(0)}")
    return hits


def main() -> int:
    tmp_root = CODE_INSIGHTS / "_tmp" / "accel_noninterference_test"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    try:
        src_files = [REAL_DATA_DIR / n for n in FIXTURE_NAMES]
        for f in src_files:
            if not f.exists():
                print(f"[abort] fixture file missing: {f}")
                return 2
            shutil.copy2(f, tmp_root / f.name)
        mock_files = sorted(tmp_root.glob("*.arrow"))
        check("fixtures copied", len(mock_files) == len(FIXTURE_NAMES))

        # ---------------------------------------------------------------
        # (a) A1 loader invisibility
        # ---------------------------------------------------------------
        from mecanum_pinn import data as a1_data

        def load_a1():
            return a1_data.load_all_arrow_trajectories(
                data_dir=tmp_root, whitelist=None, mu_values=None, chi_values=None,
                profiles=None, friction_models=None, target_hz=None,
                load_probes=False, cache_dir="", verbose=False)

        before_a1 = load_a1()

        for f in mock_files:
            r = MS.build_sidecar(f, MS.SidecarConfig(data_dir=tmp_root), dry_run=False)
            check(f"sidecar build: {f.name}", r.status == "DONE", r.error)

        after_a1 = load_a1()

        check("A1 loader: same trajectory count with accel/ present",
              len(before_a1) == len(after_a1), f"{len(before_a1)} vs {len(after_a1)}")
        nb = sorted(t["name"] for t in before_a1)
        na = sorted(t["name"] for t in after_a1)
        check("A1 loader: same filenames", nb == na, f"{nb} vs {na}")
        bb = {t["name"]: t for t in before_a1}
        ba = {t["name"]: t for t in after_a1}
        arrays_equal = all(
            np.array_equal(bb[n]["states"], ba[n]["states"])
            and np.array_equal(bb[n]["forces"], ba[n]["forces"])
            and np.array_equal(bb[n]["controls"], ba[n]["controls"])
            for n in nb
        )
        check("A1 loader: identical arrays with accel/ present vs absent", arrays_equal)

        # ---------------------------------------------------------------
        # (a) A2 loader invisibility
        # ---------------------------------------------------------------
        from mecanum_observer.config import ObserverConfig
        from mecanum_observer.data import discover

        parsed = [MS.parse_arrow_filename(f.name) for f in mock_files]
        mus = sorted({p["mu"] for p in parsed})
        chis = sorted({p["chi"] for p in parsed})
        cfg = ObserverConfig(data_dir=tmp_root, whitelist_csv=tmp_root / "no_such_whitelist.csv",
                              mu_values=mus, chi_values=chis).resolved()
        disc = discover(cfg)
        check("A2 discover: only top-level files, accel/ excluded",
              all(p.parent == tmp_root for p in disc),
              f"{[str(p) for p in disc]}")
        check("A2 discover: finds exactly the fixture files",
              sorted(p.name for p in disc) == sorted(f.name for f in mock_files),
              f"{sorted(p.name for p in disc)}")

        # ---------------------------------------------------------------
        # (b) recursive-glob audit
        # ---------------------------------------------------------------
        hits = audit_recursive_globs()
        check("recursive-glob audit clean (mecanum_pinn + mecanum_observer + Julia modules)",
              len(hits) == 0, "; ".join(hits))

        # ---------------------------------------------------------------
        # (c) pairing join
        # ---------------------------------------------------------------
        f0 = mock_files[0]
        orig_table = MS._read_arrow(f0)
        joined = MS.load_with_accel(f0)
        check("load_with_accel: row count preserved", joined.num_rows == orig_table.num_rows,
              f"{joined.num_rows} vs {orig_table.num_rows}")
        check("load_with_accel: sidecar columns present",
              set(MS.SIDECAR_COLS).issubset(set(joined.column_names)))

        st = f0.stat()
        os.utime(f0, (st.st_atime, st.st_mtime + 3600))
        raised = False
        try:
            MS.load_with_accel(f0)
        except ValueError:
            raised = True
        check("load_with_accel raises on stale (fingerprint-mismatched) pairing", raised)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"NONINTERFERENCE TEST: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("NONINTERFERENCE TEST: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
