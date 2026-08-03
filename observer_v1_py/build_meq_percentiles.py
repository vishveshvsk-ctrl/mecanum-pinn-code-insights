#!/usr/bin/env python
# =============================================================================
# build_meq_percentiles.py — robust p50/p95/p99/max scaler stats for the ASMC
# equivalent-control torque channel (M_eq, per-wheel Meq_1..4) and two derived
# saturation/switching-gap channels, upserted into the existing
# variable_scaler_percentiles.csv (produced by build_variable_percentiles.py,
# which reads Msat_i but never Meq_i/Msw_i):
#   M_sat_minus_M_eq         = Msat_i - Meq_i               (saturation gap)
#   M_sat_minus_M_eq_minus_M_sw = Msat_i - Meq_i - Msw_i     (residual after
#                                 also removing the switching term; ideally
#                                 ~0 pre-saturation, nonzero only where the
#                                 sat() clip actually bites)
#
# Mirrors build_variable_percentiles.py's streaming map-reduce (one file at a
# time, abs-value subsample for percentiles, exact min/max over all decimated
# rows). Meq_i/Msat_i/Msw_i are stored root-Arrow columns (no accel sidecar,
# no LPF needed — control-law outputs, decimated exactly like Msat/w in v1)
# and are pooled across the 4 wheels the same way Msat is (one scale for the
# wheel-shared encoder). Diff channels are formed AFTER decimation (pure
# downsampling with no filtering in between, so subtract-then-decimate and
# decimate-then-subtract are identical; decimating first avoids reading each
# column group more than once). Writer is the same byte-preserving upsert as
# build_accel_percentiles.py, so the existing rows are never disturbed and
# re-running is idempotent.
#
# Torch-free (numpy/pandas/pyarrow) — run with claude-venv from code_insights/:
#   PYTHONPATH=observer_v1_py <py> observer_v1_py/build_meq_percentiles.py
# =============================================================================
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from mecanum_observer.data import load_whitelist, _parse_name
from mecanum_observer import config as C

MEQ_COLS = [f"Meq_{i}" for i in range(1, C.N_WHEELS + 1)]
MSAT_COLS = [f"Msat_{i}" for i in range(1, C.N_WHEELS + 1)]
MSW_COLS = [f"Msw_{i}" for i in range(1, C.N_WHEELS + 1)]
NEW_CHANNELS = ["M_eq", "M_sat_minus_M_eq", "M_sat_minus_M_eq_minus_M_sw"]


# ---------------------------------------------------------------------------
# Read + decimate
# ---------------------------------------------------------------------------
def read_meq(path: Path) -> dict:
    """Read Meq_1..4, Msat_1..4, Msw_1..4 straight from the root Arrow file at
    native 2000 Hz. Returns {'Meq','Msat','Msw'}: each [T_native, 4] float64."""
    tbl = feather.read_table(path, columns=MEQ_COLS + MSAT_COLS + MSW_COLS)
    def _stack(cols):
        return np.stack([tbl.column(c).to_numpy().astype(np.float64) for c in cols], axis=1)
    return {"Meq": _stack(MEQ_COLS), "Msat": _stack(MSAT_COLS), "Msw": _stack(MSW_COLS)}


def meq_channels(raw: dict, decim: int) -> dict:
    """Decimate to 500 Hz (same phase as v1: x[::DECIM]) and ravel across the
    4 wheels — one scale for the wheel-shared encoder, matching how Msat/w/
    gamma are pooled in v1 `channels()`. Also forms the two saturation/
    switching-gap channels:
      M_sat_minus_M_eq            = Msat_i - Meq_i
      M_sat_minus_M_eq_minus_M_sw = Msat_i - Meq_i - Msw_i
    Returns {'M_eq', 'M_sat_minus_M_eq', 'M_sat_minus_M_eq_minus_M_sw'}, each
    [4*T500]."""
    meq = raw["Meq"][::decim]                  # [T500, 4]
    msat = raw["Msat"][::decim]                 # [T500, 4]
    msw = raw["Msw"][::decim]                   # [T500, 4]
    gap1 = msat - meq
    gap2 = msat - meq - msw
    return {"M_eq": meq.ravel(),
            "M_sat_minus_M_eq": gap1.ravel(),
            "M_sat_minus_M_eq_minus_M_sw": gap2.ravel()}


# ---------------------------------------------------------------------------
# Streaming reducer (v1 pattern, verbatim in spirit)
# ---------------------------------------------------------------------------
def accumulate_stats(acc: dict, rmin: dict, rmax: dict, chans: dict, row_stride: int) -> None:
    """Fold one file: append abs(chan)[::row_stride] (float32) to acc[name];
    update rmin/rmax with exact min/max over ALL decimated rows. Mutates in place."""
    for name, full in chans.items():
        full = np.asarray(full, dtype=np.float64)
        rmin[name] = min(rmin.get(name, np.inf), float(full.min()))
        rmax[name] = max(rmax.get(name, -np.inf), float(full.max()))
        acc.setdefault(name, []).append(np.abs(full[:: row_stride]).astype(np.float32))


def finalize_rows(acc: dict, rmin: dict, rmax: dict) -> list:
    """Reduce accumulators to CSV rows using the v1 schema/formulas exactly:
    variable, n_samples, abs_p50, abs_p95, abs_p99, abs_max, raw_min, raw_max,
    p99_over_p95, max_over_p95."""
    rows = []
    for name in NEW_CHANNELS:
        if name not in acc:
            continue
        arr = np.concatenate(acc[name])
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        abs_max = float(max(abs(rmin[name]), abs(rmax[name])))
        rows.append(dict(variable=name, n_samples=int(arr.size),
                         abs_p50=float(p50), abs_p95=float(p95), abs_p99=float(p99),
                         abs_max=abs_max, raw_min=rmin[name], raw_max=rmax[name],
                         p99_over_p95=float(p99 / max(p95, 1e-12)),
                         max_over_p95=float(abs_max / max(p95, 1e-12))))
    return rows


# ---------------------------------------------------------------------------
# Byte-preserving upsert (identical to build_accel_percentiles.py)
# ---------------------------------------------------------------------------
def upsert_rows(csv_path: Path, new_rows: list, protected: set):
    """Read existing CSV preserving untouched rows' exact text; replace rows whose
    'variable' is in the new set; append the rest; write back. Never edits a row not
    in the new set. Returns (n_kept, n_added, n_replaced)."""
    csv_path = Path(csv_path)
    columns = list(new_rows[0].keys())
    existing_vars = set()
    kept_lines = []
    header = ",".join(columns)
    if csv_path.exists():
        raw_lines = csv_path.read_text().splitlines()
        if raw_lines:
            header = raw_lines[0]
            columns = header.split(",")
            for line in raw_lines[1:]:
                if not line:
                    continue
                var = line.split(",", 1)[0]
                existing_vars.add(var)
                if var not in protected:
                    kept_lines.append(line)

    n_added = sum(1 for r in new_rows if r["variable"] not in existing_vars)
    n_replaced = sum(1 for r in new_rows if r["variable"] in existing_vars)
    n_kept = len(existing_vars) - n_replaced

    new_lines = [",".join(str(r[c]) for c in columns) for r in new_rows]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join([header] + kept_lines + new_lines) + "\n")
    return (n_kept, n_added, n_replaced)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def write_provenance(args, files, nread, nskip) -> Path:
    path = Path(args.out).parent / "variable_scaler_percentiles.provenance_meq.json"
    payload = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="build_meq_percentiles.py",
        data_dir=str(args.data_dir),
        whitelist_csv=str(args.whitelist_csv),
        file_stride=args.file_stride,
        row_stride=args.row_stride,
        limit=args.limit,
        n_files_selected=len(files),
        n_files_used=nread,
        n_files_skipped=nskip,
        sim_hz=float(C.SIM_HZ),
        decim=int(C.DECIM),
        train_hz=float(C.SIM_HZ) / int(C.DECIM),
        meq_cols=MEQ_COLS,
        msat_cols=MSAT_COLS,
        msw_cols=MSW_COLS,
        channels=NEW_CHANNELS,
        formulas={
            "M_sat_minus_M_eq": "Msat_i - Meq_i",
            "M_sat_minus_M_eq_minus_M_sw": "Msat_i - Meq_i - Msw_i",
        },
        formula_note="all three source columns decimated to 500 Hz then differenced "
                     "(pure downsampling, no LPF, so decimate-then-subtract == subtract-then-decimate)",
        cache="none (reads Meq_i/Msat_i/Msw_i straight from root Arrow every call, no npz cache)",
    )
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Robust p50/p95/p99/max scaler stats for the ASMC equivalent-"
                    "control torque channel (M_eq); upserted into "
                    "variable_scaler_percentiles.csv.")
    ap.add_argument("--data-dir", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--whitelist-csv", default="diagnostics_combined.csv")
    ap.add_argument("--file-stride", type=int, default=1, help="keep every Nth whitelisted file (1 = ALL)")
    ap.add_argument("--row-stride", type=int, default=25, help="keep every Nth row within a file "
                    "(memory guard for percentiles; min/max are exact over all rows regardless)")
    ap.add_argument("--limit", type=int, default=0, help="cap selected files (smoke); 0 = all")
    ap.add_argument("--out", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv")
    ap.add_argument("--no-provenance", dest="write_provenance", action="store_false", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    wl = load_whitelist(Path(args.whitelist_csv))
    files = sorted(data_dir.glob("*.arrow"))
    files = [f for f in files if _parse_name(f.name) and (wl is None or f.name in wl)]
    files = files[:: args.file_stride]
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[meq-pct] {len(files)} files selected (file-stride {args.file_stride}, "
         f"row-stride {args.row_stride}, cache=none)")

    acc: dict = {}
    rmin: dict = {}
    rmax: dict = {}
    nread, nskip = 0, 0
    for j, f in enumerate(files):
        try:
            raw = read_meq(f)
        except Exception as e:
            print(f"[meq-pct] skip {f.name}: {e!r}")
            nskip += 1
            continue
        chans = meq_channels(raw, C.DECIM)
        accumulate_stats(acc, rmin, rmax, chans, args.row_stride)
        del raw, chans
        nread += 1
        if nread % 200 == 0:
            print(f"[meq-pct] {nread}/{len(files)} files")

    rows = finalize_rows(acc, rmin, rmax)
    out_path = Path(args.out)
    n_kept, n_added, n_replaced = upsert_rows(out_path, rows, protected={r["variable"] for r in rows})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20)
    print(df.round(6).to_string(index=False))
    print(f"\n[meq-pct] {nread} files read, {nskip} skipped (of {len(files)} selected)")
    print(f"[meq-pct] upsert: kept={n_kept} added={n_added} replaced={n_replaced} -> {out_path}")

    if args.write_provenance:
        prov_path = write_provenance(args, files, nread, nskip)
        print(f"[meq-pct] provenance -> {prov_path}")


if __name__ == "__main__":
    main()
