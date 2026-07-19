#!/usr/bin/env python
# =============================================================================
# build_accel_percentiles.py — robust p50/p95/p99/max scaler stats for the
# ACCELERATION channels (a_x, a_y, dVx, dVy, dpsi_dot, dw), upserted into the
# existing variable_scaler_percentiles.csv (produced by build_variable_percentiles.py
# before the accel data existed -> 14 rows, no accel scalers).
#
# Mirrors build_variable_percentiles.py's streaming map-reduce (one file at a
# time, abs-value subsample for percentiles, exact min/max over all decimated
# rows) with three additions: (a) a paired root+accel read, (b) the IMU
# accelerometer-observable derivation + anti-alias LPF before decimation, and
# (c) a byte-preserving upsert writer (the existing 14 rows must not change).
#
# Torch-free (numpy/pandas/pyarrow/scipy.signal) — run with claude-venv from
# code_insights/:
#   PYTHONPATH=observer_v1_py <py> observer_v1_py/build_accel_percentiles.py
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.signal import butter, filtfilt

from mecanum_observer.data import load_whitelist, _parse_name
from mecanum_observer import config as C
from mecanum_observer.sensor_frontend_v2 import ANTI_ALIAS_CUTOFF_HZ as SF_ANTI_ALIAS_CUTOFF_HZ

ROOT_COLS = ["Vx", "Vy", "psi_dot"]
ACCEL_COLS = ["dVx", "dVy", "dpsidot", "dw1", "dw2", "dw3", "dw4"]
NEW_CHANNELS = ["a_x", "a_y", "dVx", "dVy", "dpsi_dot", "dw"]


@dataclass
class LPFSpec:
    cutoff_hz: float
    order: int
    fs_hz: float
    enabled: bool


# ---------------------------------------------------------------------------
# Paired read
# ---------------------------------------------------------------------------
def paired_read(root_path: Path, accel_dir: Path) -> dict:
    """Memory-map root + accel arrows for one stem; assert equal row counts.
    Returns native-2000 Hz float64 columns:
      root : Vx, Vy, psi_dot            [T_native]
      accel: dVx, dVy, dpsidot, dw1..4  [T_native]
    Raises on a missing/short accel sidecar (reported, file skipped by caller)."""
    accel_path = accel_dir / f"{root_path.stem}_accel.arrow"
    if not accel_path.exists():
        raise FileNotFoundError(f"missing accel sidecar: {accel_path}")
    root_tbl = feather.read_table(root_path, columns=ROOT_COLS)
    accel_tbl = feather.read_table(accel_path, columns=ACCEL_COLS)
    if root_tbl.num_rows != accel_tbl.num_rows:
        raise ValueError(f"row mismatch {root_path.name}: root {root_tbl.num_rows} "
                         f"!= accel {accel_tbl.num_rows}")
    out = {}
    for name in ROOT_COLS:
        out[name] = root_tbl.column(name).to_numpy().astype(np.float64)
    for name in ACCEL_COLS:
        out[name] = accel_tbl.column(name).to_numpy().astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Anti-alias LPF + accelerometer observable
# ---------------------------------------------------------------------------
def apply_antialias_lpf(x: np.ndarray, lpf: LPFSpec) -> np.ndarray:
    """Forward-backward Butterworth low-pass at fs = lpf.fs_hz, cutoff lpf.cutoff_hz.
    Identity when lpf.enabled is False. Input/Output: [T_native].
    Zero-phase (filtfilt), unlike the deployed causal one-pole filter in
    sensor_frontend_v2 -- this is an offline stats job, not a deployable filter,
    so there is no causality constraint and zero-phase avoids biasing the tails."""
    if not lpf.enabled:
        return x
    nyq = lpf.fs_hz / 2.0
    b, a = butter(lpf.order, lpf.cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def accel_channels(paired: dict, lpf: LPFSpec, decim: int) -> dict:
    """Form accelerometer observable at native rate, LPF, then stride-decimate.
      a_x = dVx - psi_dot*Vy ;  a_y = dVy + psi_dot*Vx     (transport term)
    Order per file: build a_x/a_y at 2000 Hz -> LPF all accel-derived channels ->
    x[::decim]. Body channels -> [T500]; 'dw' -> ravel(dw1..dw4 decimated) [4*T500]."""
    Vx, Vy, psi_dot = paired["Vx"], paired["Vy"], paired["psi_dot"]
    a_x = paired["dVx"] - psi_dot * Vy
    a_y = paired["dVy"] + psi_dot * Vx

    def _filt_decim(x: np.ndarray) -> np.ndarray:
        return apply_antialias_lpf(x, lpf)[::decim]

    out = {
        "a_x": _filt_decim(a_x),
        "a_y": _filt_decim(a_y),
        "dVx": _filt_decim(paired["dVx"]),
        "dVy": _filt_decim(paired["dVy"]),
        "dpsi_dot": _filt_decim(paired["dpsidot"]),
    }
    dw = np.concatenate([_filt_decim(paired[f"dw{i}"]) for i in range(1, 5)])
    out["dw"] = dw
    return out


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
    """Reduce to CSV rows using the v1 schema/formulas exactly:
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
# Byte-preserving upsert
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
# Verification gate (§10) — run on a handful of files BEFORE the full sweep
# ---------------------------------------------------------------------------
def run_verification_gate(files: list, accel_dir: Path) -> bool:
    """Alignment gate + transport-term justification: stored dVx/dVy/dpsidot
    should match a central finite difference of Vx/Vy/psi_dot (confirming they
    are the pure component derivatives, NOT already specific force). A large
    mismatch here means the transport-term formula / column mapping is wrong."""
    print(f"[accel-pct] verification gate on {len(files)} file(s)...")
    ok = True
    dt = 1.0 / float(C.SIM_HZ)
    for f in files:
        try:
            paired = paired_read(f, accel_dir)
        except Exception as e:
            print(f"[accel-pct]   {f.name}: FAIL paired_read ({e!r})")
            ok = False
            continue
        for stored_key, raw_key in (("dVx", "Vx"), ("dVy", "Vy"), ("dpsidot", "psi_dot")):
            stored = paired[stored_key]
            fd = np.gradient(paired[raw_key], dt)
            scale = max(float(np.percentile(np.abs(stored), 95)), 1e-9)
            rel_err = float(np.percentile(np.abs(stored - fd), 95) / scale)
            status = "OK" if rel_err < 0.05 else "WARN"
            if status == "WARN":
                ok = False
            print(f"[accel-pct]   {f.name} {stored_key} vs FD({raw_key}): "
                 f"p95 rel-err={rel_err:.4f} -> {status}")
    return ok


def check_sample_consistency(rows: list, csv_path: Path) -> None:
    """n_samples for body-level accel channels should equal the existing Vx
    n_samples (same file set + row_stride); 'dw' should equal the existing 'w'
    n_samples. A mismatch means the file set or strides drifted from the v1 run."""
    if not csv_path.exists():
        return
    existing = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            existing[row["variable"]] = int(row["n_samples"])
    body_ref, wheel_ref = existing.get("Vx"), existing.get("w")
    for r in rows:
        name, n = r["variable"], r["n_samples"]
        ref, ref_name = (wheel_ref, "w") if name == "dw" else (body_ref, "Vx")
        status = "OK" if ref is not None and n == ref else "MISMATCH"
        print(f"[accel-pct] n_samples check: {name}={n} vs {ref_name}={ref} -> {status}")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def write_provenance(args, files, nread, nskip, lpf: LPFSpec) -> Path:
    path = Path(args.out).parent / "variable_scaler_percentiles.provenance.json"
    payload = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="build_accel_percentiles.py",
        data_dir=str(args.data_dir),
        accel_subdir=args.accel_subdir,
        whitelist_csv=str(args.whitelist_csv),
        file_stride=args.file_stride,
        row_stride=args.row_stride,
        limit=args.limit,
        n_files_selected=len(files),
        n_files_used=nread,
        n_files_skipped=nskip,
        lpf_cutoff_hz=lpf.cutoff_hz,
        lpf_order=lpf.order,
        lpf_enabled=lpf.enabled,
        sim_hz=float(C.SIM_HZ),
        decim=int(C.DECIM),
        train_hz=float(C.SIM_HZ) / int(C.DECIM),
        transport_term_formula="a_x = dVx - psi_dot*Vy; a_y = dVy + psi_dot*Vx",
        channels=NEW_CHANNELS,
    )
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Robust p50/p95/p99/max scaler stats for acceleration channels "
                    "(a_x, a_y, dVx, dVy, dpsi_dot, dw); upserted into "
                    "variable_scaler_percentiles.csv.")
    ap.add_argument("--data-dir", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--accel-subdir", default="accel")
    ap.add_argument("--whitelist-csv", default="diagnostics_combined.csv")
    ap.add_argument("--file-stride", type=int, default=1, help="keep every Nth whitelisted file (1 = ALL)")
    ap.add_argument("--row-stride", type=int, default=25, help="keep every Nth row within a file "
                    "(memory guard for percentiles; min/max are exact over all rows regardless)")
    ap.add_argument("--limit", type=int, default=0, help="cap selected files (smoke); 0 = all")
    ap.add_argument("--out", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv")
    ap.add_argument("--lpf-cutoff-hz", type=float, default=SF_ANTI_ALIAS_CUTOFF_HZ,
                    help="Butterworth anti-alias cutoff at native 2000 Hz (< 250 Hz decimated Nyquist); "
                        "defaults to sensor_frontend_v2.ANTI_ALIAS_CUTOFF_HZ")
    ap.add_argument("--lpf-order", type=int, default=4)
    ap.add_argument("--no-lpf", action="store_true", help="disable the anti-alias LPF (aliasing-headroom audit)")
    ap.add_argument("--verify-files", type=int, default=5,
                    help="run the alignment+FD verification gate on this many files before the full "
                        "sweep; 0 = skip")
    ap.add_argument("--no-provenance", dest="write_provenance", action="store_false", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    accel_dir = data_dir / args.accel_subdir
    lpf = LPFSpec(cutoff_hz=args.lpf_cutoff_hz, order=args.lpf_order,
                 fs_hz=float(C.SIM_HZ), enabled=not args.no_lpf)

    wl = load_whitelist(Path(args.whitelist_csv))
    files = sorted(data_dir.glob("*.arrow"))
    files = [f for f in files if _parse_name(f.name) and (wl is None or f.name in wl)]
    files = files[:: args.file_stride]
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[accel-pct] {len(files)} files selected (file-stride {args.file_stride}, "
         f"row-stride {args.row_stride}, "
         f"lpf={'off' if args.no_lpf else f'{args.lpf_cutoff_hz:g}Hz/order{args.lpf_order}'})")

    if args.verify_files > 0:
        gate_ok = run_verification_gate(files[: args.verify_files], accel_dir)
        print(f"[accel-pct] verification gate: {'PASS' if gate_ok else 'WARN (see above)'}")

    acc: dict = {}
    rmin: dict = {}
    rmax: dict = {}
    nread, nskip = 0, 0
    for j, f in enumerate(files):
        try:
            paired = paired_read(f, accel_dir)
        except Exception as e:
            print(f"[accel-pct] skip {f.name}: {e!r}")
            nskip += 1
            continue
        chans = accel_channels(paired, lpf, C.DECIM)
        accumulate_stats(acc, rmin, rmax, chans, args.row_stride)
        del paired, chans
        nread += 1
        if nread % 200 == 0:
            print(f"[accel-pct] {nread}/{len(files)} files")

    rows = finalize_rows(acc, rmin, rmax)
    out_path = Path(args.out)
    check_sample_consistency(rows, out_path)
    n_kept, n_added, n_replaced = upsert_rows(out_path, rows, protected={r["variable"] for r in rows})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20)
    print(df.round(6).to_string(index=False))
    print(f"\n[accel-pct] {nread} files read, {nskip} skipped (of {len(files)} selected)")
    print(f"[accel-pct] upsert: kept={n_kept} added={n_added} replaced={n_replaced} -> {out_path}")

    if args.write_provenance:
        prov_path = write_provenance(args, files, nread, nskip, lpf)
        print(f"[accel-pct] provenance -> {prov_path}")


if __name__ == "__main__":
    main()
