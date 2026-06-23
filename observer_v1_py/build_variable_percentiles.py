#!/usr/bin/env python
# =============================================================================
# build_variable_percentiles.py — robust per-channel scale statistics for the
# max-normalization scaler (p95-based, NOT raw max — single outliers wreck max).
#
# Writes a CSV (default to ../data/) with, per model channel, the |value|
# percentiles p50 / p95 / p99 and the exact |min|/|max|, pooled across wheels for
# per-wheel channels. The p95 column is the intended scaler: x_norm = x / p95(|x|)
# (so the 95th-pct magnitude maps to ~1 and the rare outliers sit beyond 1,
# instead of compressing the whole channel toward 0 the way max-scaling would).
#
# MEMORY-SAFE: never holds the whole dataset. Streams one file at a time (via the
# decimated cache), SUBSAMPLES both files (--file-stride) and rows (--row-stride),
# and accumulates only the bounded subsample for the percentile estimate. Exact
# min/max are tracked over the full loaded rows (cheap, no storage). The small
# quantile error from subsampling is acceptable (a p95 scaler is robust by design).
#
# Torch-free (numpy/pandas/pyarrow) — run with claude-venv from code_insights/:
#   PYTHONPATH=observer_v1_py <py> observer_v1_py/build_variable_percentiles.py
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mecanum_observer.data import read_arrays, load_whitelist, _parse_name


def channels(a) -> dict:
    """Map a decimated-arrays dict to flat 1-D per-channel arrays (per-wheel
    channels are raveled across the 4 wheels — the encoder is wheel-shared, so
    one scale per channel)."""
    G, P, Y = a["G"], a["P"], a["Y"]
    return {
        "Vx": G[:, 0], "Vy": G[:, 1], "psi_dot": G[:, 2],
        "Msat": P[:, :, 0].ravel(), "w": P[:, :, 1].ravel(),
        "sin_tt": P[:, :, 2].ravel(), "cos_tt": P[:, :, 3].ravel(),
        "gamma": Y[:, :, 0].ravel(), "zx": Y[:, :, 1].ravel(), "zy": Y[:, :, 2].ravel(),
        "wz": a["wz"].ravel(), "Vpx0": a["Vpx0"].ravel(), "Vpy0": a["Vpy0"].ravel(),
        "vpm": a["vpm"].ravel(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust p50/p95/p99/max per channel (max-norm scaler stats).")
    ap.add_argument("--data-dir", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov")
    ap.add_argument("--whitelist-csv", default="diagnostics_combined.csv")
    ap.add_argument("--cache-dir", default="C:/Users/vishv/mecanum_cache_decim")
    ap.add_argument("--file-stride", type=int, default=1, help="keep every Nth whitelisted file (1 = ALL)")
    ap.add_argument("--row-stride", type=int, default=25, help="keep every Nth row within a file "
                    "(memory guard for percentiles; min/max are exact over all rows regardless)")
    ap.add_argument("--limit", type=int, default=0, help="cap selected files (smoke); 0 = all")
    ap.add_argument("--out", default="../data/Simulation_Data_MecanumSlipSpin_LugreAdamov/variable_scaler_percentiles.csv")
    args = ap.parse_args()

    wl = load_whitelist(Path(args.whitelist_csv))
    files = sorted(Path(args.data_dir).glob("*.arrow"))
    files = [f for f in files if _parse_name(f.name) and (wl is None or f.name in wl)]
    files = files[:: args.file_stride]
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[pct] {len(files)} files sampled (file-stride {args.file_stride}, row-stride {args.row_stride})")

    acc: dict[str, list] = {}
    rmin: dict[str, float] = {}
    rmax: dict[str, float] = {}
    nread = 0
    for j, f in enumerate(files):
        try:
            a = read_arrays(f, args.cache_dir)
        except Exception as e:
            print(f"[pct] skip {f.name}: {e!r}")
            continue
        for name, full in channels(a).items():
            full = np.asarray(full, dtype=np.float64)
            rmin[name] = min(rmin.get(name, np.inf), float(full.min()))   # exact over loaded rows
            rmax[name] = max(rmax.get(name, -np.inf), float(full.max()))
            acc.setdefault(name, []).append(np.abs(full[:: args.row_stride]).astype(np.float32))
        del a
        nread += 1
        if nread % 200 == 0:
            print(f"[pct] {nread}/{len(files)} files")

    rows = []
    for name in acc:
        arr = np.concatenate(acc[name])
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        rows.append(dict(variable=name, n_samples=int(arr.size),
                         abs_p50=float(p50), abs_p95=float(p95), abs_p99=float(p99),
                         abs_max=float(max(abs(rmin[name]), abs(rmax[name]))),
                         raw_min=rmin[name], raw_max=rmax[name],
                         p99_over_p95=float(p99 / max(p95, 1e-12)),
                         max_over_p95=float(max(abs(rmin[name]), abs(rmax[name])) / max(p95, 1e-12))))
    df = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20)
    print(df.round(4).to_string(index=False))
    print(f"\n[pct] wrote {len(df)} channels -> {out}  (from {nread} files)")
    print("[pct] note: max_over_p95 >> 1 flags heavy outliers -> p95 scaler is the right choice there.")


if __name__ == "__main__":
    main()
