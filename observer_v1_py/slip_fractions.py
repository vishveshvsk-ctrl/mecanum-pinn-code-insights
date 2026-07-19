#!/usr/bin/env python
# =============================================================================
# slip_fractions.py — GROUND-TRUTH contact-slip regime fractions from the
# lugre_adamov sim data (NOT the model-derived gate g).
#
# Per wheel i the true contact slip speed is v_slip = sqrt(Vpx_i^2 + Vpy_i^2)
# (simulator columns Vpx_{i}/Vpy_{i}, decimated to 500 Hz -> the cached `vpm`).
# We POOL all 4 wheels x all timesteps and bin into 3 regimes:
#     stick      : v_slip <  v_stribeck                (LG_V_STR)
#     slip       : v_stribeck <= v_slip < 0.6 m/s
#     high_slip  : v_slip >= 0.6 m/s
# and report the fraction of (wheel,timestep) samples in each — a single number
# per group, combined across the 4 wheels.
#
# Reuses the warm hy3 cache (reads `vpm` from read_arrays_hy3; no model needed).
#   python observer_v1_py/slip_fractions.py --mode by-chi
#   python observer_v1_py/slip_fractions.py --mode regime \
#          --regime observer_v1_py/regimes/eval_multisine.toml --group-by profile
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather  # noqa: F401

from mecanum_observer.config_v2hy3 import ObserverConfigV2Hy3
from mecanum_observer import config as C
from mecanum_observer import data as D
from mecanum_observer import data_v2hy3 as D2H

RUNS_DEFAULT = ["observer_v1_py/runs/S1_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm",
                "observer_v1_py/runs/S2_train_hy3_w32_gamma_dv_v2hy3_phys_max_norm"]
FIELDS = ObserverConfigV2Hy3.__dataclass_fields__
V_STRIBECK = C.LG_V_STR          # stick / slip boundary (m/s)
V_HIGH = 0.6                     # slip / high-slip boundary (m/s)


def _cfg_from_run(run_dir: Path) -> ObserverConfigV2Hy3:
    m = json.load(open(run_dir / "metrics.json"))
    cfg = ObserverConfigV2Hy3(**{k: v for k, v in m["cfg"].items() if k in FIELDS})
    cfg.jobs = 0
    return cfg.resolved()


def slip_fractions(files, cfg):
    """Pool v_slip over all 4 wheels x all 500 Hz timesteps; 3-bin fractions."""
    n_stick = n_slip = n_high = 0
    total = 0
    vmax = 0.0
    ssum = 0.0
    for p in files:
        a = D2H.read_arrays_hy3(p, cfg)
        if a is None:
            continue
        v = np.asarray(a["vpm"], dtype=np.float64).reshape(-1)   # [T*4] all wheels
        if v.size == 0:
            continue
        n_stick += int((v < V_STRIBECK).sum())
        n_slip += int(((v >= V_STRIBECK) & (v < V_HIGH)).sum())
        n_high += int((v >= V_HIGH).sum())
        total += v.size
        vmax = max(vmax, float(v.max()))
        ssum += float(v.sum())
    if total == 0:
        return None
    return dict(n=total, stick=n_stick / total, slip=n_slip / total,
                high=n_high / total, vmean=ssum / total, vmax=vmax)


def _row(tag, r):
    if r is None:
        print(f"  {tag:<26} (no data)"); return
    print(f"  {tag:<26} n={r['n']:>9} | stick={r['stick']*100:6.2f}%  "
          f"slip={r['slip']*100:6.2f}%  high_slip={r['high']*100:6.2f}%  "
          f"| v_mean={r['vmean']:.3f} v_max={r['vmax']:.2f} m/s")


def main():
    ap = argparse.ArgumentParser(description="Ground-truth contact-slip regime fractions.")
    ap.add_argument("--runs", nargs="+", default=RUNS_DEFAULT)
    ap.add_argument("--mode", choices=["by-chi", "regime"], required=True)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--group-by", choices=["none", "profile"], default="none")
    ap.add_argument("--per-group-cap", type=int, default=200,
                    help="strided subsample cap per group (0 = all)")
    args = ap.parse_args()

    print(f"[bins] stick: v<{V_STRIBECK}  slip: {V_STRIBECK}<=v<{V_HIGH}  "
          f"high_slip: v>={V_HIGH} m/s  (v_slip = sqrt(Vpx_i^2+Vpy_i^2), pooled over 4 wheels)")

    def strided(fl):
        c = args.per_group_cap
        return fl[:: max(1, len(fl) // c)] if c and len(fl) > c else fl

    if args.mode == "by-chi":
        for run_dir in args.runs:
            run_dir = Path(run_dir)
            cfg = _cfg_from_run(run_dir)
            cfg.chi_values = list(C.CHI_GRID)
            splits = D.split_files(D.discover(cfg), cfg)
            test = splits["test"]
            cross = "S2" if (cfg.train_fold or run_dir.name[:2]) == "S1" else "S1"
            by = {}
            for p in test:
                m = D._parse_name(p.name)
                if m:
                    by.setdefault(round(m["chi"], 3), []).append(p)
            print(f"\n########## {run_dir.name} : cross-test fold {cross}, by chi ##########")
            for chi in sorted(by):
                fl = strided(by[chi])
                _row(f"chi={chi} ({len(fl)}/{len(by[chi])}f)", slip_fractions(fl, cfg))
    else:
        cfg = _cfg_from_run(Path(args.runs[0]))
        reg = D.regime_to_kwargs(D.load_regime(Path(args.regime)))
        for k, v in reg.items():
            if k in FIELDS:
                setattr(cfg, k, v)
        cfg.train_fold = ""
        files = D.discover(cfg)
        print(f"\n########## regime {args.regime} : {len(files)} files ##########")
        groups = {"all": files}
        if args.group_by == "profile":
            groups = {}
            for p in files:
                m = D._parse_name(p.name)
                groups.setdefault(m["profile"] if m else "??", []).append(p)
        for gk in sorted(groups):
            fl = strided(groups[gk])
            _row(f"{gk} ({len(fl)}/{len(groups[gk])}f)", slip_fractions(fl, cfg))
    print("\ndone")


if __name__ == "__main__":
    main()
