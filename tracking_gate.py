#!/usr/bin/env python
# =============================================================================
# tracking_gate.py — flags trajectories that MISSED their reference.
#
# Complements the other two screens, which it is ORTHOGONAL to:
#   * chatter_diagnostics  -> force-signal contamination (chatter / hash)
#   * sampling_sensitivity -> rate fidelity
#   * tracking_gate (this) -> did the platform actually achieve the commanded
#                             trajectory, or did it lose tracking / exceed the
#                             friction circle?
#
# A trajectory can be perfectly 'clean' (uncontaminated forces) yet have
# completely missed its reference because it ran out of friction — the chatter
# screen cannot catch that (the saturated controller rails smoothly, it does not
# chatter), and the hash test even goes blind there (high slip inflates F_hash
# above the Nyquist). This gate catches it directly from the recorded data.
#
# Metrics per trajectory:
#   track_err          worst-channel normalised RMS error of actual vs desired.
#                      VelRef: max over (Vx,Vy,psi_dot) of RMS(actual-des)/scale;
#                      PosRef: RMS path error |(Xo,Yo)-(xo_des,yo_des)| / path scale.
#   util_viol_frac     fraction of time max-wheel util = |F|/(mu*N) > 1.10
#                      (1.10, not 1.0: tolerate riding AT the friction circle —
#                      at-limit saturation is valid data, not loss of control)
#                      (realized friction-circle violation — the post-solve truth
#                      the feedforward design gate could not guarantee).
#   util_budget_frac   fraction of time > 0.8 (the design budget).
#   msat_rail_frac     fraction of time a wheel torque is railed (|Msat| > 0.95*Max_torque)
#                      = the controller saturating (smooth give-up, not chatter).
#
# Verdict: 'missed' (track_err or util_viol over threshold), 'marginal', 'tracked'.
# Thresholds are first-principles PLACEHOLDERS pending a per-profile tuning pass.
#
#   python tracking_gate.py --data-dir ../data/Simulation_... --out tracking_report.csv
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import chatter_diagnostics as cd            # parse_arrow_filename

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MAX_TORQUE = 10.0       # base.toml [platform] Max_torque (per-wheel cap, N·m)
V_FLOOR = 0.05          # m/s   characteristic-speed floor for normalisation
POS_FLOOR = 0.10        # m     path-scale floor
# Platform wheel-mounting radius sqrt(h²+l²) (h=0.235, l=0.150) — converts a yaw
# RATE to an equivalent edge speed (m/s) so the yaw channel is unit-consistent
# with Vx/Vy and normalised by ONE overall motion scale. Without this, a
# heading-held reference (omega_des≡0) makes any residual yaw look like a huge
# relative error.
L_CHAR = 0.279
T_EXCLUDE = 0.5         # s, startup-transient mask: the controller legitimately
                       # starts with large error while converging, so the first
                       # 0.5 s is excluded from every metric below.
# Per-instant utilisation level that counts as a friction-circle VIOLATION. Set to
# 1.10 (10% over the circle), NOT 1.0: inspection showed the util-driven rejects
# ride right AT the circle (max-wheel util in 1.0-1.05 for ~30% of the run, never
# above 1.05) — that is at-limit saturation (|F| ≈ μN), physically valid data the
# PINN should learn, not loss of control. 1.10 tolerates that boundary operation
# while still catching genuine gross slip (util sustained > 1.10), of which this
# dataset has none. Recovered 66 trajectories (incl. the spin_creep μ=0.3/χ=0
# matched-quad anchor) with no false keeps.
UTIL_VIOL_LEVEL = 1.10

THRESHOLDS = dict(
    track_inst=0.30,        # per-instant normalized error above this = "off-command at this instant"
    track_viol_frac=0.10,   # DURATION gate: off-command > 10% of the (post-startup) run => missed
    util_viol_frac=0.10,    # >10% of time over the friction circle => missed. Raised from 0.05
                            # after inspection: 0.05 rejected 13 coupled_vomega runs (util 0.057-0.096,
                            # track_viol_frac=0) that tracked velocity perfectly but briefly exceeded
                            # the friction circle — gross-slip data the PINN should learn, not a miss.
                            # 0.10 keeps those, still rejects the sustained violators (c135 0.13, c019 0.51).
    msat_rail_frac=0.50,    # >50% railed => (soft) saturation context
)


def _rms(a) -> float:
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a * a))) if a.size else float('nan')


def analyze_file(path: Path, th: Dict[str, float] = THRESHOLDS) -> Dict[str, Any]:
    name = Path(path).name
    p = cd.parse_arrow_filename(name)
    row: Dict[str, Any] = {'file': name}
    if p is None:
        row['error'] = 'filename did not match profile scheme'
        return row
    row.update({k: v for k, v in p.items() if k != 'is_posref'})
    base = (['time', 'Vx', 'Vy', 'psi_dot', 'Xo', 'Yo']
            + [f'util_{i}' for i in range(1, 5)] + [f'Msat_{i}' for i in range(1, 5)])
    refcols = (['xo_des', 'yo_des'] if p['is_posref'] else ['Vx_des', 'Vy_des', 'omega_des'])
    try:
        df = feather.read_feather(path, columns=base + refcols)
    except Exception as e:
        row['error'] = f'read failed: {e}'
        return row
    row['error'] = None

    # Startup-transient mask — exclude the first T_EXCLUDE s from every metric.
    t = df['time'].to_numpy(float)
    keep = (t - t[0]) > T_EXCLUDE
    if keep.sum() < 16:                       # ultra-short traj: don't mask it all away
        keep = np.ones(len(t), dtype=bool)

    # --- saturation / realized friction circle (both ref kinds), post-startup ---
    util = np.column_stack([df[f'util_{i}'].to_numpy(float) for i in range(1, 5)]).max(axis=1)[keep]
    msat = np.abs(np.column_stack([df[f'Msat_{i}'].to_numpy(float) for i in range(1, 5)])).max(axis=1)[keep]
    row['util_viol_frac'] = float(np.mean(util > UTIL_VIOL_LEVEL))   # 1.10: tolerate at-limit saturation
    row['util_budget_frac'] = float(np.mean(util > 0.8))
    row['msat_rail_frac'] = float(np.mean(msat > 0.95 * MAX_TORQUE))

    # --- tracking error: per-instant normalized error e(t), post-startup ---
    if p['is_posref']:
        ex = (df['Xo'].to_numpy(float) - df['xo_des'].to_numpy(float))[keep]
        ey = (df['Yo'].to_numpy(float) - df['yo_des'].to_numpy(float))[keep]
        scale = max(_rms(np.hypot(df['xo_des'].to_numpy(float)[keep],
                                  df['yo_des'].to_numpy(float)[keep])), POS_FLOOR)
        e = np.hypot(ex, ey) / scale
        row['track_kind'] = 'pos'
    else:
        # Unit-consistent generalized-velocity error: yaw rate -> edge speed via
        # L_CHAR, everything normalised by ONE desired-motion scale.
        dVx = (df['Vx'].to_numpy(float) - df['Vx_des'].to_numpy(float))[keep]
        dVy = (df['Vy'].to_numpy(float) - df['Vy_des'].to_numpy(float))[keep]
        dW = ((df['psi_dot'].to_numpy(float) - df['omega_des'].to_numpy(float)) * L_CHAR)[keep]
        desmag = np.sqrt(df['Vx_des'].to_numpy(float) ** 2 + df['Vy_des'].to_numpy(float) ** 2
                         + (df['omega_des'].to_numpy(float) * L_CHAR) ** 2)[keep]
        scale = max(_rms(desmag), V_FLOOR)
        e = np.sqrt(dVx ** 2 + dVy ** 2 + dW ** 2) / scale
        row['track_kind'] = 'vel'
        row['track_err_V'] = _rms(np.hypot(dVx, dVy)) / scale     # translational part
        row['track_err_w'] = _rms(dW) / scale                    # rotational (edge-speed) part

    # MAGNITUDE descriptor (RMS) and the new DURATION descriptor (fraction of the
    # post-startup run the instantaneous error exceeds track_inst).
    row['track_err'] = _rms(e)
    row['track_viol_frac'] = float(np.mean(e > th['track_inst']))

    row['track_flag'] = classify(row, th)
    return row


def classify(row: Dict[str, Any], th: Dict[str, float]) -> str:
    """Verdict driven by whether the trajectory was achieved, using the DURATION
    metrics: how long it was off-command (track_viol_frac) and over the friction
    circle (util_viol_frac). track_err (RMS magnitude) is reported but not the
    gate — a single large spike shouldn't condemn an otherwise-tracked run.
    Riding the 0.8 budget / railing the torque is EXPECTED for these aggressive
    excitations, so those stay context-only."""
    if row.get('error'):
        return 'ERROR'
    if row['track_viol_frac'] > th['track_viol_frac'] or row['util_viol_frac'] > th['util_viol_frac']:
        return 'missed'
    if row['track_viol_frac'] > 0.5 * th['track_viol_frac'] or row['util_viol_frac'] > 0.2 * th['util_viol_frac']:
        return 'marginal'
    return 'tracked'


def run_batch(data_dir: Path, out_csv: Path, profiles: Optional[set] = None,
              limit: Optional[int] = None, resume: bool = True,
              flush_every: int = 200, mu: Optional[float] = None) -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob('*.arrow'))
    paths = [p for p in paths if cd.parse_arrow_filename(p.name) is not None]
    if profiles:
        paths = [p for p in paths if cd.parse_arrow_filename(p.name)['profile'] in profiles]
    if mu is not None:   # restrict to a single friction coefficient (e.g. a new mu batch)
        paths = [p for p in paths if abs(cd.parse_arrow_filename(p.name)['mu'] - mu) < 1e-9]
    if limit:
        paths = paths[:limit]
    base_df = None
    done: set = set()
    if resume and out_csv.exists():
        base_df = pd.read_csv(out_csv)
        done = set(base_df['file'].tolist())
        print(f'[resume] {len(done)} rows already in {out_csv}')
    todo = [p for p in paths if p.name not in done]
    print(f'[batch] {len(todo)} of {len(paths)} files to process')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    def flush() -> pd.DataFrame:
        df = (pd.concat([base_df, pd.DataFrame(rows)], ignore_index=True)
              if base_df is not None else pd.DataFrame(rows))
        df.to_csv(out_csv, index=False)
        return df

    for k, p in enumerate(todo, 1):
        rows.append(analyze_file(p))
        if k % flush_every == 0:
            flush()
        if k % 50 == 0 or k == len(todo):
            print(f'  [{k}/{len(todo)}] {p.name}')
    df = flush()
    if 'track_flag' in df.columns:
        print('\n[track_flag counts]\n' + df['track_flag'].value_counts().to_string())
    print(f'\n[done] wrote {len(df)} rows -> {out_csv}')
    return df


def _cli():
    ap = argparse.ArgumentParser(description='Tracking / friction-circle gate.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('tracking_report.csv'))
    ap.add_argument('--profiles', type=str, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--mu', type=float, default=None, help='restrict to a single mu value')
    ap.add_argument('--no-resume', action='store_true')
    args = ap.parse_args()
    profiles = set(args.profiles.split(',')) if args.profiles else None
    run_batch(args.data_dir, args.out, profiles=profiles, limit=args.limit,
              resume=not args.no_resume, mu=args.mu)


if __name__ == '__main__':
    _cli()
