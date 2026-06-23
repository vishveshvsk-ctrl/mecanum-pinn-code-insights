#!/usr/bin/env python
# =============================================================================
# sampling_sensitivity.py — does downsampling 2000 Hz -> 1000/500 Hz matter?
#
# Training the PINN on the native 2000 Hz grid is costly (seq_len, memory,
# throughput); lower rates are cheaper but risk losing or ALIASING physical
# content. This study quantifies the difference per trajectory so the rate
# choice is evidence-based, not a guess.
#
# Two downsampling MODES are compared (they have opposite failure modes):
#   * anti-aliased decimation  -> low-passes first; nothing folds, but genuine
#                                 content above the new Nyquist is LOST.
#   * naive subsampling (x[::k]) -> everything above the new Nyquist FOLDS back
#                                 into the band, manufacturing fake structure.
#
# Significance is judged by (decided with the user):
#   1. force reconstruction error  (resample -> re-interpolate to 2000 Hz vs
#      native, per family, normalised) — compare against the brief's
#      interpolation floor (~2.7e-4 for Mz).
#   2. chatter-verdict flips        (does the chatter screen's verdict change
#      across rate/mode? naive subsampling should manufacture 'hash' via
#      aliasing; anti-aliased should not).
# Energy-above-Nyquist is also reported (cheap, physics-grounded context).
#
# Reuses chatter_diagnostics for the verdict so the two screens stay consistent.
# No new real data needed beyond the production sweep output; run on it once it
# exists:  python sampling_sensitivity.py --data-dir ../data/Simulation_... --out samp.csv
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

import chatter_diagnostics as cd

# Native grid (base.toml) and the candidate training rates to test against it.
NATIVE_RATE_HZ = 2000.0
DEFAULT_RATES = (1000.0, 500.0)
FAMILIES = ('Fpar', 'Fperp', 'Mz')          # force/torque channels the PINN regresses
RIDGE_GRID_HZ = 250.0                        # keep the internal ridge grid ~constant across rates


# =============================================================================
# Resampling
# =============================================================================
def _resample(t: np.ndarray, W: Dict[str, np.ndarray], omega: np.ndarray,
              theta: np.ndarray, k: int, mode: str):
    """Downsample by integer factor k. mode = 'antialias' | 'naive'.
    theta is always index-sliced (it accumulates; filtering it is meaningless)."""
    if mode == 'antialias':
        Wr = {s: cd._decimate_cols(W[s], k) for s in W}
        om = cd._decimate_cols(omega, k)
    elif mode == 'naive':
        Wr = {s: W[s][::k] for s in W}
        om = omega[::k]
    else:
        raise ValueError(mode)
    th = theta[::k]
    tr = t[::k]
    L = min(len(tr), len(om), len(th), *(len(Wr[s]) for s in Wr))
    return tr[:L], {s: Wr[s][:L] for s in Wr}, om[:L], th[:L]


def _recon_error(F_native: np.ndarray, F_r: np.ndarray, k: int) -> float:
    """Median-over-wheels normalised RMS error of BAND-LIMITED reconstruction of
    the resampled force back onto the native grid (polyphase upsample by k).

    Band-limited (not linear) reconstruction is deliberate: it isolates genuine
    information change — irrecoverable HF loss (anti-aliased) and phantom in-band
    aliasing (naive) — instead of conflating it with an interpolator's crudeness.
    For content below the new Nyquist it recovers ~perfectly, so a faithful rate
    scores ~0; only real loss/aliasing shows up."""
    N = F_native.shape[0]
    errs = []
    for i in range(4):
        rec = resample_poly(F_r[:, i], k, 1)
        L = min(N, len(rec))
        a, b = F_native[:L, i], rec[:L]
        denom = np.sqrt(np.mean(a ** 2)) + 1e-12
        errs.append(np.sqrt(np.mean((b - a) ** 2)) / denom)
    return float(np.median(errs))


def _energy_above(F: np.ndarray, fs: float, fcut: float, cfg: cd.ChatterConfig) -> float:
    """Median-over-wheels fraction of off-DC PSD energy above fcut in the NATIVE
    signal — what is 'at risk' at a Nyquist of fcut (mode-independent)."""
    out = []
    for i in range(4):
        f, P = cd._welch_psd(F[:, i], fs, cfg.welch_nperseg)
        tot = P[1:].sum()
        out.append(float(P[f > fcut].sum() / tot) if tot > 1e-30 else np.nan)
    return float(np.nanmedian(out)) if np.any(np.isfinite(out)) else np.nan


# =============================================================================
# Per-trajectory analysis
# =============================================================================
def analyze_file(path: Path, cfg: cd.ChatterConfig,
                 rates: Tuple[float, ...] = DEFAULT_RATES) -> Dict[str, Any]:
    name = Path(path).name
    parsed = cd.parse_arrow_filename(name)
    row: Dict[str, Any] = {'file': name}
    if parsed is None:
        row['error'] = 'filename did not match profile scheme'
        return row
    row.update({k: v for k, v in parsed.items() if k != 'is_posref'})
    try:
        t, W, omega, theta = cd.load_columns(path)
    except Exception as e:
        row['error'] = f'read failed: {e}'
        return row
    if t.size < 256:
        row['error'] = f'too short ({t.size} samples)'
        return row

    fs = 1.0 / float(np.median(np.diff(t)))
    mu, chi = parsed['mu'], parsed['chi']
    row['fs_native'] = fs
    row['error'] = None

    # Native verdict — the baseline every flip is measured against.
    base = cd.diagnose_columns(t, W, omega, theta, mu, chi, cfg)
    row['verdict_native'] = base.get('verdict')

    for r in rates:
        k = int(round(fs / r))
        if k < 2:
            continue
        nyq = r / 2.0
        row[f'k_{int(r)}'] = k
        # energy at risk above this Nyquist (native signal, per family)
        for fam in FAMILIES:
            row[f'engabove_{int(r)}_{fam}'] = _energy_above(W[fam], fs, nyq, cfg)
        # keep the internal ridge grid near-constant so the verdict kernels are
        # comparable across rates instead of over-decimating a low-rate input
        cfg_r = dataclasses.replace(cfg, decimate_factor=max(1, int(round(r / RIDGE_GRID_HZ))))
        for mode in ('antialias', 'naive'):
            tr, Wr, omr, thr = _resample(t, W, omega, theta, k, mode)
            for fam in FAMILIES:
                row[f'recon_{mode}_{int(r)}_{fam}'] = _recon_error(W[fam], Wr[fam], k)
            v = cd.diagnose_columns(tr, Wr, omr, thr, mu, chi, cfg_r)
            row[f'verdict_{mode}_{int(r)}'] = v.get('verdict')
            row[f'flip_{mode}_{int(r)}'] = (v.get('verdict') != row['verdict_native'])
    return row


# =============================================================================
# Batch runner + CLI (streaming + resume, mirrors chatter_diagnostics)
# =============================================================================
def run_batch(data_dir: Path, out_csv: Path, cfg: cd.ChatterConfig,
              rates: Tuple[float, ...] = DEFAULT_RATES,
              profiles: Optional[set] = None, limit: Optional[int] = None,
              resume: bool = True, flush_every: int = 100,
              mu: Optional[float] = None) -> pd.DataFrame:
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
        df.to_csv(out_csv, index=False)        # crash-safe checkpoint (Modern Standby)
        return df

    for j, p in enumerate(todo, 1):
        rows.append(analyze_file(p, cfg, rates))
        if j % flush_every == 0:
            flush()
        if j % 50 == 0 or j == len(todo):
            print(f'  [{j}/{len(todo)}] {p.name}')
    df = flush()
    summarize(df, rates)
    print(f'\n[done] wrote {len(df)} rows -> {out_csv}')
    return df


def summarize(df: pd.DataFrame, rates: Tuple[float, ...] = DEFAULT_RATES) -> None:
    """Print the headline significance verdict per rate/mode."""
    ok = df[df['error'].isna()] if 'error' in df.columns else df
    if not len(ok):
        print('[summary] no successful rows'); return
    print('\n[summary] median force reconstruction error (lower = more faithful)')
    print(f"  {'rate/mode':<22} {'Fpar':>9} {'Fperp':>9} {'Mz':>9}   {'flip%':>6}")
    for r in rates:
        for mode in ('antialias', 'naive'):
            cols = [f'recon_{mode}_{int(r)}_{fam}' for fam in FAMILIES]
            if not all(c in ok.columns for c in cols):
                continue
            med = [ok[c].median() for c in cols]
            flip = 100 * ok[f'flip_{mode}_{int(r)}'].mean() if f'flip_{mode}_{int(r)}' in ok else float('nan')
            print(f"  {int(r):>4} Hz {mode:<13} {med[0]:>9.4f} {med[1]:>9.4f} {med[2]:>9.4f}   {flip:>5.1f}%")


def _cli():
    ap = argparse.ArgumentParser(description='Sampling-rate sensitivity for chatter diagnostics.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('sampling_sensitivity.csv'))
    ap.add_argument('--rates', type=str, default='1000,500', help='comma list of target Hz')
    ap.add_argument('--profiles', type=str, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--mu', type=float, default=None, help='restrict to a single mu value')
    ap.add_argument('--no-resume', action='store_true')
    args = ap.parse_args()
    rates = tuple(float(x) for x in args.rates.split(','))
    profiles = set(args.profiles.split(',')) if args.profiles else None
    run_batch(args.data_dir, args.out, cd.ChatterConfig(), rates=rates,
              profiles=profiles, limit=args.limit, resume=not args.no_resume,
              mu=args.mu)


if __name__ == '__main__':
    _cli()
