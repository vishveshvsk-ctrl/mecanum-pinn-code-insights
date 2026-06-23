#!/usr/bin/env python
# =============================================================================
# mu_identifiability.py — do the OBSERVABLE friction forces scale as A*mu + B?
#
# Companion to chi_identifiability.py, asking the model-free question for mu:
# the Coulomb/dissipative part of LuGre is F ~ mu*N*g(slip), gated by SLIP (not
# spin like chi); the sigma1*zdot bristle-damping term is the mu-INDEPENDENT
# affine B. So per (slip-regime, channel, co-directional-slip bin) we ask whether
# the slip-matched force scales multiplicatively with mu across {0.3,0.5,0.8}.
#
# Method (decided with the user — see mu_identifiability_handoff_v2.md):
#   * POOL all 3 mu (the x-axis of the A*mu+B fit). NO --mu filter. The 3 mu-sets
#     are utilization-scaled, NOT command-matched, so there are no matched quads
#     (unlike chi): "matching" is done STATISTICALLY by binning slip + covariate
#     control. Pointwise differencing across mu is invalid (trajectories decohere).
#   * NORMALIZE each force by the per-wheel static normal load N_i: y = F/N_i.
#     The per-wheel difference is MULTIPLICATIVE (mu*N_i*g), which an additive
#     wheel dummy cannot remove; dividing by N_i homogenizes all 4 wheels onto
#     one F/N = mu*g(slip) + sigma1*zdot/N relation and drops 4 regressors
#     (better-conditioned in sparse signed-slip bins). N_i comes from base.toml
#     geometry (CoM-offset => non-uniform: ~[79.6,105.1,69.6,95.1] N).
#   * BIN by SIGNED CO-DIRECTIONAL slip: Vpar = Vpx*cos d + Vpy*sin d (drive axis,
#     pairs with Fpar), Vperp = -Vpx*sin d + Vpy*cos d (free-roll axis, pairs with
#     Fperp); d = O-config [-pi/4,pi/4,pi/4,-pi/4] per wheel. Signed bins avoid the
#     sign-cancellation that |Vp| binning would cause and keep the drive/free-roll
#     anisotropy. (datastore rotates FORCE by d but NOT slip, so we rotate slip.)
#   * STRATIFY by pre-slip / transition / gross-slip on TOTAL per-wheel slip speed
#     |Vp|=hypot(Vpx,Vpy) vs the Stribeck velocity v_str=0.01 m/s. This IS the
#     mu-identifiability partition: in pre-slip F~sigma0*z is ~mu-independent
#     (mu unrecoverable); in gross-slip F->mu*N*g (mu readable). NOT util (=F/muN):
#     util has mu divided out, so conditioning on it makes F ~ mu trivially.
#   * Within each (stratum, channel, signed-bin, mu) regress y on slip controls,
#     evaluate at the COMMON pooled bin-mean slip s* so the 3 mu are slip-matched,
#     then fit F_bar(mu)=A*mu+B (n-weighted) across the 3 mu.
#   * ABLATION (requested): slip controls are tested in two forms holding |Vp|,
#     |Vp|^2 fixed:  LIN  = [1, Vcod,        |Vp|, |Vp|^2]
#                    QUAD = [1, Vcod, Vcod^2, |Vp|, |Vp|^2]
#     Both fits are reported so the reader sees whether the co-directional
#     quadratic changes the mu-verdict (mirrors chi's |Vp|^2 confound check).
#
# Measures (per cell, both ablation models):
#   * F_mu03/05/08 [N] = slip-matched cell mean force at s*, per mu (F/N fit * N_bar).
#   * affine_A, affine_B [N] of F_bar(mu)=A*mu+B (mu LINEAR; no mu^2 — unmotivated
#     and ill-conditioned on a 3-point grid, the chi lesson).
#   * mult_fraction(mu) = A*mu/(A*mu+B) at each ACTUAL mu (not a grid mean) —
#     the multiplicative share; -> 1 means purely mu-scaled, -> 0 means affine-B.
#   * mu_swing_abs [N] = |F_bar(0.8)-F_bar(0.3)| — the absolute identifiability
#     signal to compare against a force noise floor (units-trap-safe; NOT R^2/F,
#     useless at ~1e6 samples/bin). channel_rms [N], rel_swing = swing/rms.
#   * curvature_resid [N] = |F_bar(0.5) - line{0.3,0.8}(0.5)| — mu=0.5 held out;
#     low => truly linear-multiplicative.
#
# No OOM: never pools raw samples. Streams per-bin sufficient statistics
# (XtX 5x5, Xty, Syy, sum_N, n) one file at a time.
#
#   python mu_identifiability.py --data-dir ../data/Simulation_... --out mu_identifiability.csv \
#          --whitelist diagnostics_combined.csv
# =============================================================================
from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import chatter_diagnostics as cd   # parse_arrow_filename only

try:                                # Windows cp1252 can't encode mu/delta in prints
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- design knobs ---
CHANNELS = ('Fpar', 'Fperp')           # roller-frame; binned by their co-dir slip
DELTA = np.array([-np.pi / 4, np.pi / 4, np.pi / 4, -np.pi / 4])  # O-config, per wheel
COSD, SIND = np.cos(DELTA), np.sin(DELTA)

VCOD_MAX = 0.30                        # signed co-directional slip range [m/s]
VCOD_EDGES = np.linspace(-VCOD_MAX, VCOD_MAX, 17)   # 16 signed bins (values clipped)
N_BINS = len(VCOD_EDGES) - 1

V_STR = 0.01                           # translational Stribeck velocity [m/s] (base.toml)
GROSS_MULT = 3.0                       # gross-slip threshold = GROSS_MULT * v_str
# strata on TOTAL per-wheel slip |Vp| = hypot(Vpx,Vpy): pre-slip / transition / gross
SLIP_EDGES = (0.0, V_STR, GROSS_MULT * V_STR, np.inf)
SLIP_LABELS = ('preslip', 'transition', 'gross')
N_STRATA = len(SLIP_LABELS)

MU_GRID = (0.3, 0.5, 0.8)              # the swept mu values (x-axis of A*mu+B)
N_MU = len(MU_GRID)
MU_HELD = 0.5                          # held-out point for the curvature check

# regressor layout (superset): [1, Vcod, Vcod^2, |Vp|, |Vp|^2]
P = 5
COLS_QUAD = (0, 1, 2, 3, 4)            # full: includes the co-directional quadratic
COLS_LIN = (0, 1, 3, 4)                # ablation: drop Vcod^2, keep |Vp|,|Vp|^2
MIN_N = 400                            # min samples to fit a (cell, mu) regression
COND_MAX = 1e10                        # conditioning guard (chi's mu=0.3 blow-up)

_COLS = ([f'Vpx_{i}' for i in range(1, 5)] + [f'Vpy_{i}' for i in range(1, 5)]
         + [f'Fpar_{i}' for i in range(1, 5)] + [f'Fperp_{i}' for i in range(1, 5)])


def load_N_per_roller(base_toml: Path) -> np.ndarray:
    """Static per-wheel normal load, exactly as run_one.jl builds N_per_roller:
       N_total/4 * (1 +/- aX/h +/- aY/l) + m_wheel*g, signs per O-config wheel.
    Geometry is mu-invariant, so any of the mu-run base.toml files gives the same
    4-vector."""
    with open(base_toml, 'rb') as f:
        t = tomllib.load(f)
    g = 9.81
    geo = t['platform']['geometry']; mass = t['platform']['mass']
    com = t['platform']['com_offset']
    h, l = geo['h'], geo['l']
    m, m_wheel = mass['m'], mass['m_wheel']
    aX, aY = com['aX'], com['aY']
    N_total = m * g
    base = m_wheel * g
    return np.array([
        N_total / 4 * (1 + aX / h + aY / l) + base,
        N_total / 4 * (1 + aX / h - aY / l) + base,
        N_total / 4 * (1 - aX / h + aY / l) + base,
        N_total / 4 * (1 - aX / h - aY / l) + base,
    ])


def discover_files(data_dir: Path,
                   whitelist: Optional[set] = None) -> List[Dict]:
    """All Arrow files (pooled across mu), parsed for mu. No grouping — mu is the
    fit's x-axis, matched statistically by binning, not by command (the mu-sets
    are utilization-scaled, not command-matched). Whitelist (bare filenames) drops
    poorly-tracked runs file-by-file (no quad coupling like chi)."""
    out = []
    for p in sorted(Path(data_dir).glob('*.arrow')):
        m = cd.parse_arrow_filename(p.name)
        if m is None:
            continue
        if whitelist is not None and p.name not in whitelist:
            continue
        # snap mu to the nearest grid value (filenames carry 0.3/0.5/0.8)
        mu_i = int(np.argmin([abs(m['mu'] - g) for g in MU_GRID]))
        if abs(m['mu'] - MU_GRID[mu_i]) >= 1e-6:
            continue                    # an off-grid mu — skip (shouldn't happen)
        out.append(dict(path=p, mu_idx=mu_i, mu=MU_GRID[mu_i]))
    return out


class Accumulator:
    """Streaming per-(stratum, channel, signed-bin, mu) sufficient stats.
    Normalized regression stats (y=F/N_i) drive the structural fit; raw Syy and
    sum_N let us report the swing/rms back in Newtons."""
    def __init__(self):
        shp = (N_STRATA, len(CHANNELS), N_BINS, N_MU)
        self.XtX = np.zeros(shp + (P, P))      # normalized design
        self.Xty = np.zeros(shp + (P,))        # normalized X^T (F/N)
        self.Syy_raw = np.zeros(shp)           # sum F^2 (Newtons) -> channel_rms
        self.sum_N = np.zeros(shp)             # sum N_i -> N_bar for unit conversion
        self.n = np.zeros(shp, dtype=np.int64)

    def add_file(self, path: Path, mu_idx: int, N_roller: np.ndarray) -> None:
        df = feather.read_feather(path, columns=_COLS)
        for i in range(4):                                  # per wheel
            vpx = df[f'Vpx_{i+1}'].to_numpy(np.float64)
            vpy = df[f'Vpy_{i+1}'].to_numpy(np.float64)
            vcod_par = vpx * COSD[i] + vpy * SIND[i]        # drive axis  -> Fpar
            vcod_perp = -vpx * SIND[i] + vpy * COSD[i]      # free-roll   -> Fperp
            vpm = np.hypot(vpx, vpy)                         # total slip -> stratum
            strat = np.clip(np.digitize(vpm, SLIP_EDGES) - 1, 0, N_STRATA - 1)
            Ni = N_roller[i]
            Nn = len(vpm)
            for ci, ch in enumerate(CHANNELS):
                vcod = vcod_par if ch == 'Fpar' else vcod_perp
                vbin = np.clip(np.digitize(vcod, VCOD_EDGES) - 1, 0, N_BINS - 1)
                X = np.empty((Nn, P))
                X[:, 0] = 1.0
                X[:, 1] = vcod
                X[:, 2] = vcod * vcod
                X[:, 3] = vpm
                X[:, 4] = vpm * vpm
                f_raw = df[f'{ch}_{i+1}'].to_numpy(np.float64)
                y = f_raw / Ni                              # normalized force
                for s in range(N_STRATA):
                    for b in range(N_BINS):
                        msk = (strat == s) & (vbin == b)
                        if not msk.any():
                            continue
                        Xm = X[msk]; ym = y[msk]; fm = f_raw[msk]
                        self.XtX[s, ci, b, mu_idx] += Xm.T @ Xm
                        self.Xty[s, ci, b, mu_idx] += Xm.T @ ym
                        self.Syy_raw[s, ci, b, mu_idx] += float(fm @ fm)
                        self.sum_N[s, ci, b, mu_idx] += Ni * int(msk.sum())
                        self.n[s, ci, b, mu_idx] += int(msk.sum())


def _fit_mu(XtX: np.ndarray, Xty: np.ndarray, cols, xstar) -> Optional[float]:
    """Solve the slip-control regression for one mu and predict y(F/N) at the
    common reference slip xstar. Returns None on conditioning failure."""
    A = XtX[np.ix_(cols, cols)]
    b = Xty[list(cols)]
    if np.linalg.cond(A) > COND_MAX:
        return None
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return float(beta @ xstar)


def _weighted_affine(mus, fbar, w):
    """n-weighted LS line F_bar(mu)=A*mu+B over the (<=3) available mu points."""
    mus = np.asarray(mus); fbar = np.asarray(fbar); w = np.asarray(w, float)
    W = w.sum(); Sx = (w * mus).sum(); Sxx = (w * mus * mus).sum()
    Sy = (w * fbar).sum(); Sxy = (w * mus * fbar).sum()
    det = W * Sxx - Sx * Sx
    if abs(det) < 1e-30:
        return np.nan, np.nan
    A = (W * Sxy - Sx * Sy) / det
    B = (Sxx * Sy - Sx * Sxy) / det
    return A, B


def _fit_cell(acc: Accumulator, s: int, ci: int, b: int) -> Optional[Dict]:
    """Per-(stratum,channel,bin): per-mu slip-matched force, the A*mu+B fit, and
    the mult-fraction / swing / curvature measures — for BOTH ablation models."""
    n_mu = acc.n[s, ci, b]                       # (N_MU,)
    if not np.all(n_mu >= MIN_N):                # need all 3 mu for the affine fit
        return None
    n_tot = int(n_mu.sum())
    # pooled-over-mu reference slip vector x* = mean design row (the bin's center
    # of mass), so the 3 mu are evaluated at IDENTICAL slip.
    XtX_pool = acc.XtX[s, ci, b].sum(axis=0)     # (P,P); row 0 = column sums
    xbar = XtX_pool[0] / n_tot                    # [1, mean Vcod, mean Vcod^2, mean|Vp|, mean|Vp|^2]
    N_bar = float(acc.sum_N[s, ci, b].sum() / n_tot)
    channel_rms = float(np.sqrt(acc.Syy_raw[s, ci, b].sum() / n_tot))

    out = dict(stratum=SLIP_LABELS[s], channel=CHANNELS[ci],
               vcod_center=float(0.5 * (VCOD_EDGES[b] + VCOD_EDGES[b + 1])),
               vcod_mean=float(xbar[1]), n=n_tot,
               n_mu03=int(n_mu[0]), n_mu05=int(n_mu[1]), n_mu08=int(n_mu[2]),
               N_bar=N_bar, channel_rms=channel_rms)

    for tag, cols in (('q', COLS_QUAD), ('l', COLS_LIN)):
        xstar = xbar[list(cols)]
        fbar_norm = []
        ok = True
        for mi in range(N_MU):
            v = _fit_mu(acc.XtX[s, ci, b, mi], acc.Xty[s, ci, b, mi], cols, xstar)
            if v is None:
                ok = False
                break
            fbar_norm.append(v)
        if not ok:
            for k in (f'F_mu03_{tag}', f'F_mu05_{tag}', f'F_mu08_{tag}',
                      f'affine_A_{tag}', f'affine_B_{tag}',
                      f'mult_mu03_{tag}', f'mult_mu05_{tag}', f'mult_mu08_{tag}',
                      f'mu_swing_abs_{tag}', f'rel_swing_{tag}',
                      f'curvature_resid_{tag}'):
                out[k] = np.nan
            continue
        fbar_norm = np.array(fbar_norm)
        fbar_N = fbar_norm * N_bar                # back to Newtons
        A_n, B_n = _weighted_affine(MU_GRID, fbar_norm, n_mu)
        A_N, B_N = A_n * N_bar, B_n * N_bar
        denom = A_n * np.array(MU_GRID) + B_n
        mult = np.where(np.abs(denom) > 1e-12,
                        (A_n * np.array(MU_GRID)) / denom, np.nan)
        swing_N = abs(fbar_N[2] - fbar_N[0])      # |F(0.8) - F(0.3)|
        # held-out curvature: line through {0.3,0.8}, predict 0.5
        pred05 = fbar_N[0] + (fbar_N[2] - fbar_N[0]) * (MU_HELD - MU_GRID[0]) / (MU_GRID[2] - MU_GRID[0])
        curv = abs(fbar_N[1] - pred05)
        out.update({
            f'F_mu03_{tag}': fbar_N[0], f'F_mu05_{tag}': fbar_N[1], f'F_mu08_{tag}': fbar_N[2],
            f'affine_A_{tag}': A_N, f'affine_B_{tag}': B_N,
            f'mult_mu03_{tag}': mult[0], f'mult_mu05_{tag}': mult[1], f'mult_mu08_{tag}': mult[2],
            f'mu_swing_abs_{tag}': swing_N,
            f'rel_swing_{tag}': swing_N / channel_rms if channel_rms > 1e-12 else np.nan,
            f'curvature_resid_{tag}': curv,
        })

    # ablation deltas: does the co-directional quadratic move the verdict?
    out['d_swing_q_minus_l'] = out.get('mu_swing_abs_q', np.nan) - out.get('mu_swing_abs_l', np.nan)
    out['d_mult05_q_minus_l'] = out.get('mult_mu05_q', np.nan) - out.get('mult_mu05_l', np.nan)
    return out


def run(data_dir: Path, out_csv: Path, base_toml: Path,
        whitelist: Optional[set] = None, force: bool = False) -> pd.DataFrame:
    if out_csv.exists() and not force:
        print(f'[mu] {out_csv} exists -> resume/skip (use --force to recompute)')
        return pd.read_csv(out_csv)
    N_roller = load_N_per_roller(base_toml)
    print(f'[mu] N_per_roller = {np.round(N_roller, 2)} N  (from {base_toml})')
    files = discover_files(data_dir, whitelist=whitelist)
    tag = ' (whitelisted-only)' if whitelist is not None else ''
    by_mu = {g: sum(1 for f in files if f['mu'] == g) for g in MU_GRID}
    print(f'[mu] {len(files)} files{tag}  per-mu {by_mu}')

    acc = Accumulator()
    for k, f in enumerate(files, 1):
        acc.add_file(f['path'], f['mu_idx'], N_roller)
        if k % 100 == 0 or k == len(files):
            print(f'  [{k}/{len(files)}] {f["path"].name}')

    rows = []
    for s in range(N_STRATA):
        for ci in range(len(CHANNELS)):
            for b in range(N_BINS):
                r = _fit_cell(acc, s, ci, b)
                if r is not None:
                    rows.append(r)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _verdict(df)
    print(f'\n[done] wrote {len(df)} cells -> {out_csv}')
    return df


def _verdict(df: pd.DataFrame) -> None:
    """Headline per channel x stratum: gross-slip multiplicative fraction and the
    absolute mu-swing (Newtons), plus the LIN-vs-QUAD ablation gap."""
    if df.empty:
        print('[verdict] no fitted cells'); return
    print('\n[verdict] mu-multiplicativity by regime (QUAD model; n-weighted median over bins)')
    print(f"  {'stratum':<11}{'channel':<7}{'mult_frac(mu05)':>16}{'swing|dF| [N]':>15}"
          f"{'rel_swing':>11}{'ablation dSwing':>16}")
    for st in SLIP_LABELS:
        for ch in CHANNELS:
            d = df[(df['stratum'] == st) & (df['channel'] == ch)].dropna(subset=['mu_swing_abs_q'])
            if d.empty:
                continue
            w = d['n'].to_numpy(float)
            def wmed(col):
                v = d[col].to_numpy(float); m = np.isfinite(v)
                if not m.any():
                    return np.nan
                order = np.argsort(v[m]); vv = v[m][order]; ww = w[m][order]
                c = np.cumsum(ww)
                return float(vv[np.searchsorted(c, 0.5 * c[-1])])
            print(f"  {st:<11}{ch:<7}{wmed('mult_mu05_q'):>16.3f}{wmed('mu_swing_abs_q'):>15.4f}"
                  f"{wmed('rel_swing_q'):>11.3f}{wmed('d_swing_q_minus_l'):>16.4f}")
    print("  mult_frac->1: force is mu-scaled (mu identifiable). ->0: affine-B dominated (mu blind).")
    print("  Compare swing|dF| [N] to the force noise floor for the deployable verdict.")
    print("  ablation dSwing = swing(QUAD)-swing(LIN): ~0 => co-directional Vcod^2 is inert.")


def _cli():
    ap = argparse.ArgumentParser(description='mu-multiplicativity of observable friction forces.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('mu_identifiability.csv'))
    ap.add_argument('--base-toml', type=Path,
                    default=Path('trajectory_files_run_0p5_main/base.toml'),
                    help='source of platform geometry for N_per_roller (mu-invariant)')
    ap.add_argument('--whitelist', type=Path, default=None,
                    help='combined CSV: keep only files whose combined_reco is not reject')
    ap.add_argument('--force', action='store_true', help='recompute even if --out exists')
    args = ap.parse_args()
    wl = None
    if args.whitelist is not None:
        wdf = pd.read_csv(args.whitelist)
        wl = set(wdf.loc[~wdf['combined_reco'].str.startswith('reject'), 'file'])
        print(f'[mu] whitelist: {len(wl)} kept files from {args.whitelist}')
    run(args.data_dir, args.out, args.base_toml, whitelist=wl, force=args.force)


if __name__ == '__main__':
    _cli()
