#!/usr/bin/env python
# =============================================================================
# force_mu_chi_regression.py — global (UNGATED) force law  F = A*mu + B + C*chi
#
# A single pooled regression per channel over ALL regimes (no slip/spin gating),
# answering: across the whole operating distribution, what is the effective
# sensitivity of the observable forces to mu and chi simultaneously?
#
# IMPORTANT framing (see TRAJ_DIAGNOSTICRESULTS §2/§8 for the gated truth):
#   * A is NOT "mu-multiplicativity": it is the slip-AVERAGED dF/dmu (=N*<g(slip)>);
#     §8 showed the real A runs 0 (pre-slip) -> N*g_max (gross slip).
#   * C is NOT "the chi-coupling slope": the coupling is c_t=(8/3pi)|wz|chi, so C is
#     the |wz|-AVERAGED dF/dchi; §2 showed it rises with spin.
#   * B is a single constant standing in for the large slip-varying structural force.
# So these are operating-distribution-averaged effective coefficients, by design.
#
# De-confounding (decided with the user): the mu-batches are utilization-scaled,
# so mu co-varies with the slip distribution; B (structural force) varies sharply
# with slip. We therefore KEEP slip COVARIATES (Vcod, Vcod^2, |Vp|, |Vp|^2) as
# controls (this is NOT gating — no binning) so A,C are the true friction
# sensitivities at matched slip, not the mu-scaling artifact. Forces are normalized
# by per-wheel N_i (pool wheels); chi is scaled to O(1) for conditioning.
#
# Regressor layout (per channel, y = F/N_i):
#   [1, mu, chi/CHI_SCALE, Vcod, Vcod^2, |Vp|, |Vp|^2]
#    B   A   C'             '------ slip controls ------'
#   A_N = A*N_bar ; B_N = B*N_bar ; C_N = (C'/CHI_SCALE)*N_bar   (back to Newtons)
#
# Run on TWO datasets (decided with the user):
#   * 'all'  : every whitelisted file (chi=0.005-dominant; realistic mix).
#   * 'quad' : only the matched chi-quad files (chi in {0,.002,.005,.008} balanced),
#              for a clean C; pooled across mu.
#
#   python force_mu_chi_regression.py --data-dir ../data/Simulation_... \
#          --out force_mu_chi_regression.csv --whitelist diagnostics_combined.csv
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import chatter_diagnostics as cd
from chi_identifiability import discover_quads
from mu_identifiability import load_N_per_roller, COSD, SIND

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CHANNELS = ('Fpar', 'Fperp')
CHI_SCALE = 0.005
MU_GRID = (0.3, 0.5, 0.8)
CHI_GRID = (0.0, 0.008)        # for reporting the chi-swing |ΔF| over the grid
# regressors: [1, mu, chi_s, Vcod, Vcod^2, |Vp|, |Vp|^2]
P = 7
MU_COL, CHI_COL = 1, 2
COND_MAX = 1e10

_COLS = ([f'Fpar_{i}' for i in range(1, 5)] + [f'Fperp_{i}' for i in range(1, 5)]
         + [f'Vpx_{i}' for i in range(1, 5)] + [f'Vpy_{i}' for i in range(1, 5)])


class Accumulator:
    """One global cell per (dataset, channel): pooled sufficient stats."""
    def __init__(self, datasets):
        self.ds = list(datasets)
        shp = (len(self.ds), len(CHANNELS))
        self.XtX = np.zeros(shp + (P, P))
        self.Xty = np.zeros(shp + (P,))
        self.Syy = np.zeros(shp)        # sum (F/N)^2
        self.SF2 = np.zeros(shp)        # sum F^2 (Newtons) -> channel_rms
        self.SN = np.zeros(shp)         # sum N_i -> N_bar
        self.n = np.zeros(shp, dtype=np.int64)

    def add_file(self, path: Path, mu: float, chi: float, N_roller: np.ndarray,
                 ds_idx: List[int]) -> None:
        df = feather.read_feather(path, columns=_COLS)
        chis = chi / CHI_SCALE
        for i in range(4):
            vpx = df[f'Vpx_{i+1}'].to_numpy(np.float64)
            vpy = df[f'Vpy_{i+1}'].to_numpy(np.float64)
            vpar = vpx * COSD[i] + vpy * SIND[i]
            vperp = -vpx * SIND[i] + vpy * COSD[i]
            vpm = np.hypot(vpx, vpy)
            Ni = N_roller[i]
            Nn = len(vpm)
            for ci, ch in enumerate(CHANNELS):
                vcod = vpar if ch == 'Fpar' else vperp
                X = np.empty((Nn, P))
                X[:, 0] = 1.0; X[:, 1] = mu; X[:, 2] = chis
                X[:, 3] = vcod; X[:, 4] = vcod * vcod
                X[:, 5] = vpm;  X[:, 6] = vpm * vpm
                f_raw = df[f'{ch}_{i+1}'].to_numpy(np.float64)
                y = f_raw / Ni
                XtX = X.T @ X; Xty = X.T @ y
                syy = float(y @ y); sf2 = float(f_raw @ f_raw)
                for di in ds_idx:
                    self.XtX[di, ci] += XtX
                    self.Xty[di, ci] += Xty
                    self.Syy[di, ci] += syy
                    self.SF2[di, ci] += sf2
                    self.SN[di, ci] += Ni * Nn
                    self.n[di, ci] += Nn


def _fit(XtX, Xty, Syy, n):
    """Full fit + partial-R^2 for the {mu} and {chi} blocks. Returns coeffs & stats."""
    if np.linalg.cond(XtX) > COND_MAX:
        return None
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return None
    ybar = Xty[0] / n
    tss = Syy - n * ybar * ybar
    rss = Syy - float(beta @ Xty)
    out = {'beta': beta, 'R2': 1 - rss / tss if tss > 0 else np.nan}
    for name, col in (('mu', MU_COL), ('chi', CHI_COL)):
        keep = [j for j in range(P) if j != col]
        Ar = XtX[np.ix_(keep, keep)]; br = Xty[keep]
        try:
            br_beta = np.linalg.solve(Ar, br)
            rss_r = Syy - float(br_beta @ br)
            out[f'partial_R2_{name}'] = (rss_r - rss) / rss_r if rss_r > 0 else np.nan
        except np.linalg.LinAlgError:
            out[f'partial_R2_{name}'] = np.nan
    return out


def run(data_dir: Path, out_csv: Path, whitelist: Optional[set] = None,
        base_toml: Path = Path('trajectory_files_run_0p5_main/base.toml'),
        force: bool = False) -> pd.DataFrame:
    if out_csv.exists() and not force:
        print(f'[mu_chi] {out_csv} exists -> skip (use --force)')
        return pd.read_csv(out_csv)
    N_roller = load_N_per_roller(base_toml)
    print(f'[mu_chi] N_per_roller = {np.round(N_roller, 2)} N')

    # quad subset = files in any matched chi-quad (>=3 chi, all whitelisted), mu pooled
    quads = discover_quads(data_dir, min_chi=3, mu=None, whitelist=whitelist)
    quad_names = {p.name for g in quads for p in g.values()}
    print(f'[mu_chi] {len(quad_names)} files in matched chi-quads (balanced chi)')

    datasets = ('all', 'quad')
    acc = Accumulator(datasets)
    files = []
    for p in sorted(Path(data_dir).glob('*.arrow')):
        m = cd.parse_arrow_filename(p.name)
        if m is None:
            continue
        if whitelist is not None and p.name not in whitelist:
            continue
        files.append((p, m['mu'], m['chi']))
    print(f'[mu_chi] {len(files)} whitelisted files')
    for k, (p, mu, chi) in enumerate(files, 1):
        ds_idx = [0] + ([1] if p.name in quad_names else [])
        acc.add_file(p, mu, chi, N_roller, ds_idx)
        if k % 300 == 0 or k == len(files):
            print(f'  [{k}/{len(files)}] {p.name}')

    rows = []
    for di, ds in enumerate(datasets):
        for ci, ch in enumerate(CHANNELS):
            n = int(acc.n[di, ci])
            if n < P + 50:
                continue
            fit = _fit(acc.XtX[di, ci], acc.Xty[di, ci], float(acc.Syy[di, ci]), n)
            Nbar = float(acc.SN[di, ci] / n)
            rms = float(np.sqrt(acc.SF2[di, ci] / n))
            if fit is None:
                rows.append(dict(dataset=ds, channel=ch, n=n, channel_rms=rms, note='ill-conditioned'))
                continue
            b = fit['beta']
            A_N = b[MU_COL] * Nbar
            B_N = b[0] * Nbar
            C_N = (b[CHI_COL] / CHI_SCALE) * Nbar
            rows.append(dict(
                dataset=ds, channel=ch, n=n, N_bar=Nbar, channel_rms=rms,
                A_dF_dmu=A_N, B_intercept=B_N, C_dF_dchi=C_N,
                mu_swing_abs=abs(A_N) * (MU_GRID[-1] - MU_GRID[0]),     # |ΔF| over mu 0.3->0.8
                chi_swing_abs=abs(C_N) * (CHI_GRID[-1] - CHI_GRID[0]),  # |ΔF| over chi 0->0.008
                R2=fit['R2'], partial_R2_mu=fit['partial_R2_mu'],
                partial_R2_chi=fit['partial_R2_chi'],
            ))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _verdict(df)
    print(f'\n[done] wrote {len(df)} rows -> {out_csv}')
    return df


def _verdict(df: pd.DataFrame) -> None:
    if df.empty:
        print('[verdict] no fits'); return
    print('\n[verdict] global ungated force law  F = A*mu + B + C*chi  (slip-controlled, N)')
    print(f"  {'dataset':<6}{'chan':<7}{'A=dF/dmu':>10}{'B':>9}{'C=dF/dchi':>11}"
          f"{'|dF|_mu':>9}{'|dF|_chi':>9}{'R2':>6}{'pR2_mu':>8}{'pR2_chi':>9}")
    for _, r in df.iterrows():
        if 'A_dF_dmu' not in r or pd.isna(r.get('A_dF_dmu')):
            print(f"  {r['dataset']:<6}{r['channel']:<7}  (ill-conditioned)"); continue
        print(f"  {r['dataset']:<6}{r['channel']:<7}{r['A_dF_dmu']:>10.2f}{r['B_intercept']:>9.2f}"
              f"{r['C_dF_dchi']:>11.1f}{r['mu_swing_abs']:>9.2f}{r['chi_swing_abs']:>9.3f}"
              f"{r['R2']:>6.3f}{r['partial_R2_mu']:>8.3f}{r['partial_R2_chi']:>9.4f}")
    print("  A,B,C,|dF| in Newtons. |dF|_mu = swing over mu 0.3->0.8 ; |dF|_chi over chi 0->0.008.")
    print("  Slip-AVERAGED effective coeffs (NOT regime-resolved) — compare to gated §8/§2.")


def _cli():
    ap = argparse.ArgumentParser(description='Global ungated F=A*mu+B+C*chi force regression.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('force_mu_chi_regression.csv'))
    ap.add_argument('--base-toml', type=Path, default=Path('trajectory_files_run_0p5_main/base.toml'))
    ap.add_argument('--whitelist', type=Path, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    wl = None
    if args.whitelist is not None:
        wdf = pd.read_csv(args.whitelist)
        wl = set(wdf.loc[~wdf['combined_reco'].str.startswith('reject'), 'file'])
        print(f'[mu_chi] whitelist: {len(wl)} kept files')
    run(args.data_dir, args.out, whitelist=wl, base_toml=args.base_toml, force=args.force)


if __name__ == '__main__':
    _cli()
