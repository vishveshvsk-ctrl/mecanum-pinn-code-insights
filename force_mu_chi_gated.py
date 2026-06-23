#!/usr/bin/env python
# =============================================================================
# force_mu_chi_gated.py — GATED multiplicative force law  F = mu*(A + C*chi) + B
#
# The correctly-specified, regime-resolved companion to force_mu_chi_regression.py
# (which showed the GLOBAL ADDITIVE form sees ~0 mu/chi — sign-cancellation +
# averaging + additive mis-spec). Here:
#   * MULTIPLICATIVE chi:  F = mu*(A + C*chi) + B = A*mu + B + C*(mu*chi)
#     (chi modulates the mu-scaled Coulomb shape; B = mu/chi-independent affine).
#   * GATED: bin by signed co-directional slip, stratify by |Vp| (pre/trans/gross)
#     — fixes the sign-cancellation & regime-averaging that nulled the global fit.
#
# TWO forms of the chi term (decided with the user):
#   (a)  C * (mu*chi)            -- NO explicit |wz|. Tests whether a single per-cell
#                                   C can ABSORB the (unmeasurable) spin-gating.
#   (b)  C * (mu*|wz|*chi)       -- explicit |wz| (oracle): the true structural
#                                   coupling c_t=(8/3pi)|wz|chi made into a constant.
#
# TWO |wz| schemes, to test if the black-box SSM must RESOLVE spin:
#   * unbinned-|wz| : one C per slip-cell (|wz| varies inside)  -> can C absorb it?
#   * binned-|wz|   : C per |wz|-bin (spin resolved)            -> does resolving help?
# If partial_R2_chi:  (a)-binned ~ (b) >> (a)-unbinned  => SSM must reconstruct spin.
#                     (a)-unbinned ~ (a)-binned          => a constant C absorbs it.
#
# TWO datasets: quad (balanced chi) and full (chi=0.005-dominant) -- imbalance effect.
# Normalized y=F/N_i; chi scaled; reduced [1,mu,slip] always gives A,B (so full cells
# without chi-variation still report A,B, with C=NaN -- that IS the imbalance signal).
#
#   python force_mu_chi_gated.py --data-dir ../data/Simulation_... \
#          --out force_mu_chi_gated.csv --whitelist diagnostics_combined.csv
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
NCH = len(CHANNELS)
CHI_SCALE = 0.005
MU_GRID = (0.3, 0.8)                    # for the chi/mu-swing endpoints
DCHI = 0.008                           # chi swing 0 -> 0.008

# gating
V_STR, GROSS = 0.01, 0.03
SLIP_EDGES = (0.0, V_STR, GROSS, np.inf)
SLIP_LABELS = ('pre', 'trans', 'gross')
NSTRAT = len(SLIP_LABELS)
VCOD_EDGES = np.linspace(-0.30, 0.30, 17)      # 16 signed co-dir slip bins
NCODIR = 16
WZ_EDGES = np.linspace(0.0, 20.0, 9)           # 8 |wz| bins (for binned scheme)
NWZ = 8
NCELL = NSTRAT * NCODIR * NWZ

# regressor superset: [1, mu, Vcod, Vcod^2, |Vp|, |Vp|^2, mu*chis, mu*|wz|*chis]
P = 8
RED = [0, 1, 2, 3, 4, 5]               # reduced model (A,B,slip) -> always
CHI_A, CHI_B = 6, 7                     # the two chi-term columns
MIN_N = 400
COND_MAX = 1e10

_COLS = ([f'Fpar_{i}' for i in range(1, 5)] + [f'Fperp_{i}' for i in range(1, 5)]
         + [f'Vpx_{i}' for i in range(1, 5)] + [f'Vpy_{i}' for i in range(1, 5)]
         + [f'wz_{i}' for i in range(1, 5)])
_IU = np.triu_indices(P)               # upper-triangle for symmetric XtX accumulation


class Accumulator:
    """Finest-grain sufficient stats per (channel, strat, codir-bin, wz-bin).
    Unbinned-|wz| is recovered by summing over the wz axis."""
    def __init__(self):
        self.XtX = np.zeros((NCH, NCELL, P, P))
        self.Xty = np.zeros((NCH, NCELL, P))
        self.Syy = np.zeros((NCH, NCELL))
        self.SF2 = np.zeros((NCH, NCELL))
        self.SN = np.zeros((NCH, NCELL))
        self.Swz = np.zeros((NCH, NCELL))      # sum |wz| -> cell-mean spin
        self.n = np.zeros((NCH, NCELL), dtype=np.int64)

    def add_file(self, path: Path, mu: float, chi: float, N_roller: np.ndarray) -> None:
        df = feather.read_feather(path, columns=_COLS)
        chis = chi / CHI_SCALE
        for i in range(4):
            vpx = df[f'Vpx_{i+1}'].to_numpy(np.float64)
            vpy = df[f'Vpy_{i+1}'].to_numpy(np.float64)
            awz = np.abs(df[f'wz_{i+1}'].to_numpy(np.float64))
            vpar = vpx * COSD[i] + vpy * SIND[i]
            vperp = -vpx * SIND[i] + vpy * COSD[i]
            vpm = np.hypot(vpx, vpy)
            strat = np.clip(np.digitize(vpm, SLIP_EDGES) - 1, 0, NSTRAT - 1)
            wzbin = np.clip(np.digitize(awz, WZ_EDGES) - 1, 0, NWZ - 1)
            Ni = N_roller[i]
            Nn = len(vpm)
            for ci, ch in enumerate(CHANNELS):
                vcod = vpar if ch == 'Fpar' else vperp
                cbin = np.clip(np.digitize(vcod, VCOD_EDGES) - 1, 0, NCODIR - 1)
                flat = (strat * NCODIR + cbin) * NWZ + wzbin
                X = np.empty((Nn, P))
                X[:, 0] = 1.0; X[:, 1] = mu
                X[:, 2] = vcod; X[:, 3] = vcod * vcod
                X[:, 4] = vpm;  X[:, 5] = vpm * vpm
                X[:, 6] = mu * chis
                X[:, 7] = mu * awz * chis
                f_raw = df[f'{ch}_{i+1}'].to_numpy(np.float64)
                y = f_raw / Ni
                for a, b in zip(*_IU):
                    self.XtX[ci, :, a, b] += np.bincount(flat, X[:, a] * X[:, b], NCELL)
                for a in range(P):
                    self.Xty[ci, :, a] += np.bincount(flat, X[:, a] * y, NCELL)
                self.Syy[ci] += np.bincount(flat, y * y, NCELL)
                self.SF2[ci] += np.bincount(flat, f_raw * f_raw, NCELL)
                self.SN[ci] += np.bincount(flat, None, NCELL) * Ni
                self.Swz[ci] += np.bincount(flat, awz, NCELL)
                self.n[ci] += np.bincount(flat, None, NCELL).astype(np.int64)

    def symmetrize(self):
        i, j = _IU
        self.XtX[:, :, j, i] = self.XtX[:, :, i, j]


def _fit(XtX, Xty, Syy, n) -> Optional[Dict]:
    """Reduced [1,mu,slip] -> A,B (always). Then add each chi col -> C, partial_R2_chi."""
    Ar = XtX[np.ix_(RED, RED)]; br = Xty[RED]
    if n < MIN_N or np.linalg.cond(Ar) > COND_MAX:
        return None
    try:
        beta_r = np.linalg.solve(Ar, br)
    except np.linalg.LinAlgError:
        return None
    rss_red = Syy - float(beta_r @ br)
    ybar = Xty[0] / n
    tss = Syy - n * ybar * ybar
    out = {'A': float(beta_r[1]), 'B': float(beta_r[0]), 'rss_red': rss_red,
           'R2_red': float(1 - rss_red / tss) if tss > 0 else np.nan}    # form adequacy (no chi)
    for tag, col in (('a', CHI_A), ('b', CHI_B)):
        cols = RED + [col]
        Af = XtX[np.ix_(cols, cols)]; bf = Xty[cols]
        if np.linalg.cond(Af) > COND_MAX:
            out[f'C_{tag}'] = np.nan; out[f'pR2_{tag}'] = np.nan; out[f'R2_{tag}'] = np.nan; continue
        try:
            beta = np.linalg.solve(Af, bf)
        except np.linalg.LinAlgError:
            out[f'C_{tag}'] = np.nan; out[f'pR2_{tag}'] = np.nan; out[f'R2_{tag}'] = np.nan; continue
        rss_full = Syy - float(beta @ bf)
        if rss_full > rss_red or rss_full <= 0 or rss_red <= 0:
            out[f'C_{tag}'] = np.nan; out[f'pR2_{tag}'] = np.nan; out[f'R2_{tag}'] = np.nan; continue
        out[f'C_{tag}'] = float(beta[-1])
        out[f'pR2_{tag}'] = (rss_red - rss_full) / rss_red
        out[f'R2_{tag}'] = float(1 - rss_full / tss) if tss > 0 else np.nan   # full-form adequacy
    return out


def _rows_for_scheme(acc: Accumulator, dataset: str, scheme: str) -> List[Dict]:
    """scheme='unbinned' marginalizes the wz axis; 'binned' keeps it."""
    rows = []
    XtX = acc.XtX.reshape(NCH, NSTRAT, NCODIR, NWZ, P, P)
    Xty = acc.Xty.reshape(NCH, NSTRAT, NCODIR, NWZ, P)
    Syy = acc.Syy.reshape(NCH, NSTRAT, NCODIR, NWZ)
    SF2 = acc.SF2.reshape(NCH, NSTRAT, NCODIR, NWZ)
    SN = acc.SN.reshape(NCH, NSTRAT, NCODIR, NWZ)
    Swz = acc.Swz.reshape(NCH, NSTRAT, NCODIR, NWZ)
    nn = acc.n.reshape(NCH, NSTRAT, NCODIR, NWZ)
    wz_ctr = 0.5 * (WZ_EDGES[:-1] + WZ_EDGES[1:])
    cod_ctr = 0.5 * (VCOD_EDGES[:-1] + VCOD_EDGES[1:])
    wz_iter = [None] if scheme == 'unbinned' else range(NWZ)
    for ci in range(NCH):
        for s in range(NSTRAT):
            for cb in range(NCODIR):
                for wb in wz_iter:
                    if wb is None:
                        X = XtX[ci, s, cb].sum(0); Y = Xty[ci, s, cb].sum(0)
                        sy = Syy[ci, s, cb].sum(); sf = SF2[ci, s, cb].sum()
                        sn = SN[ci, s, cb].sum(); sw = Swz[ci, s, cb].sum()
                        n = int(nn[ci, s, cb].sum()); wzc = np.nan
                    else:
                        X = XtX[ci, s, cb, wb]; Y = Xty[ci, s, cb, wb]
                        sy = Syy[ci, s, cb, wb]; sf = SF2[ci, s, cb, wb]
                        sn = SN[ci, s, cb, wb]; sw = Swz[ci, s, cb, wb]
                        n = int(nn[ci, s, cb, wb]); wzc = float(wz_ctr[wb])
                    fit = _fit(X, Y, float(sy), n)
                    if fit is None:
                        continue
                    Nbar = sn / n; mubar = X[0, 1] / n; wzbar = sw / n
                    rms = float(np.sqrt(sf / n))
                    A_N = fit['A'] * Nbar; B_N = fit['B'] * Nbar
                    Ca = fit['C_a']; Cb = fit['C_b']
                    # chi-swing in Newtons at the cell operating point (mubar, wzbar).
                    # regressor is mu*(chi/CHI_SCALE), so dF/dchi term = C*mu/CHI_SCALE.
                    dchi_s = DCHI / CHI_SCALE
                    sw_a = abs(Ca) * mubar * dchi_s * Nbar if np.isfinite(Ca) else np.nan
                    sw_b = abs(Cb) * mubar * wzbar * dchi_s * Nbar if np.isfinite(Cb) else np.nan
                    rows.append(dict(
                        dataset=dataset, scheme=scheme, channel=CHANNELS[ci],
                        stratum=SLIP_LABELS[s], vcod_center=float(cod_ctr[cb]),
                        wz_center=wzc, n=n, channel_rms=rms, N_bar=float(Nbar),
                        mu_bar=float(mubar), wz_bar=float(wzbar),
                        A_dF_dmu=float(A_N), abs_A=abs(float(A_N)), B_intercept=float(B_N),
                        C_a=float(Ca) if np.isfinite(Ca) else np.nan,
                        C_b=float(Cb) if np.isfinite(Cb) else np.nan,
                        R2_form=fit['R2_red'],                 # adequacy of mu*A+B+slip (no chi)
                        R2_form_a=fit['R2_a'], R2_form_b=fit['R2_b'],
                        partial_R2_chi_a=fit['pR2_a'], partial_R2_chi_b=fit['pR2_b'],
                        chi_swing_a=sw_a, chi_swing_b=sw_b,
                    ))
    return rows


def run(data_dir: Path, out_csv: Path, whitelist: Optional[set] = None,
        base_toml: Path = Path('trajectory_files_run_0p5_main/base.toml'),
        force: bool = False) -> pd.DataFrame:
    if out_csv.exists() and not force:
        print(f'[gated] {out_csv} exists -> skip (use --force)'); return pd.read_csv(out_csv)
    N_roller = load_N_per_roller(base_toml)
    print(f'[gated] N_per_roller = {np.round(N_roller, 2)} N')
    quads = discover_quads(data_dir, min_chi=3, mu=None, whitelist=whitelist)
    quad_names = {p.name for g in quads for p in g.values()}
    files = []
    for p in sorted(Path(data_dir).glob('*.arrow')):
        m = cd.parse_arrow_filename(p.name)
        if m is None or (whitelist is not None and p.name not in whitelist):
            continue
        files.append((p, m['mu'], m['chi']))
    print(f'[gated] {len(files)} whitelisted files; {len(quad_names)} in chi-quads')

    acc_full = Accumulator(); acc_quad = Accumulator()
    for k, (p, mu, chi) in enumerate(files, 1):
        acc_full.add_file(p, mu, chi, N_roller)
        if p.name in quad_names:
            acc_quad.add_file(p, mu, chi, N_roller)
        if k % 200 == 0 or k == len(files):
            print(f'  [{k}/{len(files)}] {p.name}')
    acc_full.symmetrize(); acc_quad.symmetrize()

    rows = []
    for ds, acc in (('full', acc_full), ('quad', acc_quad)):
        for scheme in ('unbinned', 'binned'):
            rows += _rows_for_scheme(acc, ds, scheme)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _verdict(df)
    print(f'\n[done] wrote {len(df)} cells -> {out_csv}')
    return df


def _wmean(d, col):
    v = d[col].to_numpy(float); w = d['n'].to_numpy(float); m = np.isfinite(v) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def _verdict(df: pd.DataFrame) -> None:
    if df.empty:
        print('[verdict] no cells'); return
    print('\n[verdict] form F=mu*(A+C*chi)+B :  adequacy R2 + can chi be captured?')
    print(f"  {'dataset':<6}{'scheme':<10}{'form':<6}{'R2_form':>8}{'pR2_chi':>9}{'chiSwing[N]':>12}"
          f"{'|A|':>8}{'cells':>7}{'n':>15}")
    for ds in ('full', 'quad'):
        for scheme in ('unbinned', 'binned'):
            sub = df[(df['dataset'] == ds) & (df['scheme'] == scheme)]
            if sub.empty:
                continue
            absA = _wmean(sub, 'abs_A'); r2f = _wmean(sub, 'R2_form')
            for form in ('a', 'b'):
                d = sub.dropna(subset=[f'partial_R2_chi_{form}'])
                if d.empty:
                    print(f"  {ds:<6}{scheme:<10}({form})   (no identifiable C)"); continue
                print(f"  {ds:<6}{scheme:<10}({form}) {r2f:>8.3f}{_wmean(d,f'partial_R2_chi_{form}'):>9.4f}"
                      f"{_wmean(d,f'chi_swing_{form}'):>12.3f}{absA:>8.2f}{len(d):>7}{int(d['n'].sum()):>15,d}")
    print("  R2_form = adequacy of mu*A+B+slip (no chi); |A|=n-wtd |dF/dmu| (signed A flips with Vcod).")
    print("  form (a)=mu*(A+C*chi)+B [no |wz|] ; (b)=mu*(A+C*|wz|*chi)+B [explicit |wz|]")
    print("  Compare (a)-unbinned vs (a)-binned vs (b): binned~(b)>>unbinned => SSM must resolve |wz|.")


def _cli():
    ap = argparse.ArgumentParser(description='Gated multiplicative F=mu(A+C*chi)+B, two chi forms, two |wz| schemes.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('force_mu_chi_gated.csv'))
    ap.add_argument('--base-toml', type=Path, default=Path('trajectory_files_run_0p5_main/base.toml'))
    ap.add_argument('--whitelist', type=Path, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    wl = None
    if args.whitelist is not None:
        wdf = pd.read_csv(args.whitelist)
        wl = set(wdf.loc[~wdf['combined_reco'].str.startswith('reject'), 'file'])
        print(f'[gated] whitelist: {len(wl)} kept files')
    run(args.data_dir, args.out, whitelist=wl, base_toml=args.base_toml, force=args.force)


if __name__ == '__main__':
    _cli()
