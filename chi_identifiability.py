#!/usr/bin/env python
# =============================================================================
# chi_identifiability.py — is χ recoverable from the OBSERVABLE forces?
#
# χ reaches the forces only through the spin->translation coupling
# c_t = (8/3π)·|ω_z|·χ — gated by contact spin and LINEAR in χ. This asks,
# empirically and memory-safely, whether that χ-dependence in Fpar/Fperp clears
# the force noise floor. Mz (χ²-dependent, unmeasurable at deployment) is DROPPED.
#
# Method (decided with the user):
#   * Use only the matched χ-quads (same profile/combo/μ/fm, ≥3 χ values) whose
#     EVERY counterpart is whitelisted (well-tracked) — see `--whitelist`. Reference
#     is bit-identical across χ, so the force differences are χ-only.
#   * Bin NONPARAMETRICALLY by the gating variable |ω_z| (contact spin), and
#     stratify by |psi_dot| (the deployment-observable yaw rate; also a proxy
#     for the unbinned bristle-history regime).
#   * Within each (stratum, |ω_z|-bin), regress each force channel on χ (LINEAR)
#     while CONTROLLING for slip (Vpx, Vpy, |Vp|, |Vp|² — nonlinear) and per-wheel
#     offset (4 dummies). Linear-χ is physically correct (the coupling is linear)
#     and ~12 orders better conditioned than {χ,χ²} on the tiny χ grid — that
#     collinearity was the μ=0.3 blow-up. Dropping |Vp|², by contrast, shifts the
#     swing (it carries real slip-confound), so slip stays nonlinear.
#   * Identifiability = ABSOLUTE χ-swing in N (units-trap-safe), with partial-F /
#     partial-R² of the {χ} block secondary. Swing rising with |ω_z| confirms the
#     spin-coupling mechanism.
#
# Why no OOM: never pools raw samples. It accumulates per-bin sufficient
# statistics (XᵀX 9×9, Xᵀy, Σy², n) streaming one file at a time. The whole
# accumulator is a few×10⁴ floats regardless of dataset size.
#
#   python chi_identifiability.py --data-dir ../data/Simulation_... --out chi_ident.csv
# =============================================================================
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import chatter_diagnostics as cd   # parse_arrow_filename only

try:                                # Windows cp1252 can't encode χ/ω in prints
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- design knobs ---
# CHANNELS: forces only. Mz is DROPPED — it is χ²-dependent and unmeasurable at
# deployment; modelling the forces (linear-χ) does not need it.
CHANNELS = ('Fpar', 'Fperp')
CHI_SCALE = 0.005                      # scale χ to ~O(1) for conditioning
WZ_EDGES = np.linspace(0.0, 20.0, 17)  # 16 |ω_z| bins (rad/s); values clipped in
PSI_EDGES = (0.0, 0.5, 2.0, np.inf)    # |psi_dot| strata: creep / moderate / fast-yaw
PSI_LABELS = ('psi<0.5', '0.5<=psi<2', 'psi>=2')
N_BINS = len(WZ_EDGES) - 1
N_STRATA = len(PSI_LABELS)
# regressor layout: [w1,w2,w3,w4, χs, Vpx, Vpy, |Vp|, |Vp|²]
# χ enters LINEARLY: the force coupling c_t=(8/3π)|ω_z|χ is linear in χ (the χ²
# term belonged to Mz, now dropped). Linear-χ is physically correct AND ~12 orders
# better conditioned than {χ,χ²} on the tiny χ grid {0,.002,.005,.008} (the χ/χ²
# collinearity drove the μ=0.3 blow-up). SLIP stays nonlinear ({Vpx,Vpy,|Vp|,|Vp|²}):
# dropping |Vp|² shifted the χ-swing materially (it carries real slip-confound).
P = 9
CHI_COLS = (4,)                        # the {χ} block being tested (linear)
MIN_N = 400                            # min samples to fit a (stratum,bin,channel) cell


def discover_quads(data_dir: Path, min_chi: int = 3,
                   mu: Optional[float] = None,
                   whitelist: Optional[set] = None) -> List[Dict[str, Path]]:
    """Group Arrow files by (profile, combo, mu, fm); return groups that span
    >= min_chi distinct χ (the matched quads with usable χ variation).
    If `mu` is given, restrict to that single friction coefficient — χ enters the
    force scaled by the friction state, so pooling μ levels blurs the χ-swing;
    keep a μ-batch on its own.
    If `whitelist` (a set of bare filenames) is given, keep a quad ONLY when EVERY
    one of its χ-counterparts is whitelisted (well-tracked). A quad with any
    rejected counterpart is dropped wholesale — a partly-failed quad regresses
    gross-slip / saturated samples against the well-tracked ones, which both biases
    the χ-swing and ill-conditions the per-bin fit (the μ=0.3 blow-up)."""
    groups: Dict[Tuple, Dict[str, Path]] = defaultdict(dict)
    for p in sorted(Path(data_dir).glob('*.arrow')):
        m = cd.parse_arrow_filename(p.name)
        if m is None:
            continue
        if mu is not None and abs(m['mu'] - mu) >= 1e-9:
            continue
        key = (m['profile'], m['combo_idx'], m['mu'], m['friction_model'])
        groups[key][f"{m['chi']:.3f}"] = p
    out = []
    for g in groups.values():
        if len(g) < min_chi:
            continue
        if whitelist is not None and not all(p.name in whitelist for p in g.values()):
            continue
        out.append(g)
    return out


_COLS = (['psi_dot']
         + [f'wz_{i}' for i in range(1, 5)]
         + [f'Vpx_{i}' for i in range(1, 5)] + [f'Vpy_{i}' for i in range(1, 5)]
         + [f'Fpar_{i}' for i in range(1, 5)] + [f'Fperp_{i}' for i in range(1, 5)]
         + [f'Mz_{i}' for i in range(1, 5)])


class Accumulator:
    """Streaming per-(stratum, channel, wz-bin) sufficient stats."""
    def __init__(self):
        self.XtX = np.zeros((N_STRATA, len(CHANNELS), N_BINS, P, P))
        self.Xty = np.zeros((N_STRATA, len(CHANNELS), N_BINS, P))
        self.Syy = np.zeros((N_STRATA, len(CHANNELS), N_BINS))
        self.n = np.zeros((N_STRATA, len(CHANNELS), N_BINS), dtype=np.int64)

    def add_file(self, path: Path, chi: float) -> None:
        df = feather.read_feather(path, columns=_COLS)
        psidot = np.abs(df['psi_dot'].to_numpy(np.float64))
        strat = np.clip(np.digitize(psidot, PSI_EDGES) - 1, 0, N_STRATA - 1)
        chis = chi / CHI_SCALE
        for i in range(4):                              # per wheel
            wz = np.abs(df[f'wz_{i+1}'].to_numpy(np.float64))
            vpx = df[f'Vpx_{i+1}'].to_numpy(np.float64)
            vpy = df[f'Vpy_{i+1}'].to_numpy(np.float64)
            vpm = np.hypot(vpx, vpy)
            wzbin = np.clip(np.digitize(wz, WZ_EDGES) - 1, 0, N_BINS - 1)
            Nn = len(wz)
            X = np.zeros((Nn, P))
            X[:, i] = 1.0                                # wheel dummy
            X[:, 4] = chis                               # χ (LINEAR only)
            X[:, 5] = vpx;  X[:, 6] = vpy; X[:, 7] = vpm; X[:, 8] = vpm * vpm
            for ci, ch in enumerate(CHANNELS):
                y = df[f'{ch}_{i+1}'].to_numpy(np.float64)
                for s in range(N_STRATA):
                    for b in range(N_BINS):
                        msk = (strat == s) & (wzbin == b)
                        if not msk.any():
                            continue
                        Xm = X[msk]; ym = y[msk]
                        self.XtX[s, ci, b] += Xm.T @ Xm
                        self.Xty[s, ci, b] += Xm.T @ ym
                        self.Syy[s, ci, b] += float(ym @ ym)
                        self.n[s, ci, b] += int(msk.sum())


CHI_MAX_S = 0.008 / CHI_SCALE          # top of the χ grid, scaled (= 1.6)
_NAN = dict(F=np.nan, partial_R2_chi=np.nan, slope_dF_dchi=np.nan,
            chi_swing_abs=np.nan, channel_rms=np.nan, rel_swing=np.nan)


def _fit_cell(XtX: np.ndarray, Xty: np.ndarray, Syy: float, n: int) -> Dict[str, float]:
    """Covariate-controlled χ-block fit for one (stratum, channel, |ω_z|-bin).

    EFFECT SIZE is the headline, not F: with ~10⁶ samples/bin the partial-F is
    always 'significant', so it can't discriminate. We report:
      * chi_swing_abs = |F̂(χ_max) − F̂(0)| from the fitted {χ,χ²} terms — the
        χ-induced force change in ABSOLUTE units (N or N·m). This is what a real
        noise floor must be compared against.
      * rel_swing = chi_swing_abs / channel_rms — relative modulation depth.
      * partial_R2_chi, F — secondary (R²_χ→1 for a χ-dominated channel like Mz
        even when its absolute swing is negligible; that's the units trap)."""
    keep = [j for j in range(P) if XtX[j, j] > 0]      # drop empty wheel dummies
    if n <= len(keep) + 50:
        return dict(_NAN)
    A = XtX[np.ix_(keep, keep)]; b = Xty[keep]
    # Conditioning guard: a sparse high-spin bin can leave the {χ,χ²} block nearly
    # collinear with the slip controls → near-singular A → coefficients (and the
    # χ-swing) explode to ~1e17 with a nonsensical NEGATIVE partial-R². np.solve
    # only raises on EXACT singularity, so guard the condition number explicitly.
    if np.linalg.cond(A) > 1e10:
        return dict(_NAN)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return dict(_NAN)
    rss_full = Syy - float(beta @ b)
    red_idx = [jj for jj, j in enumerate(keep) if j not in CHI_COLS]
    Ar = A[np.ix_(red_idx, red_idx)]; br = b[red_idx]
    try:
        beta_r = np.linalg.solve(Ar, br)
    except np.linalg.LinAlgError:
        return dict(_NAN)
    rss_red = Syy - float(beta_r @ br)
    df2 = n - len(keep)
    # rss_full must be <= rss_red (more regressors never fit worse); if not, the
    # cell is numerically broken (the negative-partial-R² blow-up) — reject it.
    if rss_full > rss_red:
        return dict(_NAN)
    if rss_full <= 0 or df2 <= 0 or rss_red <= 0:
        return dict(_NAN)
    bmap = {j: beta[k] for k, j in enumerate(keep)}
    b_chi = bmap.get(4, 0.0)
    swing = abs(b_chi * CHI_MAX_S)             # |ΔF| over χ∈[0,0.008], linear χ
    rms = float(np.sqrt(Syy / n))
    return dict(
        F=((rss_red - rss_full) / len(CHI_COLS)) / (rss_full / df2),
        partial_R2_chi=(rss_red - rss_full) / rss_red,
        slope_dF_dchi=b_chi / CHI_SCALE,
        chi_swing_abs=swing,
        channel_rms=rms,
        rel_swing=swing / rms if rms > 1e-12 else np.nan,
    )


def run(data_dir: Path, out_csv: Path, mu: Optional[float] = None,
        whitelist: Optional[set] = None) -> pd.DataFrame:
    quads = discover_quads(data_dir, mu=mu, whitelist=whitelist)
    nfiles = sum(len(g) for g in quads)
    tag = ' (whitelisted-only)' if whitelist is not None else ''
    print(f'[chi] {len(quads)} matched χ-quads{tag}, {nfiles} files')
    acc = Accumulator()
    done = 0
    for g in quads:
        for chi_str, path in g.items():
            acc.add_file(path, float(chi_str))
            done += 1
            if done % 50 == 0 or done == nfiles:
                print(f'  [{done}/{nfiles}] {path.name}')

    rows = []
    wz_centers = 0.5 * (WZ_EDGES[:-1] + WZ_EDGES[1:])
    for s in range(N_STRATA):
        for ci, ch in enumerate(CHANNELS):
            for b in range(N_BINS):
                n = int(acc.n[s, ci, b])
                if n < MIN_N:
                    continue
                stats = _fit_cell(acc.XtX[s, ci, b], acc.Xty[s, ci, b],
                                  float(acc.Syy[s, ci, b]), n)
                rows.append(dict(stratum=PSI_LABELS[s], channel=ch,
                                 wz_center=float(wz_centers[b]), n=n, **stats))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _verdict(df)
    print(f'\n[done] wrote {len(df)} cells -> {out_csv}')
    return df


def _verdict(df: pd.DataFrame) -> None:
    """Headline: ABSOLUTE χ-induced force swing in the high-|ω_z| bins, per
    channel, and whether it rises with spin (the coupling mechanism)."""
    if df.empty:
        print('[verdict] no fitted cells'); return
    hi = df['wz_center'] >= df['wz_center'].quantile(0.6)
    print('\n[verdict] χ-induced swing |ΔF| over χ∈[0,0.008] (covariate-controlled, hi-spin median)')
    print(f"  {'channel':<7} {'|ΔF|_χ abs':>14} {'rel to RMS':>11} {'partial_R²_χ':>13} {'rises w/ spin':>14}")
    for ch in CHANNELS:
        d = df[df['channel'] == ch].dropna(subset=['chi_swing_abs'])
        if d.empty:
            continue
        dh = d[hi.loc[d.index]]
        rises = (np.corrcoef(d['wz_center'], d['chi_swing_abs'])[0, 1] > 0.2
                 if len(d) > 3 else False)
        print(f"  {ch:<7} {dh['chi_swing_abs'].median():>10.4f} N   "
              f"{dh['rel_swing'].median():>11.3f} {dh['partial_R2_chi'].median():>13.4f} "
              f"{str(rises):>14}")
    print("  Observable forces (Fpar/Fperp): compare |ΔF|_χ abs against the force")
    print("  noise floor — that is χ-identifiability for the deployable inverse map.")
    print("  (Mz dropped: χ²-dependent and unmeasurable at deployment.)")


def _cli():
    ap = argparse.ArgumentParser(description='χ-identifiability from observable forces.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('chi_identifiability.csv'))
    ap.add_argument('--mu', type=float, default=None, help='restrict to a single mu value')
    ap.add_argument('--whitelist', type=Path, default=None,
                    help='combined CSV: keep only quads whose every χ-counterpart is '
                         'whitelisted (combined_reco not reject)')
    args = ap.parse_args()
    wl = None
    if args.whitelist is not None:
        wdf = pd.read_csv(args.whitelist)
        wl = set(wdf.loc[~wdf['combined_reco'].str.startswith('reject'), 'file'])
        print(f'[chi] whitelist: {len(wl)} kept files from {args.whitelist}')
    run(args.data_dir, args.out, mu=args.mu, whitelist=wl)


if __name__ == '__main__':
    _cli()
