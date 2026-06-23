#!/usr/bin/env python
# =============================================================================
# roller_slip_fraction.py — how much of the gate variables is HIDDEN roller rate?
#
# The mu-gate rides on co-directional slip Vp_par and the chi-gate rides on
# contact spin wz. Both contain the roller rate gamma_i, which the PINN cannot
# measure at deployment. This model-free diagnostic puts a hard number on the
# split: per wheel each gate variable decomposes EXACTLY (validated to machine
# precision against stored Vpx/Vpy/wz) into measurable physical COMPONENTS plus a
# ROLLER component (proportional to gamma_i):
#
#   Vpx_i = Vx - psi_dot*(py+DY) - w*R          + g*s*(Rd*cth - R) + DY*g*c*sth
#           '-body-' '--- yaw ---'  '-wheel-'      '------------ roller ----------'
#   Vpy_i = Vy + psi_dot*px                      + g*c*(R*cth - Rd)
#   wz_i  = psi_dot                              + g*sth*c
#   (run_one.jl:680-684; DY = Rd*tan(delta)*tan(th~), th~ = sawtooth_tanh(theta),
#    sawtooth_tanh(x)=atan2(60*sin12x, 60*cos12x+1)/12; s,c=sin/cos(delta))
#
# Gate axes (datastore rotates FORCE by delta but NOT slip — we rotate slip):
#   Vp_par  = Vpx*cos d + Vpy*sin d   (drive / Fpar / mu-gate)
#   Vp_perp = -Vpx*sin d + Vpy*cos d  (free-roll / Fperp)
# Each component (body/yaw/wheel/roller) is rotated into the par/perp axes.
#
# TWO complementary fraction metrics (decided with the user):
#   * SLIP velocities (Vp_par/Vp_perp) are a LINEAR SUM of components, so the
#     honest split is a MAGNITUDE (L1) fraction: roller_mag_frac =
#     Sum|roller| / Sum(|body|+|yaw|+|wheel|+|roller|), in [0,1]. The energy
#     fraction is degenerate here (the measurable & roller parts can be large and
#     opposing -> cross term -> meas_efrac>1 and the Vp_perp near-cancellation
#     blow-up). Magnitude fraction sidesteps that and attributes the velocity
#     budget cleanly across components.
#   * CONTACT SPIN (wz) keeps the ENERGY fraction roller_energy_frac =
#     Sum(roll^2)/Sum(tot^2), because that EQUALS the squared relative error of
#     approximating wz by its measurable part (psi_dot) -- the deployment question.
#
# Method (mirrors chi_/mu_identifiability): streaming sufficient stats per bin
# (energy: Smm/Srr/Stt/Smr; magnitude: Sum|comp| per component; ratio histogram),
# never pools raw samples. NO --mu filter (fraction is mu-independent kinematics).
# Stratify by |psi_dot| (PSI_EDGES); bin Vp_par/Vp_perp by SIGNED slip (VCOD_EDGES),
# bin wz by |wz| (WZ_EDGES).
#
# VERDICT NUMBERS (do the gates need roller reconstruction?):
#   (a) wz roller ENERGY-frac in high-|wz| bins  (chi-gate; = approx error^2)
#   (b) Vp_par roller MAGNITUDE-frac in gross-slip bins (mu-gate)
#
#   python roller_slip_fraction.py --data-dir ../data/Simulation_... \
#          --out roller_slip_fraction.csv --whitelist diagnostics_combined.csv
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import chatter_diagnostics as cd   # parse_arrow_filename only

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- geometry (base.toml; mu-invariant) ---
R, RD = 0.05, 0.0355
H, L = 0.235, 0.15
DELTA = np.array([-np.pi / 4, np.pi / 4, np.pi / 4, -np.pi / 4])
SD, CD, TD = np.sin(DELTA), np.cos(DELTA), np.tan(DELTA)
PX = np.array([H, H, -H, -H])          # wc_x
PY = np.array([L, -L, L, -L])          # wc_y
TANH_K = 60.0                          # run_one.jl SAWTOOTH=:tanh, TANH_K=60


def sawtooth_tanh(theta: np.ndarray) -> np.ndarray:
    """Exact port of run_one.jl sawtooth_tanh (the ACTIVE variant, not Fourier)."""
    return np.arctan2(TANH_K * np.sin(12 * theta), TANH_K * np.cos(12 * theta) + 1) / 12.0


# --- binning (reuse the sibling diagnostics' edges) ---
PSI_EDGES = (0.0, 0.5, 2.0, np.inf)
PSI_LABELS = ('psi<0.5', '0.5<=psi<2', 'psi>=2')
N_STRATA = len(PSI_LABELS)

VCOD_MAX = 0.30
VCOD_EDGES = np.linspace(-VCOD_MAX, VCOD_MAX, 17)     # 16 signed bins (mu-study)
WZ_EDGES = np.linspace(0.0, 20.0, 17)                 # 16 |wz| bins (chi-study)
N_BINS = 16
assert len(VCOD_EDGES) - 1 == N_BINS and len(WZ_EDGES) - 1 == N_BINS

QUANTS = ('Vp_par', 'Vp_perp', 'wz')                  # bin axes: signed / signed / |.|
N_Q = len(QUANTS)
COMP_NAMES = ('body', 'yaw', 'wheel', 'roller')       # roller is the LAST (hidden) one
N_COMP = len(COMP_NAMES)
ROLLER = 3                                            # index of roller in COMP_NAMES
RHIST_BINS = 100                                      # ratio |roll|/(|meas|+|roll|) hist
MIN_N = 400
TINY = 1e-12

# verdict thresholds
GROSS_VCOD = 0.15      # |Vp_par| >= this = gross / mu-identifiable regime (TRAJ §8)
HI_WZ_QUANTILE = 0.6   # high-spin bins (mirror chi_identifiability verdict)

_COLS = (['Vx', 'Vy', 'psi_dot']
         + [f'theta{i}' for i in range(1, 5)]
         + [f'w{i}' for i in range(1, 5)]
         + [f'gamma{i}' for i in range(1, 5)])


def discover_files(data_dir: Path, whitelist: Optional[set] = None) -> List[Path]:
    out = []
    for p in sorted(Path(data_dir).glob('*.arrow')):
        if cd.parse_arrow_filename(p.name) is None:
            continue
        if whitelist is not None and p.name not in whitelist:
            continue
        out.append(p)
    return out


class Accumulator:
    """Per-(quantity, stratum, bin) streaming stats:
      * energy (meas vs roller, two-way): Smm, Srr, Stt, Smr  -> energy_frac, align
      * magnitude (per component, L1): Sabs[component]         -> mag_frac
      * ratio histogram                                        -> pointwise median
    meas = body + yaw + wheel (combined measurable); roller is the hidden component."""
    def __init__(self):
        shp = (N_Q, N_STRATA, N_BINS)
        self.Smm = np.zeros(shp); self.Srr = np.zeros(shp)
        self.Stt = np.zeros(shp); self.Smr = np.zeros(shp)
        self.n = np.zeros(shp, dtype=np.int64)
        self.Sabs = np.zeros((N_Q, N_COMP, N_STRATA, N_BINS))   # sum |component|
        self.Rhist = np.zeros(shp + (RHIST_BINS,))

    def _accum(self, qi: int, strat: np.ndarray, vbin: np.ndarray, comps) -> None:
        # comps: [body, yaw, wheel, roller] signed arrays (already rotated to the axis)
        meas = comps[0] + comps[1] + comps[2]
        roll = comps[ROLLER]
        tot = meas + roll
        flat = strat * N_BINS + vbin
        nc = N_STRATA * N_BINS
        self.Smm[qi] += np.bincount(flat, meas * meas, nc).reshape(N_STRATA, N_BINS)
        self.Srr[qi] += np.bincount(flat, roll * roll, nc).reshape(N_STRATA, N_BINS)
        self.Stt[qi] += np.bincount(flat, tot * tot, nc).reshape(N_STRATA, N_BINS)
        self.Smr[qi] += np.bincount(flat, meas * roll, nc).reshape(N_STRATA, N_BINS)
        self.n[qi] += np.bincount(flat, None, nc).reshape(N_STRATA, N_BINS).astype(np.int64)
        for ci in range(N_COMP):
            self.Sabs[qi, ci] += np.bincount(flat, np.abs(comps[ci]), nc).reshape(N_STRATA, N_BINS)
        denom = np.abs(meas) + np.abs(roll)
        ok = denom > TINY
        if ok.any():
            r = np.clip(np.abs(roll[ok]) / denom[ok], 0.0, 1.0)
            rbin = np.clip((r * RHIST_BINS).astype(np.int64), 0, RHIST_BINS - 1)
            flat2 = flat[ok] * RHIST_BINS + rbin
            self.Rhist[qi] += np.bincount(flat2, None,
                                          nc * RHIST_BINS).reshape(N_STRATA, N_BINS, RHIST_BINS)

    def add_file(self, path: Path) -> None:
        df = feather.read_feather(path, columns=_COLS)
        Vx = df['Vx'].to_numpy(np.float64); Vy = df['Vy'].to_numpy(np.float64)
        pdt = df['psi_dot'].to_numpy(np.float64)
        strat = np.clip(np.digitize(np.abs(pdt), PSI_EDGES) - 1, 0, N_STRATA - 1)
        z = np.zeros_like(pdt)
        for i in range(4):
            th = df[f'theta{i+1}'].to_numpy(np.float64)
            w = df[f'w{i+1}'].to_numpy(np.float64)
            g = df[f'gamma{i+1}'].to_numpy(np.float64)
            tt = sawtooth_tanh(th); st, ct = np.sin(tt), np.cos(tt)
            DY = RD * TD[i] * np.tan(tt)
            cdi, sdi = CD[i], SD[i]
            # wheel-frame component pieces (x, y) — measurable: body, yaw, wheel
            bx, by = Vx, Vy
            yx, yy = -pdt * (PY[i] + DY), pdt * PX[i]
            wx = -w * R
            rx = g * sdi * (RD * ct - R) + DY * g * cdi * st     # roller x
            ry = g * cdi * (R * ct - RD)                         # roller y

            def par(X, Y, c=cdi, s=sdi):  return X * c + Y * s   # Vp_par  = X cos d + Y sin d
            def perp(X, Y, c=cdi, s=sdi): return -X * s + Y * c  # Vp_perp = -X sin d + Y cos d

            comps_par = [par(bx, by), par(yx, yy), par(wx, z), par(rx, ry)]
            tot_par = comps_par[0] + comps_par[1] + comps_par[2] + comps_par[3]
            self._accum(0, strat, np.clip(np.digitize(tot_par, VCOD_EDGES) - 1, 0, N_BINS - 1), comps_par)

            comps_perp = [perp(bx, by), perp(yx, yy), perp(wx, z), perp(rx, ry)]
            tot_perp = comps_perp[0] + comps_perp[1] + comps_perp[2] + comps_perp[3]
            self._accum(1, strat, np.clip(np.digitize(tot_perp, VCOD_EDGES) - 1, 0, N_BINS - 1), comps_perp)

            # wz = psi_dot + roller: only yaw (psi_dot) and roller components
            comps_wz = [z, pdt, z, g * st * cdi]
            tot_wz = comps_wz[1] + comps_wz[3]
            self._accum(2, strat, np.clip(np.digitize(np.abs(tot_wz), WZ_EDGES) - 1, 0, N_BINS - 1), comps_wz)


def _rhist_median(h: np.ndarray) -> float:
    tot = h.sum()
    if tot <= 0:
        return np.nan
    c = np.cumsum(h)
    k = int(np.searchsorted(c, 0.5 * tot))
    return float((k + 0.5) / RHIST_BINS)


def run(data_dir: Path, out_csv: Path, whitelist: Optional[set] = None,
        force: bool = False) -> pd.DataFrame:
    if out_csv.exists() and not force:
        print(f'[roller] {out_csv} exists -> resume/skip (use --force to recompute)')
        return pd.read_csv(out_csv)
    files = discover_files(data_dir, whitelist=whitelist)
    tag = ' (whitelisted-only)' if whitelist is not None else ''
    print(f'[roller] {len(files)} files{tag}')
    acc = Accumulator()
    for k, p in enumerate(files, 1):
        acc.add_file(p)
        if k % 200 == 0 or k == len(files):
            print(f'  [{k}/{len(files)}] {p.name}')

    edges = {'Vp_par': VCOD_EDGES, 'Vp_perp': VCOD_EDGES, 'wz': WZ_EDGES}
    rows = []
    for qi, q in enumerate(QUANTS):
        ctr = 0.5 * (edges[q][:-1] + edges[q][1:])
        for s in range(N_STRATA):
            for b in range(N_BINS):
                n = int(acc.n[qi, s, b])
                if n < MIN_N:
                    continue
                Smm = acc.Smm[qi, s, b]; Srr = acc.Srr[qi, s, b]
                Stt = acc.Stt[qi, s, b]; Smr = acc.Smr[qi, s, b]
                sabs = acc.Sabs[qi, :, s, b]; tabs = sabs.sum()
                magf = sabs / tabs if tabs > 0 else np.full(N_COMP, np.nan)
                align = Smr / np.sqrt(Smm * Srr) if Smm > 0 and Srr > 0 else np.nan
                rows.append(dict(
                    stratum=PSI_LABELS[s], quantity=q, bin_center=float(ctr[b]), n=n,
                    meas_rms=float(np.sqrt(Smm / n)),
                    roller_rms=float(np.sqrt(Srr / n)),
                    total_rms=float(np.sqrt(Stt / n)),
                    # magnitude (L1) attribution — the headline for slip velocities
                    roller_mag_frac=float(magf[ROLLER]),
                    body_mag_frac=float(magf[0]),
                    yaw_mag_frac=float(magf[1]),
                    wheel_mag_frac=float(magf[2]),
                    # energy split — the headline for wz (= meas-only approx error^2)
                    roller_energy_frac=float(Srr / Stt) if Stt > 0 else np.nan,
                    meas_energy_frac=float(Smm / Stt) if Stt > 0 else np.nan,
                    roller_over_total_median=_rhist_median(acc.Rhist[qi, s, b]),
                    meas_roller_align=float(align),
                ))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _verdict(df)
    print(f'\n[done] wrote {len(df)} cells -> {out_csv}')
    return df


def _wmean(d: pd.DataFrame, col: str) -> float:
    v = d[col].to_numpy(float); w = d['n'].to_numpy(float)
    m = np.isfinite(v) & (w > 0)
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def _verdict(df: pd.DataFrame) -> None:
    if df.empty:
        print('[verdict] no cells'); return
    print('\n[verdict] roller (hidden gamma) share of each gate variable')
    # (a) wz at high spin — ENERGY fraction = squared error of measured-only approx
    wz = df[df['quantity'] == 'wz']
    if not wz.empty:
        thr = wz['bin_center'].quantile(HI_WZ_QUANTILE)
        hi = wz[wz['bin_center'] >= thr]
        ef = _wmean(hi, 'roller_energy_frac')
        print(f"  chi-gate  wz   |wz|>={thr:.1f}: roller_ENERGY_frac={ef:.3f}  "
              f"(meas-only RMS err {np.sqrt(ef)*100:.0f}%)  median={_wmean(hi,'roller_over_total_median'):.2f}  "
              f"meas_mag_frac={_wmean(hi,'yaw_mag_frac'):.3f}")
    # (b) Vp_par at gross slip — MAGNITUDE fraction
    vp = df[df['quantity'] == 'Vp_par']
    if not vp.empty:
        gr = vp[vp['bin_center'].abs() >= GROSS_VCOD]
        lo = vp[vp['bin_center'].abs() < GROSS_VCOD]
        for lab, d in (('gross |Vp_par|>=%.2f' % GROSS_VCOD, gr), ('bulk  |Vp_par|<%.2f' % GROSS_VCOD, lo)):
            print(f"  mu-gate   Vp_par {lab}: roller_MAG_frac={_wmean(d,'roller_mag_frac'):.3f}  "
                  f"[body={_wmean(d,'body_mag_frac'):.2f} yaw={_wmean(d,'yaw_mag_frac'):.2f} "
                  f"wheel={_wmean(d,'wheel_mag_frac'):.2f} roller={_wmean(d,'roller_mag_frac'):.2f}]")
    print("  SLIP uses MAGNITUDE (L1) fraction: roller's share of |body|+|yaw|+|wheel|+|roller|.")
    print("  SPIN uses ENERGY fraction = (relative error of approximating wz by psi_dot)^2.")
    print("  High roller share => reconstruction MANDATORY; low => gate rides on measurables.")


def _cli():
    ap = argparse.ArgumentParser(description='Roller (hidden gamma) fraction of the gate variables.')
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=Path('roller_slip_fraction.csv'))
    ap.add_argument('--whitelist', type=Path, default=None,
                    help='combined CSV: keep only files whose combined_reco is not reject')
    ap.add_argument('--force', action='store_true', help='recompute even if --out exists')
    args = ap.parse_args()
    wl = None
    if args.whitelist is not None:
        wdf = pd.read_csv(args.whitelist)
        wl = set(wdf.loc[~wdf['combined_reco'].str.startswith('reject'), 'file'])
        print(f'[roller] whitelist: {len(wl)} kept files from {args.whitelist}')
    run(args.data_dir, args.out, whitelist=wl, force=args.force)


if __name__ == '__main__':
    _cli()
