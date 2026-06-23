#!/usr/bin/env python
# =============================================================================
# test_chatter_diagnostics.py — synthetic-injection acceptance test (PLAN §7.1).
#
# No real new-scheme data exists yet (the production sweep is gated on the
# solver-ablation winner landing in base.toml), so this is BOTH the unit test
# for chatter_diagnostics.py AND the acceptance gate on the discriminating
# metrics M3/M4/M5: it manufactures Arrow files in the exact datastore.jl schema
# with controllable injected ripple / LuGre / ASMC-chatter / numerical-hash and
# asserts the classifier separates them. Run before trusting any real verdict.
#
#   python test_chatter_diagnostics.py        # -> exit 0 on pass, 1 on failure
#
# Archetypes and required verdicts:
#   clean        roller ripple + slip-modulated LuGre                  -> clean
#   chatter      + coherent tone in BOTH Msw and Fpar (above f_hash)   -> chatter (kept, flagged)
#   hash         + white HF noise in Fpar ONLY (above f_hash)          -> hash    (hard reject)
#   lugre_heavy  + strong high-slip LuGre below the ceiling            -> clean
#   lugre_above  + slip-modulated tone ABOVE f_hash, NOT in Msw        -> clean   (M5 rescue)
# =============================================================================
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatter_diagnostics as cd

_RNG = np.random.default_rng(0)
_FS = 2000.0
_T = 6.0
_t = np.arange(0, _T, 1 / _FS)
_N = len(_t)


def _base_signals(omega0=6.0, vp_mean=0.15):
    omega = omega0 + 0.5 * np.sin(2 * np.pi * 0.2 * _t)
    theta = np.cumsum(omega) / _FS
    vp_slow = np.abs(vp_mean + 0.4 * vp_mean * np.sin(2 * np.pi * 0.3 * _t))
    return omega, theta, vp_slow


def _ripple(theta, amp=4.0):
    return amp * (np.sin(12 * theta) + 0.3 * np.sin(24 * theta))


def _lugre_band(vp_slow, f_c=90.0, amp=3.0):
    return amp * (vp_slow / vp_slow.mean()) * np.sin(2 * np.pi * f_c * _t)


def make_traj(kind: str):
    import pandas as pd
    omega, theta, vp_slow = _base_signals()
    df = pd.DataFrame({'time': _t, 'Vx': 0.3 * np.ones(_N), 'Vy': np.zeros(_N),
                       'psi_dot': np.zeros(_N)})
    f_hash = cd.hash_cutoff_hz(np.abs(omega), vp_slow, mu=0.4, cfg=cd.ChatterConfig())
    for i in range(1, 5):
        om_i = omega + 0.1 * (i - 1)
        th_i = theta + (i - 1) * 0.5
        fpar = _ripple(th_i) + _lugre_band(vp_slow) + 0.05 * _RNG.standard_normal(_N)
        fperp = 0.4 * _ripple(th_i + 0.2) + 0.05 * _RNG.standard_normal(_N)
        msw = 0.05 * _RNG.standard_normal(_N)
        meq = 1.5 + 0.2 * np.sin(2 * np.pi * 0.2 * _t)
        msat = np.clip(meq, -10.0, 10.0)              # smooth applied torque (no burst)
        if kind == 'chatter':
            tone = np.sin(2 * np.pi * (f_hash + 100) * _t)
            msw, fpar = msw + 0.8 * tone, fpar + 1.2 * tone
        elif kind == 'hash':
            sos = cd.butter(6, f_hash + 60, btype='highpass', fs=_FS, output='sos')
            fpar = fpar + 3.0 * cd.sosfiltfilt(sos, _RNG.standard_normal(_N))
        elif kind == 'lugre_heavy':
            fpar = fpar + 2.0 * _lugre_band(vp_slow, f_c=110.0, amp=4.0)
        elif kind == 'lugre_above':
            slip_tone = (vp_slow / vp_slow.mean()) * np.sin(2 * np.pi * (f_hash + 120) * _t)
            fpar = fpar + 3.0 * slip_tone
        elif kind == 'burst':
            # localized 25 Hz control-torque oscillation in the LAST 0.8 s
            bwin = (_t > _t[-1] - 0.8).astype(float)
            msat = msat + bwin * 4.0 * np.sin(2 * np.pi * 25.0 * _t)
        df[f'w{i}'] = om_i
        df[f'theta{i}'] = th_i
        df[f'Fpar_{i}'] = fpar
        df[f'Fperp_{i}'] = fperp
        df[f'util_{i}'] = 0.5 + 0.1 * np.abs(np.sin(2 * np.pi * 0.2 * _t))
        df[f'Msw_{i}'] = msw
        df[f'Meq_{i}'] = meq
        df[f'Msat_{i}'] = msat
        df[f'Vpx_{i}'] = vp_slow
        df[f'Vpy_{i}'] = 0.0 * vp_slow
        df[f'wz_{i}'] = 0.1 * np.ones(_N)
        df[f'Mz_{i}'] = 0.2 * np.sin(12 * th_i)
    return df


# (kind, expected verdict, expected burst_flag). 'burst' has clean FORCES (verdict
# clean) but a localized Msat oscillation that M7 must flag.
CASES = {
    'octagon_c001_mu_0.4_case1_lugre_adamov_chi_0.005.arrow':        ('clean',       'clean',   False),
    'spin_creep_c002_mu_0.4_case1_lugre_adamov_chi_0.005.arrow':     ('chatter',     'chatter', False),
    'coupled_vomega_c003_mu_0.4_case1_lugre_adamov_chi_0.005.arrow': ('hash',        'hash',    False),
    'spiral_orbit_c004_mu_0.4_case1_lugre_adamov_chi_0.005.arrow':   ('lugre_heavy', 'clean',   False),
    'long_circle_c005_mu_0.4_case1_lugre_adamov_chi_0.005.arrow':    ('lugre_above', 'clean',   False),
    'multisine_50percent_cap_c006_mu_0.4_case1_lugre_adamov_chi_0.005.arrow': ('burst', 'clean', True),
}


def main() -> int:
    cfg = cd.ChatterConfig()
    print(f"{'file':<48} {'kind':<12} {'verdict':<9} {'burstiness':>11} {'burst?':>7}")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for fname, (kind, expect, exp_burst) in CASES.items():
            p = Path(tmp) / fname
            feather.write_feather(make_traj(kind), p)
            row = cd.diagnose_file(p, cfg)
            if row.get('error'):
                print(f"  ERROR {fname}: {row['error']}"); ok = False; continue
            got, gb = row['verdict'], row['burst_flag']
            good = (got == expect) and (gb == exp_burst)
            ok &= good
            print(f"{'OK ' if good else 'XX '}{fname[:45]:<45} {kind:<12} {got:<9} "
                  f"{row['msat_burstiness']:>11.1f} {str(gb):>7}  "
                  f"(exp {expect}/{exp_burst})")
    print('\nRESULT:', 'ALL PASS' if ok else 'FAILURE')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
