#!/usr/bin/env python
# =============================================================================
# test_sampling_sensitivity.py — synthetic validation for sampling_sensitivity.py.
#
# No real data yet, so this proves the measurement separates the regimes it must:
#   * a clean low-frequency trajectory survives downsampling intact at every
#     rate and mode (recon error ~ 0, no verdict flips);
#   * a trajectory with genuine content ABOVE 250 Hz shows the two modes DIVERGE
#     at 500 Hz (anti-aliased loses it; naive aliases it -> larger recon error)
#     but AGREE at 1000 Hz (content below the 500 Hz Nyquist is preserved).
# This is the acceptance gate before trusting any real sampling verdict.
#   python test_sampling_sensitivity.py     # -> exit 0 on pass, 1 on failure
# =============================================================================
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatter_diagnostics as cd
import sampling_sensitivity as ss
import test_chatter_diagnostics as tc


def _analyze(kind: str):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f'octagon_c001_mu_0.4_case1_lugre_adamov_chi_0.005.arrow'
        feather.write_feather(tc.make_traj(kind), p)
        return ss.analyze_file(p, cd.ChatterConfig(), rates=(1000.0, 500.0))


def main() -> int:
    ok = True
    print(f"{'kind':<12} {'rate/mode':<16} {'recon_Fpar':>10} {'recon_Mz':>9} {'verdict':>9} flip")
    rows = {}
    for kind in ('clean', 'lugre_above'):
        r = _analyze(kind)
        rows[kind] = r
        print(f"{kind:<12} {'native':<16} {'-':>10} {'-':>9} {r['verdict_native']:>9}")
        for rate in (1000, 500):
            for mode in ('antialias', 'naive'):
                print(f"{'':<12} {f'{rate}Hz {mode}':<16} "
                      f"{r[f'recon_{mode}_{rate}_Fpar']:>10.4f} "
                      f"{r[f'recon_{mode}_{rate}_Mz']:>9.4f} "
                      f"{r[f'verdict_{mode}_{rate}']:>9} {r[f'flip_{mode}_{rate}']}")

    c = rows['clean']
    la = rows['lugre_above']

    # 1. clean trajectory: faithful at every rate/mode, no flips
    clean_recon = [c[f'recon_{m}_{rate}_{fam}'] for rate in (1000, 500)
                   for m in ('antialias', 'naive') for fam in ss.FAMILIES]
    cond1 = all(e < 0.15 for e in clean_recon)
    cond1 &= not any(c[f'flip_{m}_{rate}'] for rate in (1000, 500) for m in ('antialias', 'naive'))
    print(f"\n[check 1] clean stays faithful + no flips: {'PASS' if cond1 else 'FAIL'}"
          f"  (max recon {max(clean_recon):.3f})")
    ok &= cond1

    # 2. HF content (~325 Hz): modes AGREE at 1000 Hz (Nyq 500, preserved)
    a1k, n1k = la['recon_antialias_1000_Fpar'], la['recon_naive_1000_Fpar']
    cond2 = (a1k < 0.20) and (abs(a1k - n1k) < 0.10)
    print(f"[check 2] 1000 Hz preserves HF, modes agree: {'PASS' if cond2 else 'FAIL'}"
          f"  (antialias {a1k:.3f}, naive {n1k:.3f})")
    ok &= cond2

    # 3. HF content: modes DIVERGE at 500 Hz (Nyq 250) — naive aliasing costs
    #    more reconstruction error than anti-aliased loss.
    a500, n500 = la['recon_antialias_500_Fpar'], la['recon_naive_500_Fpar']
    cond3 = (n500 > a500) and (a500 > 0.05)
    print(f"[check 3] 500 Hz: naive aliases worse than anti-aliased loss: "
          f"{'PASS' if cond3 else 'FAIL'}  (antialias {a500:.3f}, naive {n500:.3f})")
    ok &= cond3

    print('\nRESULT:', 'ALL PASS' if ok else 'FAILURE')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
