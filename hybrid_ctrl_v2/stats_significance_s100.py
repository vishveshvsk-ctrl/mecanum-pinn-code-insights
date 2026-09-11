# =============================================================================
# hybrid_ctrl_v2/stats_significance_s100.py — paired significance tests, v3 vs s100
# =============================================================================
# Post-hoc statistics for RESULTS_v3_s100.md §9. Two questions:
#
#   Q1  controller gap:  PID-CT vs ASMC, frozen-ESKF closed loop, at BOTH
#       margins (0.9 archived vs 1.0 ablation), both tiers, ct and nt.
#   Q2  margin effect:   s=1.0 minus s=0.9 within each (family, ct/nt) config.
#
# Pairing unit: (trajectory, noise_seed) — the seed sets the noise realisation
# and the trajectory set is a shared design, so differences are paired, not
# pooled. Two aggregation levels:
#
#   run-level   n = 70 (train14) / 40 (test_v3) pairs — paired t + Wilcoxon
#               signed-rank + Cohen's dz. Overstates independence (14/8 distinct
#               trajectories), hence:
#   traj-level  seeds averaged first, n = 14 / 8 pairs — the conservative check.
#
# Inference generalises over the sensor-noise distribution and THIS fixed
# trajectory set; trajectories are design points, not random draws.
#
# Inputs (untouched archived + s100 ESKF CSVs, 5 seeds each):
#   runs_eskf_v3_train14/  runs_eskf_v3_test/          (margin 0.9, 6 configs)
#   runs_eskf_v3_s100/     runs_eskf_v3_s100_test_v3/  (margin 1.0, 4 configs)
# Run:  python stats_significance_s100.py   (from hybrid_ctrl_v2/)
# =============================================================================
import pandas as pd
import numpy as np
from scipy import stats


def load(d):
    df = pd.concat([pd.read_csv(f'{d}/runs_seed{s}.csv') for s in range(1, 6)])
    return df[df['ok'] == True].copy()


def paired_runs(dfA, ca, dfB, cb, label):
    """Run-level paired test on (trajectory, noise_seed); A minus B."""
    A = dfA[dfA.controller == ca].set_index(['trajectory', 'noise_seed'])['score']
    B = dfB[dfB.controller == cb].set_index(['trajectory', 'noise_seed'])['score']
    d = (A[A.index.intersection(B.index)] - B[B.index.intersection(A.index)]).dropna()
    t, pt = stats.ttest_rel(d, np.zeros(len(d)))
    _, pw = stats.wilcoxon(d)
    dz = d.mean() / d.std()
    print(f"{label:46s} n={len(d):3d}  meanD={d.mean():+.5f}  medD={d.median():+.5f}"
          f"  p_t={pt:.2e}  p_W={pw:.2e}  dz={dz:+.2f}")


def paired_traj(dfA, ca, dfB, cb, label):
    """Traj-level check: average seeds first, pair across trajectories."""
    A = dfA[dfA.controller == ca].groupby('trajectory')['score'].mean()
    B = dfB[dfB.controller == cb].groupby('trajectory')['score'].mean()
    d = (A - B).dropna()
    t, pt = stats.ttest_rel(d, np.zeros(len(d)))
    _, pw = stats.wilcoxon(d)
    print(f"{label:46s} n={len(d):2d}  meanD={d.mean():+.5f}"
          f"  p_t={pt:.4f}  p_W={pw:.4f}  A-better={int((d < 0).sum())}/{len(d)}")


def main():
    tr09, te09 = load('runs_eskf_v3_train14'), load('runs_eskf_v3_test')
    tr10, te10 = load('runs_eskf_v3_s100'), load('runs_eskf_v3_s100_test_v3')

    print("=== Q1: PID-CT - ASMC (negative = PID-CT better), run-level ===")
    for df, tier, tag in [(tr09, 'train14', '0.9'), (tr10, 'train14', '1.0'),
                          (te09, 'test_v3', '0.9'), (te10, 'test_v3', '1.0')]:
        sfx = {'0.9': ('ct(s5)', 'nt(s4)', 'ct(s2)', 'nt(s4)'),
               '1.0': ('s100 ct', 's100 nt', 's100 ct', 's100 nt')}[tag]
        paired_runs(df, f'PID-CT {sfx[0]}', df, f'ASMC {sfx[2]}', f'{tier} {tag}: CT ct - ASMC ct')
        paired_runs(df, f'PID-CT {sfx[1]}', df, f'ASMC {sfx[3]}', f'{tier} {tag}: CT nt - ASMC nt')

    print("\n=== Q2: margin effect (s100 - 0.9), run-level ===")
    for c9, c1, tag in [('PID-CT ct(s5)', 'PID-CT s100 ct', 'PID-CT ct'),
                        ('PID-CT nt(s4)', 'PID-CT s100 nt', 'PID-CT nt'),
                        ('ASMC ct(s2)', 'ASMC s100 ct', 'ASMC ct'),
                        ('ASMC nt(s4)', 'ASMC s100 nt', 'ASMC nt')]:
        paired_runs(tr10, c1, tr09, c9, f'{tag} train14')
        paired_runs(te10, c1, te09, c9, f'{tag} test_v3')

    print("\n=== Q2: margin effect (s100 - 0.9), traj-level (seeds averaged) ===")
    for c9, c1, tag in [('PID-CT ct(s5)', 'PID-CT s100 ct', 'PID-CT ct'),
                        ('PID-CT nt(s4)', 'PID-CT s100 nt', 'PID-CT nt'),
                        ('ASMC ct(s2)', 'ASMC s100 ct', 'ASMC ct'),
                        ('ASMC nt(s4)', 'ASMC s100 nt', 'ASMC nt')]:
        paired_traj(tr10, c1, tr09, c9, f'{tag} train14')
        paired_traj(te10, c1, te09, c9, f'{tag} test_v3')


if __name__ == '__main__':
    main()
