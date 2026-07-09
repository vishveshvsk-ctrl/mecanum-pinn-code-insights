# Velocity Front-End Drift Audit — ACCEPTANCE

Sample: 5345 files — {'spin_creep': 1692, 'octagon': 1422, 'long_circle': 762, 'coupled_vomega': 647, 'ellipse': 288, 'spiral_orbit': 204, 'multisine_50percent_cap': 165, 'multisine_75percent_cap': 165} (every listed spin_creep file included: the weak-anchor stress case).
Thresholds: overall RMSE <= 0.0384 m/s (2% of PRED_P95['Vx']=1.919952); low-slip RMSE <= 0.0100 m/s (v_str, pooled over bins ('[0,0.005)', '[0.005,0.02)')).

**`rmse_pooled_*` (the pass/fail columns) is the PRIMARY metric: n-weighted RMSE computed within each profile, then averaged EQUALLY across profiles** — the sample deliberately over-represents spin_creep (every approved file, the stress case), so a raw whole-sample pool would just be reporting spin_creep's number under an "overall" label. `rmse_wholesample_overall` (population-weighted-by-sample-count) and `rmse_spin_creep_overall` (the stress case alone) are included alongside for context.

**No (crossover, integrator) at rate=500 Hz / stage=real clears both thresholds against the per-profile-equal-weighted sample — see the table below; consider widening the crossover grid or revisiting the noise-stage placeholders.**

## Pass/fail table

| integrator | crossover_hz | rate_hz | stage | rmse_pooled_overall | pass_overall | rmse_pooled_lowslip | pass_lowslip | pass_both | rmse_wholesample_overall | rmse_spin_creep_overall |
|---|---|---|---|---|---|---|---|---|---|---|
| euler | 0.2 | 500.0 | none | 0.0540 | False | 0.0172 | False | False | 0.1266 | 0.0128 |
| euler | 0.5 | 500.0 | none | 0.0585 | False | 0.0105 | False | False | 0.1373 | 0.0183 |
| euler | 1.0 | 500.0 | none | 0.0604 | False | 0.0078 | True | False | 0.1422 | 0.0212 |
| euler | 2.0 | 500.0 | none | 0.0619 | False | 0.0067 | True | False | 0.1451 | 0.0229 |
| euler | 5.0 | 500.0 | none | 0.0644 | False | 0.0044 | True | False | 0.1468 | 0.0259 |
| rot_ab2 | 0.2 | 500.0 | none | 0.0542 | False | 0.0172 | False | False | 0.1266 | 0.0131 |
| rot_ab2 | 0.5 | 500.0 | none | 0.0588 | False | 0.0105 | False | False | 0.1373 | 0.0186 |
| rot_ab2 | 1.0 | 500.0 | none | 0.0608 | False | 0.0078 | True | False | 0.1422 | 0.0215 |
| rot_ab2 | 2.0 | 500.0 | none | 0.0624 | False | 0.0068 | True | False | 0.1451 | 0.0234 |
| rot_ab2 | 5.0 | 500.0 | none | 0.0651 | False | 0.0046 | True | False | 0.1469 | 0.0267 |
| euler | 0.2 | 2000.0 | none | 0.0540 | False | 0.0171 | False | False | 0.1267 | 0.0128 |
| euler | 0.5 | 2000.0 | none | 0.0585 | False | 0.0104 | False | False | 0.1374 | 0.0183 |
| euler | 1.0 | 2000.0 | none | 0.0605 | False | 0.0077 | True | False | 0.1422 | 0.0211 |
| euler | 2.0 | 2000.0 | none | 0.0619 | False | 0.0067 | True | False | 0.1451 | 0.0228 |
| euler | 5.0 | 2000.0 | none | 0.0645 | False | 0.0044 | True | False | 0.1468 | 0.0257 |
| rot_ab2 | 0.2 | 2000.0 | none | 0.0540 | False | 0.0171 | False | False | 0.1267 | 0.0128 |
| rot_ab2 | 0.5 | 2000.0 | none | 0.0586 | False | 0.0104 | False | False | 0.1374 | 0.0183 |
| rot_ab2 | 1.0 | 2000.0 | none | 0.0605 | False | 0.0077 | True | False | 0.1422 | 0.0212 |
| rot_ab2 | 2.0 | 2000.0 | none | 0.0620 | False | 0.0067 | True | False | 0.1451 | 0.0228 |
| rot_ab2 | 5.0 | 2000.0 | none | 0.0647 | False | 0.0045 | True | False | 0.1468 | 0.0258 |
| euler | 0.2 | 500.0 | real | 0.1566 | False | 0.1363 | False | False | 0.1812 | 0.1007 |
| euler | 0.5 | 500.0 | real | 0.0952 | False | 0.0571 | False | False | 0.1476 | 0.0526 |
| euler | 1.0 | 500.0 | real | 0.0760 | False | 0.0300 | False | False | 0.1449 | 0.0341 |
| euler | 2.0 | 500.0 | real | 0.0677 | False | 0.0170 | False | False | 0.1458 | 0.0268 |
| euler | 5.0 | 500.0 | real | 0.0655 | False | 0.0076 | True | False | 0.1469 | 0.0265 |
| rot_ab2 | 0.2 | 500.0 | real | 0.1566 | False | 0.1363 | False | False | 0.1811 | 0.1007 |
| rot_ab2 | 0.5 | 500.0 | real | 0.0952 | False | 0.0572 | False | False | 0.1476 | 0.0526 |
| rot_ab2 | 1.0 | 500.0 | real | 0.0761 | False | 0.0300 | False | False | 0.1449 | 0.0343 |
| rot_ab2 | 2.0 | 500.0 | real | 0.0680 | False | 0.0170 | False | False | 0.1458 | 0.0271 |
| rot_ab2 | 5.0 | 500.0 | real | 0.0662 | False | 0.0077 | True | False | 0.1470 | 0.0273 |
| euler | 0.2 | 2000.0 | real | 0.1566 | False | 0.1360 | False | False | 0.1812 | 0.1006 |
| euler | 0.5 | 2000.0 | real | 0.0952 | False | 0.0569 | False | False | 0.1477 | 0.0525 |
| euler | 1.0 | 2000.0 | real | 0.0760 | False | 0.0298 | False | False | 0.1449 | 0.0340 |
| euler | 2.0 | 2000.0 | real | 0.0677 | False | 0.0168 | False | False | 0.1458 | 0.0266 |
| euler | 5.0 | 2000.0 | real | 0.0656 | False | 0.0076 | True | False | 0.1469 | 0.0263 |
| rot_ab2 | 0.2 | 2000.0 | real | 0.1566 | False | 0.1360 | False | False | 0.1812 | 0.1006 |
| rot_ab2 | 0.5 | 2000.0 | real | 0.0952 | False | 0.0569 | False | False | 0.1477 | 0.0525 |
| rot_ab2 | 1.0 | 2000.0 | real | 0.0760 | False | 0.0298 | False | False | 0.1449 | 0.0340 |
| rot_ab2 | 2.0 | 2000.0 | real | 0.0678 | False | 0.0168 | False | False | 0.1458 | 0.0267 |
| rot_ab2 | 5.0 | 2000.0 | real | 0.0658 | False | 0.0076 | True | False | 0.1469 | 0.0265 |

## Per-profile breakdown (best real-stage, rate=500 Hz cell) — why "overall" fails

The per-profile-equal-weighted average above hides which profiles actually fail. `extreme_slip_pct` = % of that profile's samples with slip speed > 0.65 m/s (near-total wheel slip, larger than the platform's own p95 translational speed) — the regime no wheel-odometry+IMU front-end can track, by construction, not a filter defect.

(integrator=euler, crossover_hz=5)

| profile | rmse_pooled | pass | worst_file_max_error | extreme_slip_pct |
|---|---|---|---|---|
| octagon | 0.2308 | False | 1.1554 | 12.7706 |
| coupled_vomega | 0.2028 | False | 1.8724 | 9.1904 |
| spin_creep | 0.0265 | True | 0.3832 | 0.2734 |
| multisine_75percent_cap | 0.0186 | True | 0.0629 | 0.0000 |
| multisine_50percent_cap | 0.0177 | True | 0.0499 | 0.0000 |
| ellipse | 0.0133 | True | 0.0528 | 0.0000 |
| spiral_orbit | 0.0082 | True | 0.0352 | 0.0000 |
| long_circle | 0.0065 | True | 0.0381 | 0.0000 |

## Why "overall" fails: cumulative slip-bin inclusion (best real-stage, rate=500 Hz cell)

RMSE (per-profile-equal-weighted) as progressively higher slip-speed bins are folded into the pool, at the crossover/integrator with the lowest full-overall RMSE. This isolates whether a failing overall number reflects filter mistuning or a hard floor set by physically unrecoverable extreme-slip states (no wheel-odometry+IMU front-end can estimate velocity once a wheel is in near-total slip).

(integrator=euler, crossover_hz=5)

| slip_upper_bound | rmse_cumulative | passes |
|---|---|---|
| 0.0050 | 0.0055 | True |
| 0.0200 | 0.0076 | True |
| 0.0650 | 0.0128 | True |
| 0.2000 | 0.0155 | True |
| 0.6500 | 0.0239 | True |
| 1.5000 | 0.0502 | False |
| inf | 0.0655 | False |

## Euler vs rot_ab2 delta (overall RMSE, rot_ab2 - euler, m/s)

| crossover_hz | rate_hz | stage | delta_rot_ab2_minus_euler |
|---|---|---|---|
| 0.2000 | 500.0000 | none | 0.0003 |
| 0.2000 | 500.0000 | real | -0.0000 |
| 0.2000 | 2000.0000 | none | 0.0000 |
| 0.2000 | 2000.0000 | real | -0.0000 |
| 0.5000 | 500.0000 | none | 0.0003 |
| 0.5000 | 500.0000 | real | 0.0000 |
| 0.5000 | 2000.0000 | none | 0.0000 |
| 0.5000 | 2000.0000 | real | 0.0000 |
| 1.0000 | 500.0000 | none | 0.0004 |
| 1.0000 | 500.0000 | real | 0.0001 |
| 1.0000 | 2000.0000 | none | 0.0001 |
| 1.0000 | 2000.0000 | real | 0.0000 |
| 2.0000 | 500.0000 | none | 0.0005 |
| 2.0000 | 500.0000 | real | 0.0003 |
| 2.0000 | 2000.0000 | none | 0.0001 |
| 2.0000 | 2000.0000 | real | 0.0001 |
| 5.0000 | 500.0000 | none | 0.0007 |
| 5.0000 | 500.0000 | real | 0.0006 |
| 5.0000 | 2000.0000 | none | 0.0002 |
| 5.0000 | 2000.0000 | real | 0.0001 |

## mech_only drift rate (anchor OFF, stage=real, m/s per s — unbounded-drift reference)

| integrator | rate_hz | drift_rate |
|---|---|---|
| euler | 500.0000 | 0.0706 |
| euler | 2000.0000 | 0.0699 |
| rot_ab2 | 500.0000 | 0.0696 |
| rot_ab2 | 2000.0000 | 0.0696 |

## odom_only slip-binned RMSE (anchor alone — why the accel path is mandatory)

| slip_bin | rmse_combined |
|---|---|
| [0,0.005) | 0.0025 |
| [0.005,0.02) | 0.0114 |
| [0.02,0.065) | 0.0462 |
| [0.065,0.2) | 0.0727 |
| [0.2,0.65) | 0.1385 |
| [0.65,1.5) | 0.4895 |
| [1.5,inf) | 0.7922 |
