# Controller tuning results — matched whitelisted 6-trajectory set (trajset3), clean oracle, dxNES

Training set: octagon(206), spin_creep(178), coupled_vomega(12), spiral_orbit(37), ellipse_tangent(55), ellipse_crab(83).


## Table 1 — ASMC 5-seed (pin_kmax=150/150/300, lambda_gamma=2.0, budget 50). FINAL = seed 3.

| seed | score | gamma_x | gamma_y | gamma_psi | eps | eps_psi | K_max(x,y,psi) |
|---|---|---|---|---|---|---|---|
| 1 | 15.421 | 4.88 | 20.05 | 84.54 | 0.0752 | 0.0677 | 150/150/300 |
| 2 | 15.115 | 6.97 | 4.55 | 98.22 | 0.0575 | 0.0643 | 150/150/300 |
| 3 ★ | 15.133 | 5.21 | 9.11 | 97.91 | 0.0567 | 0.0863 | 150/150/300 |
| 4 | 15.737 | 10.90 | 23.95 | 81.89 | 0.0786 | 0.0835 | 150/150/300 |
| 5 | 15.306 | 1.09 | 30.33 | 99.94 | 0.0742 | 0.1227 | 150/150/300 |

_Identified: gamma_psi (~90), eps/eps_psi. Don't-cares: gamma_x/gamma_y (cv 54-55%). K_max pinned (max headroom)._

## Table 2 — PID 5-seed (Kd_pos PD-outer fix, raised bounds, budget 150). CANDIDATE = seed 2.

| seed | score | Kp[x,y,psi] | Ki[x,y,psi] | Kd[x,y,psi] | Kp_pos[x,y,psi] | Kd_pos[x,y,psi] |
|---|---|---|---|---|---|---|
| 42 | 26.735 | [157.0, 397.0, 800.0] | [0.8, 0.7, 1.5] | [0.11, 0.35, 8.99] | [1.42, 4.96, 1.01] | [1.09, 0.89, 2.2] |
| 1 | 24.023 | [164.0, 360.0, 771.0] | [0.6, 0.2, 142.4] | [1.93, 1.64, 1.57] | [0.4, 0.27, 0.51] | [1.91, 0.19, 1.78] |
| 2 ★ | 23.809 | [166.0, 317.0, 787.0] | [2.7, 3.1, 130.4] | [3.66, 2.98, 4.22] | [1.99, 1.8, 0.89] | [2.1, 0.21, 0.77] |
| 3 | 25.780 | [175.0, 281.0, 800.0] | [4.9, 16.9, 2.9] | [3.48, 0.67, 0.63] | [0.78, 3.18, 4.8] | [0.24, 0.9, 0.79] |
| 4 | 25.241 | [344.0, 472.0, 799.0] | [0.8, 0.1, 149.1] | [1.65, 5.07, 2.26] | [0.74, 0.85, 4.68] | [0.55, 2.8, 0.29] |

_Score cv 4% (reproducible); gains cv up to 154% (non-identifiable). Only Kp_psi identified (~800, ceiling — yaw authority-starved)._

## Table 3 — Clean per-trajectory comparison: ASMC(seed3) vs PID(seed2)

| trajectory | mode | ASMC | PID(seed2) |
|---|---|---|---|
| octagon | vel | Vx 3.7 / Vy 8.8 mm/s, w 5.4 mrad/s | Vx 20.4 / Vy 26.0 mm/s, w 14.9 mrad/s |
| spin_creep | vel | Vx 1.4 / Vy 1.8 mm/s, w 4.9 mrad/s | Vx 6.3 / Vy 10.1 mm/s, w 57.8 mrad/s |
| coupled_vomega | vel | Vx 9.2 / Vy 26.2 mm/s, w 7.0 mrad/s | Vx 6.2 / Vy 29.1 mm/s, w 19.0 mrad/s |
| spiral_orbit | vel | Vx 135.9 / Vy 4.8 mm/s, w 47.2 mrad/s | Vx 90.8 / Vy 9.6 mm/s, w 132.9 mrad/s |
| ellipse_tangent | pose | posMax 6.1 cm, headMax 0.040 rad | posMax 10.4 cm, headMax 0.058 rad |
| ellipse_crab | pose | posMax 0.1 cm, headMax 0.001 rad | posMax 1.4 cm, headMax 0.007 rad |

_ASMC aggregate tracking ~14.5 vs PID ~24-25. ASMC dominates yaw (spin_creep ~12x) + pose (crab ~40x)._
## Noise model — physical justification (1× level)

Per-channel: `measured = true + bias + (σ0 + SF·|true|)·N(0,1)`. The 1× level models a **realistic, deployable indoor-AMR sensor suite** (mid-grade MEMS IMU + wheel odometry + LiDAR-SLAM/AMCL localization — the hospital-micrologistics class).

| channel | sensor | σ0 (white) | SF | bias | basis |
|---|---|---|---|---|---|
| ψ̇ (yaw rate) | MEMS gyro | 3 mrad/s | 0.5% | 3 mrad/s | ARW 0.01°/s/√Hz × √100 Hz ≈ 1.7 mrad/s (rounded up); SF 0.1–1% typical; bias = modest turn-on (~0.17°/s) |
| Vx,Vy | wheel odometry | 10 mm/s | 2% | 5 mm/s | encoder quant/jitter ~mm/s; scale = wheel-radius/roller-compliance error (1–3%); bias = radius mismatch |
| ψ (heading) | gyro-int + fix | 10 mrad (~0.6°) | — | 5 mrad | short-term drift between absolute corrections (mag/fix) |
| Xo,Yo | exteroceptive fix | 20 mm (2 cm) | — | 1 cm | LiDAR-SLAM/AMCL class (indoor AMR); additive (absolute accuracy ~constant, not ∝ distance) |

Scale ladder: **1× = realistic deployed AMR; 2× = degraded/cheaper suite; 5× = stress envelope** (pos σ=10 cm ≈ UWB/poor-SLAM, gyro 15 mrad/s — beyond typical, shows the breakdown boundary, not an operating point).
