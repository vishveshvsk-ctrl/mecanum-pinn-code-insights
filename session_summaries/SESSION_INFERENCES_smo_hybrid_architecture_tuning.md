# Session Inferences — Hybrid SMO-MPC-PID-Fuzzy Architecture Tuning

## Date
2026-07-22

## Key Topics Covered

1. Dynamics model comparison (v1 vs sonnet5 hybrid notebooks)
2. Pham & Han (2025) consolidated control architecture paper
3. SMO observer tuning using acceleration envelope bounds
4. Open-loop SMO validation with envelope-tuned gains

---

## 1. Hybrid Notebooks — Dynamics Model

Both `Mecanum_Hybrid_SMO_MPC_PID_Fuzzy_v1.ipynb` and `_sonnet5.ipynb` share the **same plant model**: 30-D (or 34-D) ODE with LuGre friction + Adamov coupling + roller kinematics for the KUKA youBot. The plant is identical to `run_one.jl`'s `dynamics_full_mf_asmc!`.

**Key differences between v1 and sonnet5:**

| Aspect | v1 | sonnet5 |
|---|---|---|
| SMO gains L | [20, 20, 15] | [15, 15, 15] |
| SMO integral K | [80, 80, 60] | [5, 5, 5] |
| SMO δ | 0.05 | 0.01 |
| Motor i_max | 12.8 A | 5.0 A |
| Max_torque | 6.0 N·m | 10.0 N·m |
| MPC constraints | voltage + current-limit + slew-rate | voltage-only |
| Default controller | ASMC-only | All three enabled (fuzzy-blended) |
| ASMC initial K | (0,0,0) | (6, 6, 24) |
| Sensor cpr | 4000 | 103,360 |

---

## 2. Pham & Han (2025) Paper — Conceptual Validation

The user's architecture was verified against the paper's approach:

| Aspect | Pham & Han | User's approach |
|---|---|---|
| Disturbance type | External (force impulses, parameter variations) | Internal (LuGre slip dynamics) |
| Disturbance source | `F_ext(t)` injected into plant | ``d = f_actual - f_nominal`` (slip-induced force error) |
| Disturbance bandwidth | Low (~1 Hz impulse train) | High (bristle τ ≈ 16 ms) |
| Plant mass | 4.5 kg | 35.6 kg |
| SMO gains | L=[15,15,15], K=[5,5,5] | L=[6,6,20], K=[9,9,100] (per envelope tuning) |

**Conclusion:** The user's interpretation is correct and justifiable — the nominal no-slip model serves as the ASMC baseline, and the LuGre friction-induced slip is treated as a disturbance to be rejected. The user's problem is fundamentally harder because the disturbance is state-dependent and higher-bandwidth.

---

## 3. SMO Tuning from Acceleration Envelope

From `docs/Mecanum_Analytical_Limits_AxisVel_AccelEnvelope.tex`, the acceleration caps at rest are:

| Axis | Max accel | Recommended L (2× margin) | K (critical damping L²/4) |
|---|---|---|---|
| Vx | 2.93 m/s² | 6 | 9 |
| Vy | 2.86 m/s² | 6 | 9 |
| ψ̇ | 9.62 rad/s² | 20 | 100 |

**v1 L=[20,20,15] was wrong** — Vx/Vy over-gained by 3.3×, ψ under-gained by 1.3×.

**Envelope-tuned L=[6,6,20]** is physically principled: gains proportional to per-axis acceleration capacity.

---

## 4. SMO Validation Results

Open-loop validation (4 s multisine voltage command, 1000 Hz SMO rate):

| Metric | Vx | Vy | ψ̇ |
|---|---|---|---|
| RMSE (after 0.2s) | 0.0002 m/s | 0.0002 m/s | 0.0043 rad/s |
| Max error | 0.0008 m/s | 0.0007 m/s | 0.0123 rad/s |

**Convergence:** Instantaneous (error below threshold from first sample). Both v1 and envelope-tuned gains produce equivalent tracking quality, but envelope-tuned gains use 70% less control authority on translation.

---

## 5. Key Decisions for the Paper

1. **SMO over Kalman:** The SMO handles slip-induced bias in the wheel-speed pseudo-measurement via its switching term. The Kalman filter would need IMU acceleration in the prediction step to handle slip events.

2. **Per-axis gain tuning:** SMO gains should be tuned per the acceleration envelope (L_x:L_y:L_ψ ≈ 1:1:3.3), not uniform — yaw has 3.3× the acceleration capacity of translation.

3. **SMO convergence is controller-independent:** Verified with open-loop voltage commands; convergence condition `L > |f_x|_max` is bounded by torque saturation, not the controller.

4. **Data pipeline:** The SMO uses only wheel-speed pseudo-measurement + gyro (no accelerometer). For slip-heavy maneuvers, IMU acceleration should be added to the Kalman prediction step.

---

## 6. Exteroceptive Localization for PosRef — Two-Tier Model (hospital micrologistics)

**Decision:** for **PosRef (position-tracked) cases, the position is provided to the estimator as a NOISY exteroceptive input** (an absolute pose fix), not left to pure dead reckoning. VelRef cases stay proprioceptive-only.

**Why:** with IMU + encoders only, global position and heading are **unobservable** — dead reckoning always works but drifts *unboundedly* (heading error dominates: a yaw bias rotates the velocity vector → position error grows ~`V·b_ψ·t²`). This is a structural observability limit, not a filter-quality gap. Under PosRef the dead-reckoned pose feeds back into the position controller, so its drift is load-bearing; a bounded pose therefore requires an absolute fix.

**Industry practice = prediction + correction.** Proprioceptive (odometry/IMU) gives the high-rate, smooth, short-term motion estimate; an exteroceptive/infrastructure fix gives the low-rate absolute anchor; an EKF/factor-graph fuses them. For **bounded** global pose over time, an exteroceptive fix (or engineered infrastructure) is effectively **compulsory**; for short-horizon/relative motion, dead reckoning alone is used and re-anchored periodically.

**Deployable sensor choice (indoor mecanum / youBot-class):** primary = **2D LiDAR map-localization (AMCL / scan-matching)** — gives full planar pose `(x,y,ψ)` incl. heading; secondary = **AprilTag/fiducial + camera** for intermittent fixes. (UWB = position only, no heading; motion capture = lab-only ground truth / validation, *not* the modeled deployable sensor.)

**Hospital micrologistics → two-tier model.** Hospitals are dynamic (crowds), feature-poor/ambiguous (corridors), reflective (glass), multi-floor, and safety-critical, so a single continuous fix is unrealistic — real hospital AMRs (TUG/Moxi/Relay-class) fuse LiDAR SLAM + depth cameras + odometry/IMU + waypoint infrastructure. Model the fix in two regimes:

| Tier | Sensor model | Rate | σ_pos | σ_ψ | Availability |
|---|---|---|---|---|---|
| **Transit** (corridors, dynamic) | degraded LiDAR-AMCL | ~5–10 Hz | 0.05 m (0.1–0.2 degraded) | 1–2° | **intermittent** — 10–30% dropout in dynamic/aliased segments, ~1% gross-outlier jumps |
| **Docking** (med room, elevator, lab drop-off) | close-range fiducial/vision | ~20–30 Hz | 0.01–0.02 m | 0.5–1° | reliable at the station |

Fix enters as a **low-rate correction on the estimator's pose states** `(X̂o,Ŷo,ψ̂)` with `R_ext = diag(σ_pos², σ_pos², σ_ψ²)`; the estimator still dead-reckons at 1 kHz between fixes. Behind config flags: `use_pose_fix`, `fix_rate_hz`, `sigma_pos`, `sigma_psi`, `dropout_frac`, `outlier_frac`, `latency_ms`. In sim the fix = ground-truth pose + noise sampled at the fix rate (mocap-grade truth is used to *generate* and *validate* it, never as the reported sensor).

**Experimental implications (resolves the pose-comparison confound):**
- **Velocity-tracking comparison in transit** — observable, control-law-faithful; the fix is intermittent so slip-robust dead reckoning keeps drift bounded between fixes (this is what *motivates* the good velocity estimator).
- **Pose-tracking comparison at docking** — a *reliable* fix exists there, so pose is bounded and the **3-controller pose comparison is unconfounded** (differences reflect control law, not shared drift). Directly comparable to Pham & Han's position-RMSE framing, now honestly (their implicit absolute pose measurement is made explicit).
- **Fix-dropout robustness experiment** — drop the absolute fix for N s (occlusion / corridor aliasing) and measure pre-re-anchor drift per controller+estimator (safety-relevant, showcases dead-reckoning quality).
- **Confidence-gated conservatism** — use KF covariance `P` / slip indicator to slow the robot / reduce authority when uncertainty is high (hospital safety behavior).

**Cross-refs:** folded into `instructions/estimator-rewrite-posref-logging.md` (exteroceptive pose-fix path + fix-dropout experiment) and consumed by `instructions/estimator-kf-smo-tuner.md`. Ties to the estimator-first plan: KF is the default state feed (accel-fused, adaptive-R, `P` for confidence), SMO/`d̂` is the disturbance/slip channel; all three controllers compared on a **frozen** estimator.

---

## Files Created

| Path | Description |
|---|---|
| `observer_validation_v1_envelope_tuned/validate_smo_v1.jl` | SMO validation script (envelope-tuned) |
| `observer_validation_v1_envelope_tuned/run.bat` | Windows batch launcher |
| `observer_validation_v1_envelope_tuned/01_velocity_tracking.png` | True vs SMO vs measurement |
| `observer_validation_v1_envelope_tuned/02_estimation_error.png` | Absolute error plots |
| `observer_validation_v1_envelope_tuned/03_convergence_zoom.png` | First 0.5s convergence |
| `observer_validation_v1_envelope_tuned/04_disturbance_estimate.png` | d̂_x, d̂_y, d̂_ψ |
| `observer_validation_v1_envelope_tuned/05_summary_dashboard.png` | 3×2 summary panel |
