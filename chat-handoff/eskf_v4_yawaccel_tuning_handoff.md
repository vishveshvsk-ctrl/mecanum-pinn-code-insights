# ESKF v4 (yaw-accel state) tuning handoff

## Run context

- **Estimator**: NEW `ESKFEstimatorV3` (13-dim) in `hybrid_ctrl_v2/estimators_v3.jl` — `ESKFEstimatorV2` + yaw-acceleration state (state 13), fixing the structural yaw-rate tracking lag. Replay over pre-simulated Arrow files (no plant re-solve).
- **Trajectory set**: controller-tuning-aligned `:train12` from `hybrid_ctrl_v2/controller_tuning/trajsets.jl` — 12 `mu=0.5` trajectories (includes `ellipse_stress_tangent` c55 + `ellipse_stress_crab` c83).
- **Sensor suite**: `:realistic` (full IMU/encoder/flow/pose-fix noise + biases).
- **Search space**: 7 dims in `hybrid_ctrl_v2/estimator_tuning/param_space_v4.jl` — v3's 6 update-rate dims (`alpha_acc`, `alpha_yaw`, `grip_slip_scale`, `r_boost`, `pose_Qn_heading`, `slip_R_inflate`) + `q_alpha` (yaw-accel random-walk intensity, TUNABLE per user direction). All P0s + `pose_Qn_pos` + `P0_alpha` PINNED at physical priors (`PINNED_V4`).
- **Optimizer**: dxNES → BOBYQA staged (`optimize_staged`).
- **Seeds**: 5, launched in parallel.
- **Output root**: `hybrid_ctrl_v2/runs_estimator_v4_mu0p5_train12/`

## Results (5 seeds)

| Seed | Best score | Evals | Stop |
|------|-----------:|------:|------|
| 1    | 2.6670     | 166   | plateau |
| 2    | 2.6251     | 119   | plateau |
| 3    | 2.6287     | 118   | plateau |
| 4    | **2.5818** | 86    | plateau |
| 5    | 2.6306     | 166   | plateau |

**Best overall**: seed 4 (`best_score = 2.5818`). Score spread 3.3%.

## Cross-eval (5 configs × 3 common noise seeds 101–103 × train12+test, 300 replays, 0 fails)

- All 5 configs are **functionally identical** under common noise: train12 2.657–2.665, test 2.840–2.851; per-term metrics agree to 3–4 sig figs. Residual gain scatter (slip-policy dims) is irrelevant even held-out.
- **Generalization excellent**: test ≈ train12 (+7%); `rate_rmse` better on test than train12. No overfitting, no noise-realization luck (tuned scores match common-noise scores).
- Error budget now: `rate_rmse` 0.0025 (tol 1e-2, 4× under; was 0.12 in v2 — 50×), `vel_rmse` 0.0009 (at tol), `pos` 0.0032 / `heading` 0.0013 (both under tol). **Sensor-noise-limited, not model-limited.**
- Data: `runs_estimator_v4_mu0p5_train12/cross_eval/cross_eval_v4_results.json` (+ `_raw.json`).

## Gain consistency across seeds (the multi-modality verdict)

- `q_alpha`: **0.064–0.067 (±2%) — sharply identified.** Physical reading: α random-walk wanders ~8 rad/s² over a 1 s horizon.
- `alpha_yaw`: collapsed 3.87 (v3-wide) → 0.015–0.23 — the α state absorbed maneuver-scale yaw uncertainty; `q_yr` is back to sub-tick jitter margin only (predicted diagnostic, confirmed).
- `pose_Qn_heading`: 1e-8 (lower bound) in all seeds — decided; pin next iteration.
- Slip-policy dims (`grip_slip_scale` 2e-4..0.32, `r_boost`, `slip_R_inflate` to bound 100): still scattered, confirmed irrelevant by cross-eval. Slip events too rare on train12 + flow backstop. Needs a slip-heavy eval set (spin_creep / low-μ) if slip robustness must be pinned.

## The arc (what happened this session, in order)

1. **k3 identifiability analysis** of the v2 3-seed run: score spread 5% but wild gain scatter → 10 dims split into transient-invisible P0s, boundary-identified, and weak slip dims.
2. **Decisive cross-eval v2** (`cross_eval_mu0p5.jl`): P0_vel/yaw/slip bit-identical (confirmed flat); `P0_heading`/`P0_pos` NOT transient-only — they select a lock/no-lock basin via the pose-fix NIS gate (16.27). `(P0_heading, pose_Qn_heading)` compensating pair = the multi-modality mechanism. Seed 2's "best" was partly noise luck (each tuning seed also reseeds the sensor suite).
3. **BUG FIX — slip mean-reversion units** (`gauss_markov_q` in `estimators_v2.jl`): returned per-tick `dt/τ` for a continuous-time A diagonal (`F = I + dt·A` applies dt again) → effective τ ~60 s instead of 0.06 s (1000× under-decayed; slip still ~random walk). Now returns `1/τ`; fixes all three v2 variants (`estimators_v2.jl`, `_iae`, `_slipobs`). `q_slip` was already correct.
4. **Physical P0 priors**: P0 = prior variance of the true initial error; where the sim's draw statistics are known, v2's tuned optima matched them (P0_bias_acc ≈ (0.02)², P0_bias_gyro ≈ (0.003)²). Pruned space v3: 6 update-rate dims.
5. **v3 runs**: 8.84 (bounds 1e1) → 6.63 (bounds 1e2); `alpha_acc` pinned at bound, `alpha_yaw` interior 3.87 — q_scale trade exhausted (bias-variance invariant `e_lag·σ² = α·dt·R`).
6. **Yaw-accel state (ESKFEstimatorV3)**: type-2 tracker, zero steady-state ramp error; residual O(jerk) not O(accel). Untuned midpoint 3.22 < v3-wide tuned 6.63. Tuned 2.58. `q_alpha` tunable per user direction.
7. **5-seed v4 + cross-eval** (this file's tables).

## Score journey

27.8 (v2 tuned, 10-dim P0 space) → 6.63 (v3-wide, 6-dim update-rate) → 3.22 (V4 untuned midpoint) → **2.58 (V4 tuned)** ≈ sensor-noise floor.

## Entry points

- V4 tuning: `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5_v4.jl` (+ `_3seed.bat`; output root `runs_estimator_v4_mu0p5_train12/`)
- V4 cross-eval: `hybrid_ctrl_v2/estimator_tuning/cross_eval_mu0p5_v4.jl` (+ `.bat`)
- V3 tuning (superseded): `run_estimator_replay_mu0p5_v3.jl` (+ `.bat`; roots `runs_estimator_v3_mu0p5_train12/`, `runs_estimator_v3wide_mu0p5_train12/`)
- V2 cross-eval (historical): `cross_eval_mu0p5.jl` (root `runs_estimator_v2_mu0p5_train12/cross_eval/`)

## Files changed / added

- `hybrid_ctrl_v2/estimators_v2.jl` — **BUG FIX** `gauss_markov_q` returns continuous-time rate `1/τ` (+docstring). Also fixes `estimators_v2_iae.jl` / `estimators_v2_slipobs.jl` (shared `derive_process_noise`).
- `hybrid_ctrl_v2/estimators_v3.jl` — NEW `ESKFEstimatorV3` (13-dim, `A[3,13]=1`, `Q[13,13]=q_alpha`, `x[3]+=dt·x[13]`).
- `hybrid_ctrl_v2/estimator_tuning/param_space_v3.jl` — NEW 6-dim pruned space + `PINNED_V3` (alpha bounds widened to 1e2 mid-run).
- `hybrid_ctrl_v2/estimator_tuning/param_space_v4.jl` — NEW 7-dim space (+`q_alpha`), `PINNED_V4` (+`P0_alpha=0.25`).
- `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl` — added `build_estimator_v3`, `builder` kwarg on `run_and_log_replay_v2`, extended `build_estimator_v2` kw pass-through (`alpha_acc/alpha_yaw/grip_slip_scale/r_boost`).
- `hybrid_ctrl_v2/estimator_tuning/objective_v2.jl` — `decode` + `builder` kwargs (defaults preserve v2 callers).
- `run_estimator_replay_mu0p5_v3.jl` / `_v4.jl` (+ `.bat`s), `cross_eval_mu0p5.jl` / `_v4.jl` (+ `.bat`s) — NEW.

## Open items / next steps

- **Slip-state decay inconsistency — FIXED**: the slip *state* `x[10:11]` now mean-reverts per tick (`x[10:11] *= (1 − dt/τ_slip)` in the nominal propagation of `estimators_v3.jl`, `estimators_v2.jl`, `_iae`, `_slipobs`), matching the covariance-side OU model. Effect on seed4|test (common noise 101–103): score 2.848 → 2.880 (+1.1%), all terms within ±2% — negligible on this trajectory set (measurements dominate ŝ at 1 kHz), as predicted; kept for model consistency and honest `d_hat`. NOTE: v4 tuned gains predate this change; deltas are small enough that a retune is optional, not required.
- **Slip-policy dims unidentifiable on train12** — needs slip-heavy eval set (spin_creep / low-μ) if they must be pinned. `pose_Qn_heading` can be pinned at 1e-8.
- **`straightline` combo 1 Arrow missing** from `../data/Simulation_Data_MecanumSlipSpin_LugreAdamov` — `:test` tier evaluates 8/9 trajectories; generate it or accept the gap.
- **Physical τ_slip cross-check recorded**: δ* = μs/σ₀ ≈ 0.34 mm breakaway deflection → τ = δ*/s ≈ 34 ms at Stribeck speed 0.01 m/s, consistent with measured 60 ms. State-dependent decay (`dt·|ŝ|/δ*`) is the more faithful model if ever revisited.
- Adopted config: **seed 4** (`runs_estimator_v4_mu0p5_train12/seed4/best_config.json`).
- `q_alpha` was tuned, not derived; if a derived cross-check is wanted, yaw-jerk statistics live in the `*_accel.arrow` sidecars (`dpsidot`).
