# Handoff — IMM Kalman estimator + advanced optimizer tuning (2026-07-23)

## What this session did

Replaced the tuned legacy `KalmanEstimator` with a new **`IMMKalmanEstimator`**
(`:kalman_imm`) that bundles ALL proposed upgrades in one architecture (no ablation),
and tuned it with a real global optimizer instead of `CoarseThenLocal`.

### Estimator (`hybrid_ctrl/estimators.jl`, lines ~431-745)

2-model IMM EKF over a shared 10-dim state `x = [Vx,Vy,ψ̇,ψ,Xo,Yo,bx,by,sx,sy]`:

- **Slip states** `sx,sy` (random walk): wheel pseudo-measurement is `z = v + s`,
  gyro channel clean. `bus.d_hat = [sx, sy, 0]`.
- **True covariance propagation** `P ← F·P·F' + Q` with the linearized 10×10
  transition (replaces legacy `P + Q`), **Joseph-form** updates + symmetrization.
- **Coupled pose fix**: no Kalman-gain rows zeroed — exteroceptive fixes
  back-correct velocity through F-built cross-covariance.
- **Adaptive Q**: translational block × `(1 + α_acc·‖a‖ + α_yaw·|ψ̇|)`.
- **Adaptive R**: NIS-triggered (`ν > 9.21` ⇒ R × `r_boost`, one recompute).
- **IMM**: grip model (nominal R, slip_Qn×1e-3, bias correction on) vs slip model
  (R × slip_R_inflate, free slip walk, bias off); log-domain mode update with
  tunable stay probabilities; divergence guard (NIS > 1e4 × 10 ticks ⇒ reset to
  fused state, P×10).
- Slip-state prior at init is `P0×1e-3` — equal-variance init leaks ~28% of body
  velocity into the slip states (weak v/s observability). Do not "simplify" this.
- Legacy `:kalman` / `:smo` paths untouched (verified: solo legacy compare
  reproduces the archived 0.9217662777403691 to 16 digits).

### Optimizer (`tuning/optimizer_bbo.jl` + `tune_estimator.jl --optimizer`)

- Backends: `:dxnes` (BlackBoxOptim, primary), `:de`, `:bo` (Surrogates Kriging +
  DYCORS, SRBF fallback). Lazy `Base.require` — default `coarse` path needs no deps.
- `MaxFuncEvals = budget` (NOT `MaxSteps` — one dxNES step = whole population).
- **Degenerate-dim compression**: fixed dims (rate_hz, lower==upper) are removed
  from the search box and re-expanded before decode. Without this BBO emits NaN
  candidates and poisons its NES covariance (this killed smoke run #2).
- NIS penalty clamp: non-finite/failed evals → finite 1e6 (GP/population-safe).
- `NThreads` clamped to `Threads.nthreads()-1` (BBO requires NThreads < nthreads).
- Runtime `include` of the backend module needs `Base.invokelatest` (world age).

### Robustness (`rerank_topk.jl`)

Top-10 trials re-evaluated on held-out sensor seeds 43,44; re-ranked by
mean+std across seeds {42,43,44}. Winner = search-rank 1 (0.573 search,
0.894 robust) — the minimum is real, not seed-42 luck. Seed spread is ~25%
of mean — inherent, worth remembering when comparing future runs.

## Results (5-trajectory subset + 2 held-out, seeds 42,43)

| metric | legacy KF (budget-30) | IMM KF (budget-150 dxnes) |
|---|---|---|
| composite score | 0.922 | **0.865** |
| overall NRMSE | 0.305 | **0.274** |
| in-slip NRMSE | 0.314 | **0.280** |
| pose drift (ellipse) | 0.504 | **0.436** |
| docking pose RMSE | 1.37 | **0.71** |
| held-out long_circle NRMSE | 0.306 | **0.205** |
| smoothness (lower=better) | **0.50** | 0.92 |

Caveat: the IMM estimate is ~2× less smooth — the optimizer bought accuracy,
not damping. The ASMC sees the jitter. If controller tuning complains, raise
λ_smooth in the objective and re-tune from the current winner.

Winner frozen: `runs_estimator/frozen/best_config.json` (legacy backup:
`best_config_legacy_kalman.bak.json` in the same dir).

## Files

- New: `tuning/optimizer_bbo.jl`, `rerank_topk.jl`, `tune_kalman_imm_smoke.bat`,
  `tune_kalman_imm_dxnes.bat`, `rerank_topk.bat`, `compare_kalman_solo.bat`
- Modified: `hybrid_ctrl/estimators.jl`, `hybrid_ctrl/config.jl` (doc),
  `hybrid_ctrl/scheduler.jl` (default-constructor branch), `tuning/param_space.jl`
  (`imm_kf_param_space`, 17 dims), `tuning/harness.jl` (`_build_estimator` branch),
  `tune_estimator.jl` (`--optimizer`, `--estimator kalman_imm`),
  `compare_estimators.jl` (`load_frozen_estimator` :kalman_imm branch),
  `compare_estimators.bat` (now frozen-vs-kalman → `runs_estimator/compare_imm`),
  `Project.toml`/`Manifest.toml` (BlackBoxOptim 0.6.9, Surrogates 7.5.2 — user-installed)
- Outputs: `runs_estimator_imm/kalman_imm_dxnes/` (trials.arrow, best_config,
  diagnostics, validation.json), `runs_estimator_imm/subset_manifest.json`,
  `rerank_topk.log` + `runs_estimator_imm/kalman_imm_dxnes/rerank_results.json`,
  `runs_estimator/compare_imm/`, `runs_estimator/compare_kalman_solo/`

## Known issues / open threads

1. **Second-estimator-in-a-process aborts**: in joint compares, the estimator
   evaluated SECOND gets dt_epsilon aborts on ellipse (pose mode), manufacturing
   pose_drift ≈ 1571. Hit SMO in the original compare and legacy KF in the joint
   IMM compare. Solo runs are always clean. Root cause not found (RNG, probe-log,
   and solver-stack versions all ruled out). Workaround: one estimator per process.
   NOTE: this means the original "SMO scored 2462" comparison is suspect in its
   pose metrics (its velref 6.3 was genuinely bad though).
2. IMM runs ~2× costlier per tick than legacy KF — tuning evals are slower.
3. BBO `NThreads>1` uses MultithreadEvaluator; worked fine here (3 workers on -t4).

## Reproduce

```
tune_kalman_imm_dxnes.bat    # budget-150 dxnes, -t4, out runs_estimator_imm
rerank_topk.bat              # top-10 seeds 43,44 re-rank + freeze
compare_estimators.bat       # frozen(IMM) vs legacy, out runs_estimator/compare_imm
compare_kalman_solo.bat      # legacy-only reference (no cross-contamination)
```
