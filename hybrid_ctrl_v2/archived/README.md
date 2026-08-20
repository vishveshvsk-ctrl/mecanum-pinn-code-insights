# archived/ — superseded runs and code

Everything here is **superseded**, kept for provenance only. Nothing in this directory is
needed to reproduce any number in `../RESULTS_v3.md`.

The split was made by include-closure, not by hand: the four reproduction entry points

```
controller_tuning/run_stage.jl                       (controller tuning, v3)
estimator_tuning/run_estimator_replay_mu0p5_v4.jl    (estimator tuning, v4)
eval_v3_eskf.jl                                      (ESKF closed-loop result)
eval_v3_*.jl / diag_v3_*.jl                          (evaluation + diagnostics)
```

were traced transitively through their `include` statements; every `.jl` in the closure
stayed, everything else moved here. All four were re-run after the move and load clean.

## What is here

| group | why archived |
|---|---|
| `runs_asmc_v2*`, `runs_pid_v2*`, `runs_mpc_v2_chatter/` | v2-era controller tuning — different objective (`--metric v2`, `lambda_chatter 3.0`) and different trajectory tier (`train12`). Scores are NOT comparable with v3; see RESULTS_v3.md "Reading the numbers". |
| `runs_pid_v3_ABANDONED_imbalanced_metric/` | the 8.4 h run lost to the position-only `k_traj` scaler, which inverted the PID FB/CT ranking. Cited as evidence in RESULTS_v3.md §1.2. |
| `runs_estimator_v2*`, `runs_estimator_v3*` | estimator generations superseded by v4 (`ESKFEstimatorV3`, 13-dim yaw-accel). v4 seed4 is the frozen observer used everywhere in §7. |
| `runs_controller_eskf_v3/` | the 2026-08-09 ESKF comparison: **v2-era controllers, `train12`, v2 metric with no chatter term**. Superseded by `runs_eskf_v3_train14/`. Its `asmc score_std = 29.7` is the `train12` infeasibility signature that `train14_v3` exists to remove. |
| `estimator_tuning/*_iae*`, `*_slipobs*` | alternative estimator channels (IAE adaptive-Q, slip observer SMO/ESO) — explored, not adopted. |
| `estimator_tuning/param_space_v2*`, `run_estimator_replay{,_mu0p5,_mu0p5_v3}.jl`, `warm_refine*` | earlier search spaces and runners for the superseded estimator generations. |
| `estimator_tuning/cross_eval_mu0p5.{jl,bat}` | v2/v3 cross-eval. The **v4** cross-eval was kept — RESULTS_v3.md §7.3's test-tier row reads `cross_eval/cross_eval_v4_results.json`. |
| `*.log` | console logs, 0.2–4.5 MB each of progress-bar output. The results they record live in the run dirs' `best_config.json`. |
| `run_stage_*_v2*.bat`, `run_stage_{asmc,pid,mpc}_5seed.bat` | launchers for the above. |
| `controller_tuning/analyze_seeds.jl` | seed-analysis helper, not on any reproduction path. |

## Two files that look archivable but were NOT

- `../eval_controllers_eskf_v3.jl` — its `run_closed_loop_eskf` is `include`d by
  `eval_v3_eskf.jl`, so the ESKF sensor/estimator wiring stays byte-identical to the
  estimator's own tuning conditions. Its **launcher** (`run_eval_controllers_eskf_v3.bat`)
  is archived, since the run it drove is superseded — the file is now a library.
- `../estimator_tuning/param_space_v3.jl` — `param_space_v4.jl` builds `PINNED_V4` from
  `Main.ParamSpaceV3Mod.PINNED_V3`, so v4 does not load without it.

## Restoring

These are plain `git mv`-able directories; nothing was rewritten. To bring one back, move it
up one level. Note that v2-era run dirs will not reproduce under the current code without
also restoring their launchers and checking the metric flags in their `best_config.json`.
