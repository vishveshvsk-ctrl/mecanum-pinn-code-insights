# ESKF v2 mu=0.5 / chi=0.005 re-tune handoff

> **SUPERSEDED** — the k3 analysis, decay bug fix, P0 pruning, and the yaw-accel
> `ESKFEstimatorV3` / v4 5-seed run (best 2.58) that grew out of this handoff's
> "next steps" are recorded in `chat-handoff/eskf_v4_yawaccel_tuning_handoff.md`.
> This file remains as the record of the v2 3-seed run.


## Run context

- **Estimator**: `ESKFEstimatorV2` replay over pre-simulated Arrow files (no plant re-solve).
- **Trajectory set**: controller-tuning-aligned `:train12` from `hybrid_ctrl_v2/controller_tuning/trajsets.jl` — 12 `mu=0.5` PoseRef trajectories.
- **Sensor suite**: `:realistic` (full IMU/encoder/flow/pose-fix noise + biases).
- **Search space**: 10 tunable dimensions in `hybrid_ctrl_v2/estimator_tuning/param_space_v2.jl`:
  - `P0_vel`, `P0_yaw`, `P0_heading`, `P0_bias_acc`, `P0_bias_gyro`, `P0_slip`, `P0_pos`
  - `pose_Qn_heading`, `pose_Qn_pos`, `slip_R_inflate`
- **Optimizer**: dxNES → BOBYQA staged (`optimize_staged` in `hybrid_ctrl_v2/controller_tuning/optimizer_stage.jl`).
- **Seeds**: 3 independent runs launched sequentially.
- **Output root**: `hybrid_ctrl_v2/runs_estimator_v2_mu0p5_train12/`

## Entry points

- Main tuning script: `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5.jl`
- Warm-refine script: `hybrid_ctrl_v2/estimator_tuning/warm_refine_mu0p5.jl`
- Batch launcher: `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5_3seed.bat`

A smoke-test fix was required: `TrajSetsMod` from `hybrid_ctrl_v2/controller_tuning/trajsets.jl` is now explicitly `include`d/`using`d in both entry points.

## Results

| Seed | Best score | Phase 1 evals | Phase 2 evals | Total evals | Stop reason |
|------|-----------:|--------------:|--------------:|------------:|-------------|
| 1    | 27.7938    | 121           | 48            | 169         | plateau     |
| 2    | 26.4260    | 181           | 52            | 233         | plateau     |
| 3    | 27.5575    | 91            | 27            | 118         | plateau     |

**Best overall**: seed 2 (`best_score = 26.4260`).

## Tuned gains per seed

| Gain               | Seed 1              | Seed 2              | Seed 3              |
|--------------------|--------------------:|--------------------:|--------------------:|
| `P0_vel`           | 4.88e-2             | 7.14e-1             | 8.13e-4             |
| `P0_yaw`           | 1.60e-4             | 1.28e-4             | 1.08e-1             |
| `P0_heading`       | 9.77e-1             | 6.23e-1             | 9.89e-1             |
| `P0_bias_acc`      | 3.57e-4             | 4.03e-4             | 6.48e-4             |
| `P0_bias_gyro`     | 4.09e-6             | 7.08e-6             | 6.96e-7             |
| `P0_slip`          | 5.23e-4             | 1.12e-2             | 1.80e-5             |
| `P0_pos`           | 2.58e-1             | 1.08e-3             | 2.81e-1             |
| `pose_Qn_heading`  | 3.77e-4             | 1.61e-5             | 1.17e-5             |
| `pose_Qn_pos`      | 1.04e-7             | 9.30e-8             | 1.42e-7             |
| `slip_R_inflate`   | 30.77               | 52.38               | 5.77                |

`use_dhat` is fixed to `false` for all seeds.

## Interpretation

- The objective value is relatively stable across seeds: score spread is only ~5% (26.4–27.8).
- The gain posteriors are **not** stable. Several parameters occupy very different regions across seeds, most notably `P0_vel`, `P0_pos`, `P0_slip`, `slip_R_inflate`, and `P0_yaw`.
- This pattern points to **multi-modality / weak identifiability** in the ESKF v2 tuning landscape on the `train12` set, rather than a single well-defined optimum.

## Next steps / open questions

- Investigate the cause of the spread using the k3 model (per user direction).
- Optional follow-ups:
  - Warm BOBYQA refine starting from seed 2 (`warm_refine_mu0p5.jl`).
  - Cross-evaluate the three seed configs on the held-out `:test` trajectory set.
  - Reduce the search space or add regularization if the k3 analysis shows redundant/ill-conditioned parameters.

## Files changed / added

- `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5.jl`
- `hybrid_ctrl_v2/estimator_tuning/warm_refine_mu0p5.jl`
- `hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_mu0p5_3seed.bat`
- `hybrid_ctrl_v2/estimator_tuning/warm_refine_mu0p5_3seed.bat`
- `hybrid_ctrl_v2/estimator_tuning/harness_v2.jl` — added `mu0p5_train12_replay_trajset()` and `t_window` support.
- `chat-handoff/eskf_v2_mu0p5_train12_retune_handoff.md` (this file).
