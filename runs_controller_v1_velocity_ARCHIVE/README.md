# v1 velocity-mode controller results — FALLBACK ARCHIVE (do not overwrite)

Frozen snapshot (2026-07-24) of the **velocity-mode** ASMC/PID comparison — kept as
the fallback for an initial paper submission if the pose-mode redo isn't ready in time.

**Known limitation (why v2 pose-mode is being built):** the 4 velref trajectories were
tracked in velocity mode → only the 1-loop PID was exercised, and position/heading drift
unbounded (velocity+yaw-rate feedback only). See
`instructions/controller-posref-retune-reeval.md`.

## Contents
- `asmc_FINAL_seed3.json` — final ASMC gains (pinned K_max, seed 3)
- `pid_FINAL_seed2.json` — final PID gains (Kd_pos fix, seed 2)
- `RESULTS_controller_tuning.md` — Tables 1–3 + noise-model justification
- `noise_eval_10seed.csv` / `_summary.txt` — clean + 1×/2×/5× × 10 seeds (tracking, ce, chatter)
- `subset_manifest.json` — 6-traj training set + 15-traj eval grid
- `viz/*.png` — all 6 comparison figures

## Provenance (live copies, do NOT let v2 overwrite these)
- `runs_controller_asmc_pin/`, `runs_controller_pid_5seed/`
- `runs_controller/{RESULTS_controller_tuning.md, noise_eval_10seed.csv, subset_manifest.json, viz/}`

The v2 pose-mode work MUST write to NEW dirs (`runs_controller_asmc_pose_5seed/`,
`runs_controller_pid_pose_5seed/`, `noise_eval_pose_10seed.csv`) and must NOT edit the
velocity-mode code path (`default_trajs_3` / `--trajset 3`) — see the brief's preservation constraint.
