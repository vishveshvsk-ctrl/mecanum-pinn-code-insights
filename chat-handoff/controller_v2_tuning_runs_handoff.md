# Controller v2 Tuning Runs — Handoff

Continues a session that implemented ASMC v2 / PID v2 / MPC v2 (physically-
derived gain briefs) and wired PID v2 into the staged tuning pipeline, then
found and fixed two correctness bugs in that pipeline while trying to launch
real tuning runs. **This task: launch correct, complete tuning runs for all
three v2 controllers** (PID v2 FB + CT, ASMC v2, MPC v2) on the KUKA-youBot
Mecanum digital-twin project (IMECE 2026 paper).

## Context this task depends on

**Root:** `C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\`

**Pipeline files** (in `hybrid_ctrl_v2/controller_tuning/` unless noted):
- `run_stage.jl` — CLI driver (`run_asmc`/`run_pid`/`run_pid_v2`/`run_mpc`)
- `stage_objective.jl` — `StageObjectiveMod.make_stage_objective` (FIXED this session)
- `optimizer_stage.jl` — `StageOptimizerMod.optimize_staged` (FIXED this session)
- `trajsets.jl` — `TrajSetsMod.trajset`, tiers `:screen`/`:train_full`/`:test`/`:train12` (new)
- `pid_cascade.jl`, `mpc_design.jl` — v1/Stage-3 legacy, still used by `run_pid`/`run_mpc`
- `../tune_controller_v2.jl` — `build_controller_v2`, `PID_SPACE_V2`, `ASMC_SPACE_V2`, `MPC_SPACE_V2`, `default_physical_limits()`
- `../controllers_v2.jl` — `ControllerV2Mod`: `PIDControllerV2`, `ASMCControllerV2`, `MPCControllerV2`, `bryson_Q_pose`/`bryson_R`/`u_eq_horizon`/`terminal_cost`
- NEVER edit: `hybrid_ctrl/{controllers,estimators,sensors,scheduler,config}.jl`, `tune_controller.jl`, `tuning/*.jl`

**Solver** (never overridden by `run_controller_v2`): `FBDF()`, `reltol=1e-9`,
per-state `abstol` (bristle states `1e-10`, body-vel `1e-8`, rest `1e-7`),
`saveat_hz=500`, `dtmax=1e-3`. `:train12`'s 12 trajectories sum to
**404.33s** simulated time/pass (mean 33.7s each); ~11-15 min wall-clock per
eval at 3 noise replicates (16 logical cores, single-threaded per process,
`BLAS` pinned to 1 thread by `tune_controller.jl`).

**`:train12`** (added to `trajsets.jl`, additive — doesn't touch
`:screen`/`:train_full`/`:test`): `TRAIN_FULL` minus `docking_a`/
`docking_step`, plus `octagon_mid` (combo 2) and `octagon_stress_hdg30`
(combo 309, exact 15°→30° heading twin of `octagon_stress`/combo 206).
`multisine75_broadband` (combo 55) remains, inherited from `TRAIN_FULL` —
open question, see below.

**dxNES population size:** λ=8 for a 3-dim space (verified via
`BlackBoxOptim.DXNESOpt` introspection) — `trace.csv` "eval N" ≈ generation N/8.

## Purpose

Produce correct, converged tuning results (gains + scores + trace/checkpoint
artifacts) for PID v2 FB, PID v2 CT, ASMC v2, MPC v2 — 5 seeds each. Success
= `best_config.json` per (controller, variant, seed) with `stop_reason` of
`:plateau`/`:no_refine_gain` (not `:cap`), gains compared across seeds for
convergence, plus a note on clean-vs-noisy-oracle generalization.

## Key design decisions (already made)

1. **Objective now builds a fresh controller instance per (trajectory,
   replicate) call**, not once per eval. Old code reused one mutable
   `PIDControllerV2` across all 12 trajectories, leaking `prev_vcmd`/
   `vcmd_initialized` (feeds `vcmd_limits`'s rate limiter) across trajectory
   boundaries — order-dependent contamination. Fixed and verified
   (`_tmp/verify_state_leak_fix.jl`: per-trajectory scores now bit-identical
   regardless of list order). `prev_e`/`initialized`/`prev_e_pos`/
   `pos_initialized` are inert (Kd=Kd_pos=0 by IMC design) — not part of the
   real leak.
2. **BOBYQA phase now checks `plateau_reached` explicitly** (throws
   `NLopt.ForcedStop()`), instead of relying only on NLopt's xtol/ftol
   criterion, which measurably did not fire after the score had clearly
   flatlined (seed 2, first PID-FB run: plateaued at eval 30/60, ran to cap
   anyway). Verified on a synthetic quadratic (stopped at eval 19/60,
   `stop_reason=:plateau`).
3. **Noise-replicate averaging added** (`noise_replicates` kwarg,
   `_derive_noise_seeds(seed,k)`): averages the objective over `k`
   independent `:noisy`-oracle draws per trajectory, sub-seeds derived
   deterministically from the top-level seed. Default 1 = old behavior.
   CLI: `--noise-replicates N`.
4. **Checkpointing added**: `optimize_staged` takes `checkpoint_every`/
   `checkpoint_cb`; `run_stage.jl`'s `_make_checkpoint_cb` writes one rolling
   `checkpoint.json` (current + best decoded gains) every
   `CHECKPOINT_EVERY=10` evals.
5. **Process-level parallelism (1 seed per OS process), not thread-level.**
   `Get-Counter` confirmed each `julia.exe` correctly uses ~100% of one core
   (not oversubscribed); system total was only ~55-59% of 16 cores. The real
   bottleneck: each eval's `N_trajectories × N_replicates` sims run serially
   within one process. **Proposed, NOT implemented**: parallelize that inner
   loop via `Distributed.pmap` — not `Threads.@threads`, since
   `SchedulerMod.ESTIMATOR_PROBE_LOG` is a shared, non-thread-safe `Dict`.
6. **`--noise-replicates 5` was too slow** (~25-30 min/eval under 5-way
   contention, days-scale ETA) — killed, output deleted. `--noise-replicates
   3` was also killed before completing (~11.5 min/eval avg, first eval
   >8 min) — no usable noisy results exist yet.

## Open decisions / blocking relationships

- **ASMC v2 not reachable via `run_stage.jl` yet**: `stage_objective.jl`'s
  dispatch only special-cases PID v2 (`is_pid_v2`); `:asmc` always hits the
  old `Main.build_controller`. Needs an `is_asmc_v2` check before
  `ASMCControllerV2`/`ASMC_SPACE_V2` is actually tunable.
- **MPC v2 wired to the wrong (older) search**: `run_mpc` still uses
  `mpc_design.jl`'s Stage-3 machinery (2-dim ratio search, fixed-
  linearization DARE), not `MPC_SPACE_V2` (1-dim `S_scale`) +
  `bryson_Q_pose`/`bryson_R`/`terminal_cost`/`u_eq_horizon`. Needs a
  `run_mpc_v2` analogous to `run_pid_v2`.
- **All completed PID v2 FB clean-oracle results predate the state-leak
  fix** (`hybrid_ctrl_v2/runs_pid_v2/seed{1-5}/pid_v2_fb_clean/`) — treat
  as reference-only: seed1 `(0.1309,0.1119,0.0500)` score 1.7131; seed2
  `(0.1363,0.0937,0.0500)` 1.7126; seed3 `(0.1312,0.1029,0.0500)` 1.7060;
  seed4 `(0.1312,0.1015,0.0500)` 1.7051; seed5 `(0.1320,0.1018,0.0500)`
  1.7053. All 5 pinned `lam_inner_psi` at the box floor (0.05) — box
  `[0.05,0.155]` may need widening.
- **PID v2 CT never launched at all.**
- **Noisy-oracle re-tune scope undecided**: affordable `noise_replicates`,
  and whether to build the `Distributed` parallelization (decision 5) first.
- **`multisine75_broadband` in `:train12`**: intentional per current
  `trajsets.jl` (a different, older v1 6-trajectory subset excludes
  multisine entirely) — user hasn't said whether to keep it.
- Hand-back to parent thread: final gains/scores per controller, plus
  whether `:train12` becomes the permanent v2 training tier.

## Deliverables

1. `is_asmc_v2` routing in `stage_objective.jl` + verification.
2. `run_mpc_v2` in `run_stage.jl` using `MPC_SPACE_V2`/proper Bryson derivation.
3. Completed `best_config.json`/`trace.csv`/`checkpoint.json` under
   `hybrid_ctrl_v2/runs_pid_v2/` (or new `runs_asmc_v2`/`runs_mpc_v2` roots)
   for PID v2 FB (rerun), PID v2 CT, ASMC v2, MPC v2 — 5 seeds each.
4. Short comparison: gains across seeds, `stop_reason` distribution, any
   parameter pinned at a search-box bound.

## Conventions to respect

- **`_tmp/` for all throwaway scripts**: write `_tmp/name.jl`, run via
  `julia _tmp/name.jl > _tmp/name.log 2>&1`, read `_tmp/name.log` itself —
  never inline `julia -e`, never read the harness's external
  `AppData\Local\Temp\claude\...` task-output path (both bypass the
  pre-authorized `_tmp/` allowlist and trigger permission prompts).
- **`keep_awake.py`** (`C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe`)
  must run in the background before any long sweep.
- **Scheduling is unreliable this session**: `ScheduleWakeup` and
  `CronCreate` both failed to fire as scheduled — rely on background-task
  completion notifications plus on-demand status checks instead.
- **Never edit** the protected files above; extend via new types/methods in
  `hybrid_ctrl_v2/*.jl` only (multiple-dispatch extension pattern).
- **Verify before launching**: a `_tmp/`-based correctness check on a small
  trajectory subset caught both bugs above — do this before any full 5-seed
  background launch.
- User prefers direct, precise answers over hedged summaries — cite exact
  measured numbers, retract claims plainly when disproven.
