# Monitor brief: compare_controllers_eskf 10-seed run

**Task:** babysit a running Julia comparison job to completion, then verify + report. Implementation is DONE and validated; only the long run remains. Do NOT rewrite code unless a failure demands it.

## What's running
- Script: `code_insights/compare_controllers_eskf.jl` (ASMC/PID/MPC on a frozen ESKF, 2 subset variants).
- Job: 210 runs = 3 ctrl × [4 trajs (with_coupled) + 3 trajs (no_coupled)] × 10 seeds. ~1 hr, single-thread.
- Launched in background; stdout redirected to `code_insights/_tmp/compare_controllers_eskf_run.log` (Julia **block-buffers** this → it stays near-empty until flush/exit; empty log is NOT failure).
- Wake-lock: `keep_awake.py` running (was PID 5230) — leave it until the job ends, then it can be killed.
- Julia exe: `C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe`
- cwd for all commands: `code_insights/`

## How to check status (don't poll tightly)
- Alive? `tasklist | findstr julia` (expect one julia.exe, ~1.5–3 GB RSS).
- Progress = output files appearing (written per-variant at end of each loop):
  - `runs_controller_compare_eskf/with_coupled/{runs,summary}.{csv,arrow}` ← after first ~120 runs
  - `runs_controller_compare_eskf/no_coupled/{runs,summary}.{csv,arrow}` ← after all 210
- Done when BOTH `no_coupled/summary.csv` exists AND julia.exe is gone.

## On completion — verify then report
1. Both variants' `runs.csv` (per-seed rows) + `summary.csv` (mean±std by controller×trajectory) exist and are non-empty.
2. `runs.csv` sanity: `ok` column mostly true; `tracking`/`ce` finite; ~120 rows (with_coupled) and ~90 rows (no_coupled).
3. Report per variant: the **controller ranking** (the script prints it; or sort `summary.csv` by `tracking_mean`). Lower `tracking` = better. Expect ASMC ≫ PID ≫ MPC.
4. Flag MPC as NOT finalized (coarse config) — already tagged via `finalized` column.
5. Cross-ref: frozen-ESKF `tracking` vs the oracle-clean baseline (ASMC≈15, PID≈24 from `runs_controller/RESULTS_controller_tuning.md`) to state the oracle→real-estimator gap. Smoke seen so far: ASMC/octagon `tracking≈23–24` (worse than oracle, as expected).
6. Note `est_nrmse_*` is CONTEXT only (frozen-ESKF quality, closed-loop-coupled) — velocity channels good (vx≈0.04), pose channels drift (use_pose_fix=false); NOT a controller metric.

## Failure modes
- Julia gone but NO `no_coupled/summary.csv` → it crashed. Tail the log: `type _tmp\compare_controllers_eskf_run.log` (filter out `hybrid run`/`Activating`). Per-run failures are caught (warn + NaN row), so a crash means a top-level error (bad path / OOM). Re-launch with `compare_controllers_eskf.bat` if needed.
- OOM risk: keep it single-thread (`-t 1`); do NOT parallelize (machine commit-limited, max_parallel=2).

## Key facts (don't re-derive)
- `run_controller_on_estimator` injects a FRESH ESKF per (traj,seed) — determinism already verified (identical metrics on repeat).
- Two variants already verified: with_coupled=[octagon,spin_creep,coupled_vomega,spiral_orbit], no_coupled drops coupled_vomega; ellipse excluded; all velocity-mode.
- Do NOT touch the frozen ESKF or controller configs; no retuning; no figures (owning session does those).
