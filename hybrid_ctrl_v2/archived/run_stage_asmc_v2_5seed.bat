@echo off
REM =============================================================================
REM run_stage_asmc_v2_5seed.bat — ASMC v2 re-launch (5 independent julia processes)
REM =============================================================================
REM Re-launch per instructions/asmc-v2-tuning-launch.md: the run in
REM runs_asmc_v2\ is VOID (K exceeded its friction-circle ceiling by up to
REM 11.7x -- growth gate unenforced). This launcher targets the FIXED controller
REM (4-D search: lam_x_max/lam_y_max/lam_psi_max/tau_ceiling; gamma FIXED at
REM derived per-axis values 170/380/280, gamma_ref=250) and a FRESH output root
REM (runs_asmc_v2_relaunch) so the void run is never overwritten.
REM
REM Process-level parallelism ONLY (brief: Profiles.ACTIVE_KIND is a global Ref
REM and SchedulerMod.ESTIMATOR_PROBE_LOG is an unlocked global Dict, so the 5
REM seeds must be separate julia processes, never in-process threads).
REM
REM Budget (launch brief section 4, "decision run"): p1-cap 250 / p2-cap 60 on
REM train12 for BOTH phases (per-user direction: single 12-trajectory training
REM set, no smaller screen tier). ~310 evals x ~2.7 min/eval under 5-seed
REM concurrency = order 14 h wall clock -- start keep_awake.py first.
REM
REM Usage: run_stage_asmc_v2_5seed.bat [extra run_stage.jl args...]
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v2_relaunch
set EXTRA=%*

for /L %%S in (1,1,5) do (
    start "asmc_v2_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 1 --controller asmc --asmc-v2 --seed %%S --trajset-screen train12 --trajset-full train12 --noise clean --p1-cap 250 --p2-cap 60 --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ASMC v2 seeds in the background. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
