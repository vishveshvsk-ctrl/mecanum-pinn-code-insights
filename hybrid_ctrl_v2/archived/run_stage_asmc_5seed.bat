@echo off
REM =============================================================================
REM run_stage_asmc_5seed.bat — Stage 1 launcher (5 independent julia processes)
REM =============================================================================
REM Process-level parallelism ONLY (brief §3): Profiles.ACTIVE_KIND is a global
REM Ref and SchedulerMod.ESTIMATOR_PROBE_LOG is an unlocked global Dict, so the
REM 5 seeds must be separate julia processes, never in-process threads.
REM
REM Usage: run_stage_asmc_5seed.bat [extra run_stage.jl args...]
REM Example: run_stage_asmc_5seed.bat --p1-cap 250 --p2-cap 60 --pin-kmax
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_controller_v2_asmc
set EXTRA=%*

for /L %%S in (1,1,5) do (
    start "asmc_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 1 --controller asmc --seed %%S --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ASMC seeds in the background. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
