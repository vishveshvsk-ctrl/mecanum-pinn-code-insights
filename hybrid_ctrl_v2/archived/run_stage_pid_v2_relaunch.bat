@echo off
REM =============================================================================
REM run_stage_pid_v2_relaunch.bat — PID v2 re-tune on CURRENT code (5 seeds)
REM =============================================================================
REM The archived PID v2 scores (runs_pid_v2_psi_floor_0p01: FB 0.945-0.961;
REM runs_pid_v2: CT 0.578-0.601) predate the vcmd_limits restructure and the
REM acceleration-convention fix and are STALE: seed1's archived FB gains
REM re-evaluate to 2.2184 under current code vs 0.9480 archived
REM (_tmp/pid_v2_staleness_check.jl). This launcher re-tunes from scratch on
REM the current objective. Same protocol as the archived run: PID_SPACE_V2
REM (lam_inner_x/y/psi, psi floor 0.01), train12 for both phases, clean oracle.
REM
REM Usage: run_stage_pid_v2_relaunch.bat fb     (wave 1)
REM        run_stage_pid_v2_relaunch.bat ct     (wave 2, after wave 1 finishes)
REM Run one variant at a time -- 5 processes, within the <=8 parallelism cap.
REM =============================================================================
setlocal
set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=fb
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_pid_v2_relaunch

for /L %%S in (1,1,5) do (
    start "pid_v2_%VARIANT%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 2 --controller pid --pid-v2 --pid-variant %VARIANT% --seed %%S --trajset-screen train12 --trajset-full train12 --noise clean --p1-cap 250 --p2-cap 60 --out "%OUT%" %EXTRA% > "%OUT%_%VARIANT%_seed%%S.log" 2>&1
)

echo Launched 5 PID v2 %VARIANT% seeds in the background. Logs: %OUT%_%VARIANT%_seed1.log .. _seed5.log
endlocal
