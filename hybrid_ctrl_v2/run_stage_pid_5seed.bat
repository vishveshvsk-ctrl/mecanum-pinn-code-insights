@echo off
REM =============================================================================
REM run_stage_pid_5seed.bat — Stage 2 launcher (5 independent julia processes)
REM =============================================================================
REM Usage: run_stage_pid_5seed.bat [extra run_stage.jl args...]
REM Example: run_stage_pid_5seed.bat --p1-cap 250 --p2-cap 60
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_controller_v2_pid
set EXTRA=%*

for /L %%S in (1,1,5) do (
    start "pid_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 2 --controller pid --seed %%S --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 PID seeds in the background. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
