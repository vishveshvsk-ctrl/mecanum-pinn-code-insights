@echo off
REM =============================================================================
REM run_stage_mpc_5seed.bat — Stage 3 launcher (5 independent julia processes)
REM =============================================================================
REM Usage: run_stage_mpc_5seed.bat [extra run_stage.jl args...]
REM Example: run_stage_mpc_5seed.bat --p1-cap 250 --p2-cap 60
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_controller_v2_mpc
set EXTRA=%*

for /L %%S in (1,1,5) do (
    start "mpc_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 3 --controller mpc --seed %%S --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 MPC seeds in the background. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
