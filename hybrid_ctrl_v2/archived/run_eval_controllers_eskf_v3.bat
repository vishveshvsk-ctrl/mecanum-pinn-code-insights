@echo off
REM =============================================================================
REM run_eval_controllers_eskf_v3.bat — 5 noise-seed eval of tuned controllers
REM through the tuned ESKF V3 (realistic sensor suite), train12, pose mode.
REM One julia process per noise seed (5 total, within the <=8 parallelism cap).
REM Usage: run_eval_controllers_eskf_v3.bat            (5 seeds)
REM        julia hybrid_ctrl_v2\eval_controllers_eskf_v3.jl --smoke   (preflight)
REM        julia hybrid_ctrl_v2\eval_controllers_eskf_v3.jl --aggregate
REM =============================================================================
setlocal
set SCRIPT=%~dp0eval_controllers_eskf_v3.jl
set OUT=%~dp0runs_controller_eskf_v3

for /L %%S in (1,1,5) do (
    start "eskf_eval_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --seed %%S --out "%OUT%" > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ESKF-V3 controller-eval seeds. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
