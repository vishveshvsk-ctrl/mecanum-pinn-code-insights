@echo off
REM =============================================================================
REM run_eval_v3_eskf.bat — RESULTS_v3.md §4 re-run with STATE FEEDBACK FROM THE
REM TUNED ESKF instead of the oracle. 6 configs x 14 train14_v3 trajectories x
REM 5 noise seeds (101-105) = 420 closed-loop runs.
REM
REM One julia process per noise seed (5 total, within the <=8 parallelism cap).
REM That cap bounds the JIT type-inference transient, NOT steady-state RAM --
REM 12-wide once OOM-killed 5 runs with 15.8 GB free. Do not recompute it from
REM free headroom.
REM
REM Usage:  run_eval_v3_eskf.bat              launch 5 seeds
REM         (then)  julia --project=.. eval_v3_eskf.jl --aggregate
REM Run keep_awake.py alongside.
REM =============================================================================
setlocal
set SCRIPT=%~dp0eval_v3_eskf.jl
set OUT=%~dp0runs_eskf_v3_train14

for /L %%S in (1,1,5) do (
    start "eskf_v3_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --seed %%S --out "%OUT%" > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ESKF v3 eval seeds. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
