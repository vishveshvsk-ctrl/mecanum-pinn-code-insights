@echo off
REM =============================================================================
REM run_eval_v3_eskf.bat [tier] [outdir] — the 6 v3 configs through the tuned
REM ESKF on the :realistic sensor suite, noise seeds 101-105.
REM
REM   run_eval_v3_eskf.bat                              train14_v3 (default)
REM   run_eval_v3_eskf.bat test_v3 runs_eskf_v3_test    held-out tier
REM
REM 6 configs x N trajectories x 5 seeds. One julia process per noise seed
REM (5 total, within the <=8 parallelism cap). That cap bounds the JIT
REM type-inference transient, NOT steady-state RAM -- 12-wide once OOM-killed
REM 5 runs with 15.8 GB free. Do not recompute it from free headroom.
REM
REM Afterwards:  julia --project=.. eval_v3_eskf.jl --tier <tier> --out <outdir> --aggregate
REM Run keep_awake.py alongside.
REM =============================================================================
setlocal
set TIER=%1
set OUTDIR=%2
if "%TIER%"==""   set TIER=train14_v3
if "%OUTDIR%"=="" set OUTDIR=runs_eskf_v3_train14

set SCRIPT=%~dp0eval_v3_eskf.jl
set OUT=%~dp0%OUTDIR%

for /L %%S in (1,1,5) do (
    start "eskf_v3_%TIER%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --tier %TIER% --seed %%S --out "%OUT%" > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ESKF v3 eval seeds on tier %TIER%. Logs: %OUTDIR%_seed1.log .. _seed5.log
endlocal
