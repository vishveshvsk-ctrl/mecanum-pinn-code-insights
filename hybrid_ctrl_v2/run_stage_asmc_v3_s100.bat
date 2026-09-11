@echo off
REM =============================================================================
REM run_stage_asmc_v3_s100.bat — ASMC v3 mirror with safety_margin = 1.0
REM =============================================================================
REM Supervisor-requested ablation: the (E56) factor of safety on the AVAILABLE
REM friction circle (kmax_schedule's gain ceiling) is removed (0.9 -> 1.0). Every
REM other setting is identical to run_stage_asmc_v3.bat (train14_v3, --metric v3,
REM lambda-chatter 0.36, eps floors, lam-psi-hi 60) so the comparison against
REM runs_asmc_v3 is a single-variable change.
REM
REM Output root is AUTO-SUFFIXED by run_stage.jl (--safety-margin != 0.9 appends
REM _s100), so this writes runs_asmc_v3_s100\ and can never touch the archived
REM 0.9 runs. The noisy stage's --warm-from below points at the SUFFixed tree
REM explicitly for the same reason.
REM
REM Usage:  run_stage_asmc_v3_s100.bat <clean|noisy> [nseeds]
REM         clean: 5 seeds (cold start, mirrors the campaign)
REM         noisy: 4 seeds (warm-start from THIS tree's own clean stage)
REM Run keep_awake.py alongside; Modern Standby kills idle compute mid-sweep.
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v3
set OUTS=%~dp0runs_asmc_v3_s100

REM No REM lines inside the for-block below: cmd parses the whole parenthesised
REM block up front and REM inside it emits spurious "'M' is not recognized"
REM errors that look like a real failure.
for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUTS%\seed%%S\asmc_v2_clean\best_config.json"
    start "asmc_v3_s100_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 1 --controller asmc --asmc-v2 --seed %%S ^
        --trajset-screen train14_v3 --trajset-full train14_v3 --noise %STAGE% ^
        --metric v3 --lambda-chatter 0.36 ^
        --eps-floor-xy 0.02 --eps-floor-psi 0.08 ^
        --lam-psi-hi 60.0 ^
        --safety-margin 1.0 ^
        --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUTS%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% ASMC v3 s100 %STAGE% seeds. Logs: %OUTS%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
