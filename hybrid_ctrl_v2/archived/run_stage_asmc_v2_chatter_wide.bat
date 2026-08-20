@echo off
REM =============================================================================
REM run_stage_asmc_v2_chatter_wide.bat — ASMC v2, WARM-STARTED, tau_ceiling widened
REM =============================================================================
REM PREPARED, NOT AUTO-LAUNCHED. Gate (per-user): launch only once 4 seeds of
REM run_stage_asmc_v2_chatter.bat have converged AND the 5th is still far from
REM converging -- otherwise wait for the 5th rather than discarding it.
REM
REM WHAT MOVES AND WHAT DOES NOT:
REM   tau_ceiling  hi 300 -> %TAUHI% (default 3000)   WIDENED
REM   lam_psi_max  hi 60                              UNCHANGED -- deliberately
REM
REM WHY lam_psi_max IS LEFT ALONE: it is NOT saturated. Across the 5 seeds it
REM converged to 17.2-46.6 against a 60 ceiling (29-78% of box), spread widely,
REM no seed near the edge. That is the chatter term doing its job -- in every
REM PRIOR run (epsfloor/wide/wide2) lam_psi_max pinned at whatever edge it was
REM given (20, 40, 60). Widening a box that is not binding would only dilute
REM search resolution.
REM
REM WHY tau_ceiling NEEDS WIDENING: 3 of 5 seeds pinned at 299-300 (a 4th at 93%).
REM Chatter is NOT what stops it -- measured 0.1297 = 16.2% of the 0.80 V/ms cap,
REM leaving 2.51 of the 3.00 penalty unused. The coupling is cube-root
REM (K_eq ~ tau_ceiling^(1/3)), so a 3x box buys only 1.44x on the settled
REM switching gain: the chatter cost of railing tau_ceiling is small while the
REM robustness benefit of high settled K is not. Hence the rail.
REM
REM WHY 3000 AND NOT ANOTHER 2x: the prior series went 100 -> 200 -> 300 and
REM railed every time, because each step moves K_eq by only 1.26x / 1.14x. 10x
REM moves K_eq by 2.15x -- enough to actually reach an interior optimum instead of
REM buying one more rail. Overridable: pass the ceiling as arg 3.
REM   _tmp/tau_ceiling_sweep.jl measures chatter vs tau_ceiling at FIXED lam, which
REM   is the principled way to set this bound; prefer its answer over 3000 if it
REM   has run.
REM
REM Warm start seeds the search AT the previous rail, so the optimizer explores
REM outward from it rather than re-deriving the whole basin -- phase 1 is skipped
REM entirely (the wide2 precedent ran phase1=0, phase2=31).
REM
REM Usage:  run_stage_asmc_v2_chatter_wide.bat <clean|noisy> [nseeds=5] [tau_hi=3000]
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
set TAUHI=%3
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5
if "%TAUHI%"==""  set TAUHI=3000.0

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set IN=%~dp0runs_asmc_v2_chatter
set OUT=%~dp0runs_asmc_v2_chatter_wide

for /L %%S in (1,1,%NSEEDS%) do (
    start "asmc_v2_wide_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 1 --controller asmc --asmc-v2 --seed %%S ^
        --trajset-screen train12 --trajset-full train12 --noise %STAGE% ^
        --eps-floor-xy 0.02 --eps-floor-psi 0.08 ^
        --tau-ceiling-hi %TAUHI% --lam-psi-hi 60.0 ^
        --lambda-chatter 3.0 --p1-cap 125 --p2-cap 60 ^
        --warm-from "%IN%\seed%%S\asmc_v2_clean\best_config.json" ^
        --out "%OUT%" > "%OUT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% ASMC v2 WIDE %STAGE% seeds (tau_ceiling hi=%TAUHI%). Logs: %OUT%_%STAGE%_seed1.log ..
endlocal
