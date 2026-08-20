@echo off
REM =============================================================================
REM run_stage_asmc_v2_chatter.bat — ASMC v2 re-tune, CHATTER-PRICED
REM =============================================================================
REM The consistency half of the chatter-pricing change. Its PID counterpart is
REM run_stage_pid_v2_chatter.bat; both MUST use the same --lambda-chatter, because
REM a cross-controller comparison is only meaningful when every controller is
REM scored by the identical objective.
REM
REM WHY THIS EXISTS: no launch bat in this repo has EVER passed --lambda-chatter,
REM and run_stage.jl defaults it to 0.0 -- so runs_asmc_v2_epsfloor / _wide /
REM _wide2 were all tuned chatter-UNPRICED. That is precisely why tau_ceiling and
REM lam_psi_max kept railing at whatever box edge they were given (100->200->300
REM and 20->40->60): nothing in the objective priced the authority they were
REM buying. Compounding it, chatter was normalised by V_MAX=24 (a voltage) rather
REM than the 0.8 V/ms slew ceiling, making any value ~30x too weak even if set.
REM Both are now fixed; this is the first ASMC run where the term actually bites.
REM
REM BOX: the WIDE2 box (--tau-ceiling-hi 300 --lam-psi-hi 60) deliberately, NOT
REM the default 100/20. If chatter pricing works, the optimum should now be
REM INTERIOR to the widest box ever tried. If it STILL rails at 300/60 with the
REM term live, that is a strong, clean negative result rather than another
REM box-width argument. Cold start (no --warm-from): warm-starting from a rail
REM would bias the search toward it.
REM
REM eps floors 0.02/0.08 match the epsfloor/_wide/_wide2 series so the only
REM changed variable versus those runs is the chatter term.
REM
REM Usage:  run_stage_asmc_v2_chatter.bat <clean|noisy> [nseeds=5]
REM     run_stage_asmc_v2_chatter.bat clean
REM     run_stage_asmc_v2_chatter.bat noisy      (warm-starts from clean)
REM
REM Run keep_awake.py alongside. One process per seed; keep the combined count
REM across concurrent waves at or under 10 on this machine (16 cores, ~1.8 GB
REM per process against 40 GB total).
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v2_chatter

for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUT%\seed%%S\asmc_v2_clean\best_config.json"
    start "asmc_v2_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 1 --controller asmc --asmc-v2 --seed %%S ^
        --trajset-screen train12 --trajset-full train12 --noise %STAGE% ^
        --eps-floor-xy 0.02 --eps-floor-psi 0.08 ^
        --tau-ceiling-hi 300.0 --lam-psi-hi 60.0 ^
        --lambda-chatter 3.0 --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% ASMC v2 %STAGE% seeds. Logs: %OUT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
