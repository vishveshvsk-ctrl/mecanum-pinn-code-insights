@echo off
REM =============================================================================
REM run_stage_pid_v2_chatter.bat — PID v2 re-tune, RATE-MATCHED + chatter-priced
REM =============================================================================
REM Differences from run_stage_pid_v2_relaunch.bat (which this supersedes):
REM
REM  1. f_pid 100 -> 1000 Hz (HybridConfig default). PID is now rate-matched to
REM     the ASMC (pinned to f_est), so the controller comparison isolates the
REM     control LAW, not the update rate. MPC stays at 100 Hz -- its rate is set
REM     by QP solve time and is reported as a declared method constraint.
REM
REM  2. PID_SPACE_V2 lower bounds 0.05/0.05/0.01 -> 0.003 uniform. These are NO
REM     LONGER a design floor -- they are a numerical guard at 3*dt. The floor is
REM     supplied endogenously by the chatter term instead.
REM
REM  3. --lambda-chatter 3.0, ACTIVE ON BOTH STAGES. Chatter is now normalised by
REM     CHATTER_REF = 0.8 V/ms (the 4-wheel slew ceiling) rather than V_MAX = 24,
REM     so 3.0 is an exchange rate in score-per-unit-of-slew-ceiling. Calibrated
REM     from the ASMC box-widening series: the marginal trade collapsed from 9.5
REM     (epsfloor->wide, a good deal) to 0.91 (wide->wide2, a bad one), and 3.0
REM     is the geometric centre of that bracket.
REM
REM WHY BOTH STAGES: the clean stage is where the railing happened (every FB seed
REM pinned lam_inner_psi at its floor), and it is where chatter has the strongest
REM gradient. Noise independently pushes lambda UP (+1..57% clean->noisy on every
REM axis, all measured families), so the noisy stage is not expected to be
REM chatter-limited -- the term is carried there for consistency of the objective
REM across stages, not because it is expected to bind.
REM
REM Usage:  run_stage_pid_v2_chatter.bat <fb|ct> <clean|noisy> [nseeds=5]
REM
REM     run_stage_pid_v2_chatter.bat fb clean
REM     run_stage_pid_v2_chatter.bat fb noisy      (warm-starts from fb clean)
REM     run_stage_pid_v2_chatter.bat ct clean
REM     run_stage_pid_v2_chatter.bat ct noisy      (warm-starts from ct clean)
REM
REM PARALLELISM: one process per seed. The fb and ct families are INDEPENDENT
REM (ct does not warm-start from fb), so their `clean` waves may overlap -- keep
REM the combined process count at or under the 8 cap, e.g. fb x5 + ct x3. The
REM machine has 16 logical cores, so 8 single-threaded (BLAS pinned) processes
REM do not contend for CPU; the cap is a memory/commit-limit guard at ~1.7 GB
REM per process. A `noisy` wave must NOT overlap the `clean` wave it warm-starts
REM from -- --warm-from reads that run's best_config.json.
REM
REM Run keep_awake.py alongside -- Modern Standby kills idle compute mid-sweep.
REM =============================================================================
setlocal enabledelayedexpansion
set VARIANT=%1
set STAGE=%2
set NSEEDS=%3
if "%VARIANT%"=="" set VARIANT=fb
if "%STAGE%"==""   set STAGE=clean
if "%NSEEDS%"==""  set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_pid_v2_chatter

for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUT%\seed%%S\pid_v2_%VARIANT%_clean\best_config.json"
    start "pid_v2_%VARIANT%_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 2 --controller pid --pid-v2 --pid-variant %VARIANT% --seed %%S ^
        --trajset-screen train12 --trajset-full train12 --noise %STAGE% ^
        --lambda-chatter 3.0 --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUT%_%VARIANT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% PID v2 %VARIANT% %STAGE% seeds. Logs: %OUT%_%VARIANT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
