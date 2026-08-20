@echo off
REM =============================================================================
REM run_stage_pid_v3.bat — PID v2 (FB / CT) retune on the v3 set + v3 metric
REM =============================================================================
REM The PID half of the v3 campaign. Its ASMC counterpart is run_stage_asmc_v3.bat
REM and BOTH must use the identical objective -- a cross-controller comparison is
REM only meaningful when every controller is scored the same way. Three settings
REM are therefore locked to match it exactly:
REM
REM   --trajset-* train14_v3   feasibility-screened set. train12 held two
REM                            trajectories the hardware cannot fly
REM                            (spiral_orbit_stress at 85.4%% of no-load wheel
REM                            speed, ellipse_stress_tangent at 76.2%%), and
REM                            spiral alone carried 72-81%% of the tracking term.
REM   --metric v3              adds a time-normalised integral term, a
REM                            per-trajectory position scaler, and a
REM                            trigonometric heading error 2|sin(dpsi/2)|.
REM   --lambda-chatter 0.36    MANDATORY with v3. 3.0 is the v2-scale price and
REM                            would leave tracking at ~5%% of the score;
REM                            run_stage.jl refuses it under --metric v3.
REM
REM WHY PID MUST BE RE-TUNED, not just re-scored: on the converged v2 configs the
REM two metrics RANK THE CONTROLLERS DIFFERENTLY -- v2 puts PID-CT ahead of every
REM ASMC seed, v3 reverses it. The mechanism is a persistent x-axis offset in
REM PID-CT on coupled_vomega (rms 0.0869 vs ASMC 0.0053) that v2's final+max terms
REM could not see. Quoting v2-era PID numbers against v3-era ASMC numbers would be
REM comparing two different objectives, so the PID optimum has to be re-found
REM under the objective actually being reported.
REM
REM COLD START on `clean` (no --warm-from). The runs_pid_v2_chatter optima are
REM points in a different problem -- different metric, different chatter price,
REM different trajectory set -- so warm-starting from them would bias the search
REM toward an optimum of the objective being replaced.
REM
REM Usage:  run_stage_pid_v3.bat <fb|ct> <clean|noisy> [nseeds=5]
REM     run_stage_pid_v3.bat fb clean
REM     run_stage_pid_v3.bat ct clean
REM     run_stage_pid_v3.bat fb noisy      (warm-starts from THIS run's fb clean)
REM
REM PARALLELISM: one process per seed. fb and ct are independent (ct does not
REM warm-start from fb) so their clean waves may overlap, but that is 10 processes
REM against a documented cap of 8 -- check free memory first (~1.9 GB/process;
REM the limiter is the Windows commit limit, not core count, on 16 logical cores).
REM A noisy wave must NOT overlap the clean wave it warm-starts from.
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
set OUT=%~dp0runs_pid_v3

REM No REM lines inside the for-block below: cmd parses the whole parenthesised
REM block up front and REM inside it emits spurious "'M' is not recognized".
for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUT%\seed%%S\pid_v2_%VARIANT%_clean\best_config.json"
    start "pid_v3_%VARIANT%_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 2 --controller pid --pid-v2 --pid-variant %VARIANT% --seed %%S ^
        --trajset-screen train14_v3 --trajset-full train14_v3 --noise %STAGE% ^
        --metric v3 --lambda-chatter 0.36 ^
        --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUT%_%VARIANT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% PID v3 %VARIANT% %STAGE% seeds. Logs: %OUT%_%VARIANT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
