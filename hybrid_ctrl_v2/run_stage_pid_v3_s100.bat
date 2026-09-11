@echo off
REM =============================================================================
REM run_stage_pid_v3_s100.bat — PID-CT v3 mirror with safety_margin = 1.0
REM =============================================================================
REM Supervisor-requested ablation: the (E56) factor of safety on the AVAILABLE
REM friction circle (vcmd_limits' command gate, shared by the feedforward pass-
REM through) is removed (0.9 -> 1.0). Every other setting is identical to
REM run_stage_pid_v3.bat so the comparison against runs_pid_v3 is a
REM single-variable change.
REM
REM Only the CT variant is in scope for this ablation (PID-FB has no
REM feedforward; the supervisor's ask names V_ff). The gate technically also
REM shapes FB's correction, but the campaign's FB results stay at 0.9.
REM
REM Output root is AUTO-SUFFIXED by run_stage.jl (--safety-margin != 0.9 appends
REM _s100), so this writes runs_pid_v3_s100\ and can never touch the archived
REM 0.9 runs. The noisy stage's --warm-from below points at the SUFFIXED tree.
REM
REM Usage:  run_stage_pid_v3_s100.bat ct <clean|noisy> [nseeds]
REM         clean: 5 seeds (cold start)   noisy: 4 seeds (warm-start from own clean)
REM PARALLELISM: one process per seed; keep total Julia processes <= 10
REM (~1.9 GB/process; the limiter is the Windows commit limit).
REM Run keep_awake.py alongside -- Modern Standby kills idle compute mid-sweep.
REM =============================================================================
setlocal enabledelayedexpansion
set VARIANT=%1
set STAGE=%2
set NSEEDS=%3
if "%VARIANT%"=="" set VARIANT=ct
if "%STAGE%"==""   set STAGE=clean
if "%NSEEDS%"==""  set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_pid_v3
set OUTS=%~dp0runs_pid_v3_s100

REM No REM lines inside the for-block below: cmd parses the whole parenthesised
REM block up front and REM inside it emits spurious "'M' is not recognized".
for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUTS%\seed%%S\pid_v2_%VARIANT%_clean\best_config.json"
    start "pid_v3_s100_%VARIANT%_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 2 --controller pid --pid-v2 --pid-variant %VARIANT% --seed %%S ^
        --trajset-screen train14_v3 --trajset-full train14_v3 --noise %STAGE% ^
        --metric v3 --lambda-chatter 0.36 ^
        --safety-margin 1.0 ^
        --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUTS%_%VARIANT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% PID v3 s100 %VARIANT% %STAGE% seeds. Logs: %OUTS%_%VARIANT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
