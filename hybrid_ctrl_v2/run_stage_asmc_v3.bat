@echo off
REM =============================================================================
REM run_stage_asmc_v3.bat — ASMC v3 campaign: v3 trajectory set + v3 metric
REM =============================================================================
REM Supersedes run_stage_asmc_v2_demandk.bat. THREE things changed together, and
REM none of them is optional if the result is to be comparable:
REM
REM 1. TRAJECTORY SET  train12 -> train14_v3
REM    train12 contained two trajectories the hardware cannot fly:
REM    spiral_orbit_stress (85.4%% of no-load wheel speed, ~9%% of torque authority
REM    left) and ellipse_stress_tangent (76.2%%, and a = 4.00 against a documented
REM    a <= 0.8). spiral alone carried 72-81%% of the tracking term, so the tuner
REM    spent most of its budget on a trajectory no control law recovers. v3
REM    replaces both, adds the friction-compliant octagon twin pair, and holds
REM    every entry <= 67.7%% of no-load. See trajsets.jl's V3 change log.
REM
REM 2. TRACKING METRIC  v2 -> v3  (--metric v3)
REM    v2 scored FOUR samples out of 12k-60k per run (final + argmax, per
REM    channel), so a controller that oscillated mid-path but ended accurately
REM    scored identically to one that tracked cleanly. v3 adds a time-normalised
REM    integral term, a per-trajectory position scaler max(radius_var, pos_max),
REM    and a trigonometric heading error 2|sin(dpsi/2)|.
REM    THE METRIC CAN CHANGE THE ANSWER, so never mix v2 and v3 scores in one
REM    table. An earlier ASMC-vs-PID ranking flip was measured, but with an
REM    IMBALANCED v3 that scaled position and not heading (fixed since: the
REM    scaler now divides the whole tracking term). That flip is therefore NOT
REM    established and must be re-measured on v3-tuned configs of both
REM    controllers -- which is exactly what this run and run_stage_pid_v3.bat
REM    exist to produce. What IS measured on the corrected metric: PID-CT beats
REM    PID-FB by 4.1%% with a 0.2%% seed spread (runs_pid_v3).
REM
REM 3. CHATTER PRICE  3.0 -> 0.36  (LAMBDA_CHATTER_V3)
REM    MANDATORY, not a tuning choice. v3 tracking is 5-10x smaller than v2, so
REM    carrying 3.0 over leaves tracking at ~5%% of the score and the run would
REM    optimise chatter alone. 0.36 reproduces v2's 29.9%% tracking share at the
REM    converged operating point. stage_objective.jl REFUSES lambda > 1.44 under
REM    --metric v3 rather than let this pass silently -- the same failure once
REM    left every run chatter-unpriced for months.
REM    It is balance-preserving, NOT a re-derived exchange rate: the method that
REM    produced 3.0 fails its own v2 control here (returns 0.052 against a known
REM    0.91-9.5 bracket) because lam_psi_max is no longer a v2 trade-off lever
REM    under the corrected schedule. See LAMBDA_CHATTER_V3's docstring.
REM
REM COLD START, MANDATORY for the clean stage. --warm-from is NOT passed. Beyond
REM the usual argument, warm-starting across this change is meaningless: the
REM objective, the metric and the trajectory set all differ, so a prior optimum
REM is a point in a different problem. run_stage.jl now also REJECTS a warm-from
REM archive missing any key the current space requires.
REM
REM EVERY CONTROLLER MUST BE RE-TUNED under these settings before any comparison
REM is quoted -- PID-CT, PID-FB and MPC included. An ASMC-only v3 run compared
REM against v2-era PID numbers is not a comparison.
REM
REM Usage:  run_stage_asmc_v3.bat <clean|noisy> [nseeds=5]
REM Run keep_awake.py alongside; Modern Standby kills idle compute mid-sweep.
REM Budget at the observed pace: ~14 h per clean stage, ~3 h per noisy stage.
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v3

REM No REM lines inside the for-block below: cmd parses the whole parenthesised
REM block up front and REM inside it emits spurious "'M' is not recognized"
REM errors that look like a real failure.
REM The noisy stage warm-starts from THIS run's own clean stage -- same space,
REM same metric, same trajectory set, so the cross-change hazard does not apply.
for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUT%\seed%%S\asmc_v2_clean\best_config.json"
    start "asmc_v3_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 1 --controller asmc --asmc-v2 --seed %%S ^
        --trajset-screen train14_v3 --trajset-full train14_v3 --noise %STAGE% ^
        --metric v3 --lambda-chatter 0.36 ^
        --eps-floor-xy 0.02 --eps-floor-psi 0.08 ^
        --lam-psi-hi 60.0 ^
        --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% ASMC v3 %STAGE% seeds. Logs: %OUT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
