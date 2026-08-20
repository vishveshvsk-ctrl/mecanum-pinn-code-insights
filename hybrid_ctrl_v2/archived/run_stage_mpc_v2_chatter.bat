@echo off
REM =============================================================================
REM run_stage_mpc_v2_chatter.bat — MPC v2 re-tune, CHATTER-PRICED, tau_cl searched
REM =============================================================================
REM Third leg of the chatter-priced consistency set, alongside
REM run_stage_pid_v2_chatter.bat and run_stage_asmc_v2_chatter.bat. All three MUST
REM carry the same --lambda-chatter: the comparison being made is "at the SAME
REM level of chatter, which controller tracks better", so the weight only has to
REM be common, not optimal.
REM
REM WHAT CHANGED IN THE MPC DESIGN (all now derived, nothing hand-picked):
REM   Q_pose  = bryson_Q_pose(TOL.pos_max,   TOL.head_max,   tau_cl)   running
REM   Q_final = bryson_Q_pose(TOL.pos_final, TOL.head_final, tau_cl)   terminal
REM   R       = bryson_R(lim, motor)                                   physical
REM   S       = lambda_chatter / (dV_max/rate_hz)^2  = 0.75 at lc=3     shared
REM   P       = DARE(A_end, Bm, Q_final, R)                            falls out
REM
REM  - S is DERIVED from lambda_chatter rather than searched, so the QP prices slew
REM    internally at the same rate the objective prices chatter externally. MPC is
REM    the only controller with an internal optimiser that can simply be told;
REM    ASMC/PID reach the same trade numerically via the outer search.
REM  - P is now built from the TERMINAL tolerances, not the running ones. The
REM    objective's tracking metric demands 10x tighter final than max accuracy on
REM    both channels, and P previously inherited Q_pose's scale through the DARE --
REM    a 100x under-weighting of terminal error. Q_final >= Q_pose, so the
REM    terminal-cost decrease condition (hence stability) is preserved.
REM  - tau_cl is now SEARCHED (MPC_SPACE_V2, 3-D). It was frozen at (0.15,0.11,0.12)
REM    = PID v2's design mid-window, which the chatter-priced PID retune made stale:
REM    PID converged to lam_inner_psi ~0.0145 (FB) / ~0.0199 (CT), and tau_cl enters
REM    Q quadratically, so MPC's yaw-rate weight was 36-68x too heavy.
REM
REM Np GRID: --np-grid 15,30 (was hardcoded 10,15,20,30). Solve time is MEASURED
REM and rules nothing out -- 1.4/1.9/2.8/5.6 ms mean per tick for Np=10/15/20/30
REM against a 10 ms budget at 100 Hz (_tmp/mpc_solve_time.jl, taken with 5
REM competing processes, so upper bounds). Every horizon is admissible; {15,30}
REM brackets the control effect.
REM
REM COST -- READ BEFORE LAUNCHING: each Np gets its OWN full p1-cap/p2-cap budget,
REM so a seed runs len(np_grid) COMPLETE searches back to back. Measured eval cost
REM (_tmp/mpc_eval_time.jl) is 4.9 min at Np=15 and 7.7 min at Np=30 per 12-traj
REM eval, vs 3.3 min for ASMC. At the ~100-120 evals prior runs took to plateau
REM that is ~10 h + ~15 h = ~25 h per seed. Seeds run in parallel, so ~25 h wall.
REM
REM TO SHORTEN: pass a smaller --trajset-screen (e.g. `screen`, 4 trajectories) so
REM phase 1 explores cheaply and only phase 2 refines on train12 -- that is what the
REM two-tier design is for. NOT done by default here because PID and ASMC both ran
REM train12 for BOTH tiers, and changing it for MPC alone would make the search
REM protocol differ across the controllers being compared.
REM
REM Usage:  run_stage_mpc_v2_chatter.bat <clean|noisy> [nseeds=5]
REM Run keep_awake.py alongside.
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_mpc_v2_chatter

for /L %%S in (1,1,%NSEEDS%) do (
    start "mpc_v2_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 3 --controller mpc --seed %%S ^
        --trajset-screen train12 --trajset-full train12 --noise %STAGE% ^
        --np-grid 15,30 --lambda-chatter 3.0 ^
        --p1-cap 250 --p2-cap 60 --out "%OUT%" ^
        > "%OUT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% MPC v2 %STAGE% seeds. Logs: %OUT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
