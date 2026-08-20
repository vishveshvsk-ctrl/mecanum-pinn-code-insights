@echo off
REM =============================================================================
REM run_stage_asmc_v2_demandk.bat — ASMC v2 re-tune, DEMAND-DRIVEN GAIN LAW
REM =============================================================================
REM Joint (lam_x, lam_y, lam_psi, rho_auth) re-tune against the corrected
REM schedule. Supersedes run_stage_asmc_v2_chatter.bat, whose converged gains are
REM now stale: they were fitted against a schedule that fed ACTUATOR-side W_ff
REM into a CONTACT-side friction gate, so every lam in runs_asmc_v2_chatter was
REM converged against a ceiling that sat BELOW the resting gain on 8/12
REM trajectories. That is why this is a re-tune and not an evaluation.
REM
REM WHAT CHANGED versus runs_asmc_v2_chatter (all four frozen ON in run_stage.jl,
REM not optional and not independent):
REM   kmax_contact_b   (E54)'s contact-side b-vector -- bare mass, contact drags
REM                    only -- instead of W_ff. Fixes a 3.6x (y) / 17.4x (psi)
REM                    drag overstatement and an entirely spurious 176*Vx term on
REM                    x; median |F_par3_ff| was 1.19x the ENTIRE remaining
REM                    friction budget while wheel-torque saturation measured
REM                    0.0% of ticks.
REM   use_demand_k     dK = gamma[ s*tanh(s/eps)*(1-z) - rho*eps*z ],
REM                    z = (K-K_floor)/(Kbar-K_floor). Two-sided, so K no longer
REM                    ratchets to whatever the ceiling is. Fixing the schedule
REM                    ALONE made things worse (mean tracking 0.7404 -> 1.9925,
REM                    spiral 6.78 -> 21.97) precisely because a correct ceiling
REM                    still gets parked at; this is what stops the parking.
REM   enforce_k_floor  K >= K_floor at all times (IC and a post-update projection).
REM   kmax_sched_floor keeps z's denominator off its degenerate floor.
REM
REM RHO BOX: log[0.762, 27], both ends physically anchored -- NOT a bracket around
REM degenerate limits. lo = tanh(1): at one boundary layer (the width the
REM controller is designed to hold) the adaptation claims at most half its
REM authority. hi = 9*3: 3*eps is the excitation scale the as-designed sigma leak
REM itself uses, and z*(3eps) spans 0.9..0.1 across the box. Derivation and the
REM measurement behind it are in run_stage.jl's run_asmc_v2. Override with
REM --rho-lo / --rho-hi only with a reason.
REM
REM COLD START, MANDATORY. --warm-from is NOT passed and must not be added for the
REM clean stage. ASMC_SPACE_V2 dimension 4 changed meaning (tau_ceiling log[0.1,
REM 100] -> rho_auth log[0.762, 27]), so a warm start from ANY pre-swap archive
REM would map that archive's dimension-4 coordinate into rho's range and produce a
REM wrong-but-plausible rho, silently. Nothing in the archive records which space
REM it came from, so there is no code guard for this.
REM
REM tau_ceiling is gone from the search: it fed only the cubic barrier, which has
REM been use_cubic=false throughout, so it has been an inert dimension consuming a
REM quarter of the budget for the whole campaign.
REM
REM lambda-chatter 3.0 and the eps floors match runs_asmc_v2_chatter exactly, so
REM the objective is unchanged and the two runs are comparable.
REM
REM Usage:  run_stage_asmc_v2_demandk.bat <clean|noisy> [nseeds=5]
REM     run_stage_asmc_v2_demandk.bat clean
REM     run_stage_asmc_v2_demandk.bat noisy      (warm-starts from THIS run's clean
REM                                               stage -- same space, so legitimate)
REM
REM Run keep_awake.py alongside. One process per seed; keep the combined count
REM across concurrent waves at or under 10 on this machine (16 cores, ~1.8 GB per
REM process against 40 GB total).
REM =============================================================================
setlocal enabledelayedexpansion
set STAGE=%1
set NSEEDS=%2
if "%STAGE%"==""  set STAGE=clean
if "%NSEEDS%"=="" set NSEEDS=5

set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v2_demandk

REM NOTE: no REM lines inside the for-block below. cmd parses the whole
REM parenthesised block up front and REM inside it emits spurious
REM "'M' is not recognized" errors (harmless, but they look like a real failure).
REM
REM The noisy stage warm-starts from THIS run's OWN clean stage: same space, same
REM dimension-4 meaning, so the cross-space hazard described above does not apply.
for /L %%S in (1,1,%NSEEDS%) do (
    set WARM=
    if "%STAGE%"=="noisy" set WARM=--warm-from "%OUT%\seed%%S\asmc_v2_clean\best_config.json"
    start "asmc_v2_demandk_%STAGE%_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" ^
        --stage 1 --controller asmc --asmc-v2 --seed %%S ^
        --trajset-screen train12 --trajset-full train12 --noise %STAGE% ^
        --eps-floor-xy 0.02 --eps-floor-psi 0.08 ^
        --lam-psi-hi 60.0 ^
        --lambda-chatter 3.0 --p1-cap 250 --p2-cap 60 --out "%OUT%" !WARM! ^
        > "%OUT%_%STAGE%_seed%%S.log" 2>&1
)

echo Launched %NSEEDS% ASMC v2 demand-k %STAGE% seeds. Logs: %OUT%_%STAGE%_seed1.log .. _seed%NSEEDS%.log
endlocal
