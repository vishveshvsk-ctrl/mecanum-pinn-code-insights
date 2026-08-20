@echo off
REM =============================================================================
REM run_stage_asmc_v2_epsfloor_wide2_4seed.bat -- ASMC v2 clean retune, WIDE2:
REM box widened another 1.5x on top of runs_asmc_v2_epsfloor_wide (tau_ceiling
REM hi 200->300, lam_psi hi 40->60; lo bounds unchanged). Boxes: tau_ceiling
REM [0.1,300], lam_psi [1,60]; eps_floor_xy=0.02 / eps_floor_psi=0.08 unchanged.
REM
REM WHY: in runs_asmc_v2_epsfloor_wide, seeds 1/2/3/4 ALL converged with
REM tau_ceiling ~=197-199 and lam_psi_max ~=39.2-40.0 -- still pinned at (or
REM within ~2-3 units of) the old 200/40 ceilings even after that box was
REM already doubled once. This run checks whether the optimum is interior
REM beyond THIS rail too.
REM
REM WARM-START (per-user direction): each seed launches with --warm-from
REM pointing at its OWN runs_asmc_v2_epsfloor_wide/seedN best_config.json.
REM Per optimizer_stage.jl's warm_start path, this SKIPS phase 1's global
REM dxNES search entirely (phase1_evals=0 no matter what --p1-cap is set to --
REM the cap below is a no-op, kept only for consistency/logging) and goes
REM straight to phase 2's local BOBYQA refine in a box centred on the prior
REM converged point, clipped to the new [lo,hi]. This is a local re-check of
REM the rail off an already-converged point, not a fresh cold-start search.
REM
REM CAPS HALVED (per-user direction): p1-cap 250->125 (unused, see above),
REM p2-cap 60->30.
REM
REM SEED 5 EXCLUDED (per-user direction): its runs_asmc_v2_epsfloor_wide
REM result converged to a distinct, ~4x-worse basin (best_score ~3.55 vs
REM ~0.87 for seeds 1-4; lam_psi_max pinned at the LOWER bound ~1.0, not the
REM upper rail) -- not a rail-truncation case, so warm-starting from it would
REM just re-seed the bad basin. Only 4 seeds launched below.
REM
REM GUARD (unchanged from wide run): before accepting any result past the old
REM rails, verify max(K)/K_max_base on the winning config (log_K diagnostics).
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set IN=%~dp0runs_asmc_v2_epsfloor_wide
set OUT=%~dp0runs_asmc_v2_epsfloor_wide2

for /L %%S in (1,1,4) do (
    start "asmc_v2_wide2_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 1 --controller asmc --asmc-v2 --seed %%S --trajset-screen train12 --trajset-full train12 --noise clean --p1-cap 125 --p2-cap 30 --eps-floor-xy 0.02 --eps-floor-psi 0.08 --tau-ceiling-hi 300.0 --lam-psi-hi 60.0 --warm-from "%IN%\seed%%S\asmc_v2_clean\best_config.json" --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 4 ASMC v2 eps-floor WIDE2 warm-started seeds (1-4). Logs: %OUT%_seed1.log .. _seed4.log
endlocal
