@echo off
REM =============================================================================
REM run_stage_asmc_v2_epsfloor_wide_5seed.bat — ASMC v2 clean retune, wide layer
REM + DOUBLED tau_ceiling / lam_psi_max upper bounds (per-user direction, after
REM tau_ceiling railed at ~98-100 on 5/5 seeds and lam_psi at 20 on 2/5 in
REM runs_asmc_v2_epsfloor). Boxes: tau_ceiling [0.1,200], lam_psi [1,40];
REM eps_floor_xy=0.02 / eps_floor_psi=0.08 as in the epsfloor run.
REM GUARD (launch brief section 3/6): before accepting any result past the old
REM rails, verify max(K)/K_max_base on the winning config (log_K diagnostics) --
REM a tau_ceiling rail is also a maximum-authority / maximum-chatter result.
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v2_epsfloor_wide

for /L %%S in (1,1,5) do (
    start "asmc_v2_wide_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 1 --controller asmc --asmc-v2 --seed %%S --trajset-screen train12 --trajset-full train12 --noise clean --p1-cap 250 --p2-cap 60 --eps-floor-xy 0.02 --eps-floor-psi 0.08 --tau-ceiling-hi 200.0 --lam-psi-hi 40.0 --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ASMC v2 eps-floor WIDE-box seeds. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
