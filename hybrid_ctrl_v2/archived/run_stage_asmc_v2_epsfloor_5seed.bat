@echo off
REM =============================================================================
REM run_stage_asmc_v2_epsfloor_5seed.bat — ASMC v2 clean retune, WIDE eps floor
REM =============================================================================
REM eps_floor_xy=0.02 / eps_floor_psi=0.08 (per-user direction, succeeding the
REM 0.005/0.02 deferred config of the launch brief). Rationale (see
REM chat-handoff/asmc_v2_relaunch_pid_revalidation_handoff.md): the widened
REM layer is reachable at operating K (per-tick stride K*dt/m_eff < 2*eps), so
REM the sigma-leak engages and K relaxes as error drops; chatter -82% at
REM mid-box on the screen pair (_tmp/asmc_epsfloor_freeze_check.jl).
REM Same protocol as runs_asmc_v2_relaunch: 4-D space, train12, clean, 5 seeds.
REM =============================================================================
setlocal
set SCRIPT=%~dp0controller_tuning\run_stage.jl
set OUT=%~dp0runs_asmc_v2_epsfloor

for /L %%S in (1,1,5) do (
    start "asmc_v2_eps_seed%%S" /B julia --project="%~dp0.." "%SCRIPT%" --stage 1 --controller asmc --asmc-v2 --seed %%S --trajset-screen train12 --trajset-full train12 --noise clean --p1-cap 250 --p2-cap 60 --eps-floor-xy 0.02 --eps-floor-psi 0.08 --out "%OUT%" %EXTRA% > "%OUT%_seed%%S.log" 2>&1
)

echo Launched 5 ASMC v2 eps-floor seeds. Logs: %OUT%_seed1.log .. _seed5.log
endlocal
