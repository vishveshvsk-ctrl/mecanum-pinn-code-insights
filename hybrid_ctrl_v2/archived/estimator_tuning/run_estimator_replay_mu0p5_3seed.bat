@echo off
REM =============================================================================
REM run_estimator_replay_mu0p5_3seed.bat — 3-seed sequential ESKF v2 replay tuning
REM =============================================================================
REM Launches seeds 1, 2, and 3 sequentially for the mu=0.5 train12 realistic-
REM noise re-tune. Output goes to runs_estimator_v2_mu0p5_train12/.
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set OUTDIR=hybrid_ctrl_v2\runs_estimator_v2_mu0p5_train12

echo Running seed 1...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5.jl" --seed 1 --out %OUTDIR%

echo Running seed 2...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5.jl" --seed 2 --out %OUTDIR%

echo Running seed 3...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5.jl" --seed 3 --out %OUTDIR%

echo All seeds finished.
endlocal
