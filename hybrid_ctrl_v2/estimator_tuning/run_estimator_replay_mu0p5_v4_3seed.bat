@echo off
REM =============================================================================
REM run_estimator_replay_mu0p5_v4_3seed.bat — 3-seed sequential v4 replay tuning
REM =============================================================================
REM ESKFEstimatorV3 (13-dim yaw-accel state), 7-dim v4 space (q_alpha tunable).
REM Output goes to runs_estimator_v4_mu0p5_train12/ (v2/v3 runs untouched).
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set OUTDIR=hybrid_ctrl_v2\runs_estimator_v4_mu0p5_train12

echo Running seed 1...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v4.jl" --seed 1 --out %OUTDIR%

echo Running seed 2...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v4.jl" --seed 2 --out %OUTDIR%

echo Running seed 3...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v4.jl" --seed 3 --out %OUTDIR%

echo All seeds finished.
endlocal
