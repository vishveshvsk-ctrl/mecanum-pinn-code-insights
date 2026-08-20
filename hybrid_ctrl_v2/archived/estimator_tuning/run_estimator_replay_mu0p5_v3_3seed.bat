@echo off
REM =============================================================================
REM run_estimator_replay_mu0p5_v3_3seed.bat — 3-seed sequential v3 replay tuning
REM =============================================================================
REM Launches seeds 1, 2, and 3 sequentially for the pruned 6-dim update-rate
REM space (param_space_v3.jl) on the mu=0.5 train12 realistic-noise replay set.
REM Output goes to runs_estimator_v3_mu0p5_train12/ (v2 runs untouched).
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set OUTDIR=hybrid_ctrl_v2\runs_estimator_v3_mu0p5_train12

echo Running seed 1...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v3.jl" --seed 1 --out %OUTDIR%

echo Running seed 2...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v3.jl" --seed 2 --out %OUTDIR%

echo Running seed 3...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_mu0p5_v3.jl" --seed 3 --out %OUTDIR%

echo All seeds finished.
endlocal
