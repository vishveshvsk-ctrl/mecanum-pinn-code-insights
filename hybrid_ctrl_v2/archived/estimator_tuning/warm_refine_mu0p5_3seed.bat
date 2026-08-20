@echo off
REM =============================================================================
REM warm_refine_mu0p5_3seed.bat — 3-seed sequential warm-refine continuation
REM =============================================================================
REM Continues the mu=0.5 train12 realistic-noise tuning from the output of
REM run_estimator_replay_mu0p5_3seed.bat.
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set OUTDIR=hybrid_ctrl_v2\runs_estimator_v2_mu0p5_train12

echo Warm-refining seed 1...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\warm_refine_mu0p5.jl" --seed 1 --in %OUTDIR% --out %OUTDIR%

echo Warm-refining seed 2...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\warm_refine_mu0p5.jl" --seed 2 --in %OUTDIR% --out %OUTDIR%

echo Warm-refining seed 3...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\warm_refine_mu0p5.jl" --seed 3 --in %OUTDIR% --out %OUTDIR%

echo All warm-refine seeds finished.
endlocal
