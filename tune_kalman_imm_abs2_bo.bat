@echo off
REM Windows launcher: Bayesian (Kriging/DYCORS) refinement warm-started from the
REM dxNES plateau incumbent (294.46), narrowed box ±25%, budget 150, -t4.
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator kalman_imm --optimizer bo --budget 150 --max-parallel 4 --seed 42 --obj-seeds 42,43 --warm-start runs_estimator_abs2\kalman_imm_dxnes\checkpoint_best_dxnes270.json --out runs_estimator_abs2_bo
pause
