@echo off
REM Windows launcher: smoke for absolute-objective IMM tuning (11-state, budget 4).
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator kalman_imm --optimizer dxnes --budget 4 --max-parallel 4 --seed 42 --obj-seeds 42 --out runs_estimator_abs_smoke
pause
