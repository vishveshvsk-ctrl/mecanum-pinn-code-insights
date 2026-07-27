@echo off
REM Run A (WITH ellipse): ESKF dxNES budget 300, seeds 42,43, -t4 → runs_eskf_full
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator eskf --optimizer dxnes --budget 300 --max-parallel 4 --seed 42 --obj-seeds 42,43 --warm-start runs_eskf\eskf_dxnes\checkpoint_best_final.json --out runs_eskf_full
pause
