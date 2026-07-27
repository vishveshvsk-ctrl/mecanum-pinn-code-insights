@echo off
REM Run B v2 (WITHOUT ellipse, WITHOUT docking): ESKF dxNES budget 300, seeds 42,43, -t4 -> runs_eskf_noellipse_v2
REM docking excluded: profile IC bug (ref start != plant start) poisoned the objective in v1 runs.
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator eskf --optimizer dxnes --budget 300 --max-parallel 4 --seed 42 --obj-seeds 42,43 --exclude-entries ellipse,docking --out runs_eskf_noellipse_v2
pause
