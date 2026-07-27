@echo off
REM Windows launcher for tune_estimator.jl
REM Adjust JULIA_EXE if your juliaup path differs.

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" --project="%ROOT%" tune_estimator.jl --estimator smo --budget 30 --max-parallel 2 --seed 42 --out runs_estimator
