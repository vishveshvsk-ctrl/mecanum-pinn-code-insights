@echo off
REM Windows launcher: ESKF tuning, dxNES budget 300, multi-seed (42,43), -t4.
REM DO NOT RUN before user confirmation (theory sign-off).
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator eskf --optimizer dxnes --budget 300 --max-parallel 4 --seed 42 --obj-seeds 42,43 --out runs_eskf
pause
