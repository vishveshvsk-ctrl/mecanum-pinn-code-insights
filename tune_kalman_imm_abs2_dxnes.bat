@echo off
REM Windows launcher: absolute-objective IMM tuning on the WHITELIST-PINNED subset
REM (combo_idx pinned per diagnostics_combined.csv keep-list + envelope feasibility),
REM budget 400, multi-seed (42,43), -t4. Output: runs_estimator_abs2.
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" tune_estimator.jl --estimator kalman_imm --optimizer dxnes --budget 400 --max-parallel 4 --seed 42 --obj-seeds 42,43 --out runs_estimator_abs2
pause
