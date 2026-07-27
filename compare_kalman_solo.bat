@echo off
REM Windows launcher: legacy-KF-only compare (cross-contamination test).
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" --project="%ROOT%" compare_estimators.jl --estimator-dirs runs_estimator/kalman --seeds 42,43 --out runs_estimator/compare_kalman_solo
pause
