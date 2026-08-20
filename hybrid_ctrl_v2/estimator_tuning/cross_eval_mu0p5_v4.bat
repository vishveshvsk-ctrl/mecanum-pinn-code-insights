@echo off
REM =============================================================================
REM cross_eval_mu0p5_v4.bat — 5-config ESKFEstimatorV3 cross-evaluation
REM =============================================================================
REM Evaluates the five tuned v4 configs under a common noise-seed set on
REM :train12 and :test replay tiers. 8 Julia threads (project parallelism cap).
REM Output: runs_estimator_v4_mu0p5_train12\cross_eval\
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe

"%JULIA%" --project="." -t 8 "hybrid_ctrl_v2\estimator_tuning\cross_eval_mu0p5_v4.jl"

endlocal
