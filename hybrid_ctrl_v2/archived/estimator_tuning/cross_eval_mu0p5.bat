@echo off
REM =============================================================================
REM cross_eval_mu0p5.bat — decisive identifiability cross-evaluation
REM =============================================================================
REM Evaluates the three tuned ESKF v2 configs (orig + p0default variants) under
REM a common noise-seed set on :train12 and :test replay tiers. 8 Julia threads
REM (project parallelism cap). Output: runs_estimator_v2_mu0p5_train12\cross_eval\
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe

"%JULIA%" --project="." -t 8 "hybrid_ctrl_v2\estimator_tuning\cross_eval_mu0p5.jl"

endlocal
