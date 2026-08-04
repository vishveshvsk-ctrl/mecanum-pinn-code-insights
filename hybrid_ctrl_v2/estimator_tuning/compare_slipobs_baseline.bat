@echo off
REM =============================================================================
REM compare_slipobs_baseline.bat — fixed-gain slip-observer comparison
REM =============================================================================
REM Runs hybrid_ctrl_v2/estimator_tuning/compare_slipobs_baseline.jl from the
REM Windows live-data tree root.  Passes through any extra args (--seed S etc.).
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

julia --project="." "hybrid_ctrl_v2\estimator_tuning\compare_slipobs_baseline.jl" %*
endlocal
