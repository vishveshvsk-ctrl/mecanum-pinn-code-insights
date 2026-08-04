@echo off
REM =============================================================================
REM run_estimator_replay_slipobs.bat — Slip-observer replay tuning launcher
REM =============================================================================
REM Runs hybrid_ctrl_v2/estimator_tuning/run_estimator_replay_slipobs.jl from the
REM Windows live-data tree root.  Passes through any extra args (--observer,
REM --init-from, --seed, etc.).
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

julia --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_slipobs.jl" %*
endlocal
