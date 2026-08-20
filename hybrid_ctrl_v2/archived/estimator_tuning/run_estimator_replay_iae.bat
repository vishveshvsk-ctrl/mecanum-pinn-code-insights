@echo off
REM =============================================================================
REM run_estimator_replay_iae.bat — IAE adaptive-Q replay-tuning launcher
REM (instructions/estimator-v2-iae-adaptive.md §7)
REM =============================================================================
REM Tunes ESKFIAEEstimatorV2's 12-dim param space by replaying over the frozen
REM 11-trajectory manifest. Runs from the Windows live-data tree per AGENTS.md §6.
REM
cd /d "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
julia --project="." hybrid_ctrl_v2\estimator_tuning\run_estimator_replay_iae.jl %*
