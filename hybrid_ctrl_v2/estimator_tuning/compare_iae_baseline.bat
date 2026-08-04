@echo off
REM =============================================================================
REM compare_iae_baseline.bat — Fixed-default baseline vs IAE adaptive-Q comparison
REM (instructions/estimator-v2-iae-adaptive.md §6)
REM =============================================================================
REM Runs compare_iae_baseline.jl from the Windows live-data tree. Per AGENTS.md
REM §6, the authoritative runtime tree is the Windows-synced folder; this batch
REM cd's there and invokes Julia with the project in that tree.
REM
REM Usage: compare_iae_baseline.bat [--seed S] [--out DIR]
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
set SCRIPT=hybrid_ctrl_v2\estimator_tuning\compare_iae_baseline.jl
set EXTRA=%*

cd /d "%ROOT%"
julia --project="." "%SCRIPT%" %EXTRA%
endlocal
