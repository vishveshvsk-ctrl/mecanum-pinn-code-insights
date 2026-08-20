@echo off
REM =============================================================================
REM run_estimator_replay_3seed.bat — 3-seed parallel ESKF v2 replay tuning launcher
REM =============================================================================
REM Launches seeds 1, 2, and 3 in parallel for the realistic-noise, segmented-
REM replay re-tune. Output goes to runs_estimator_v2_replay_realistic_seg/.
REM Each seed uses a different deterministic random segment draw (seeded by its
REM optimization seed), so the three runs generalize across different windows.
REM =============================================================================
setlocal
set ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
cd /d "%ROOT%"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set OUTDIR=hybrid_ctrl_v2\runs_estimator_v2_replay_realistic_seg

echo Running seed 1...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay.jl" --seed 1 --out %OUTDIR%

echo Running seed 2...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay.jl" --seed 2 --out %OUTDIR%

echo Running seed 3...
"%JULIA%" --project="." "hybrid_ctrl_v2\estimator_tuning\run_estimator_replay.jl" --seed 3 --out %OUTDIR%

echo All seeds finished.
endlocal
