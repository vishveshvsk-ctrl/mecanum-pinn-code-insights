@echo off
REM Windows launcher: top-10 re-rank of the BO refinement run on HELD-OUT seeds
REM 44,45 (search used 42,43), then freeze the robust winner.
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" rerank_topk.jl --trials runs_estimator_abs2_bo\kalman_imm_bo\trials.arrow --manifest runs_estimator_abs2_bo\subset_manifest.json --seeds 44,45 --topk 10 --max-parallel 4 --freeze-dir runs_estimator\frozen
pause
