@echo off
REM Windows launcher: top-10 multi-seed re-rank of the dxnes IMM tuning run,
REM then freeze the robust winner into runs_estimator/frozen.
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t4 --project="%ROOT%" rerank_topk.jl --max-parallel 4 --freeze-dir runs_estimator/frozen
pause
