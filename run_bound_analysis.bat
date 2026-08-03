@echo off
REM =============================================================================
REM run_bound_analysis.bat -- end-to-end PCRLB-no-odometry ellipse bound study.
REM
REM Per instructions/pcrlb-no-odometry-ellipse-bound.md: screens the ellipse
REM combo grid, exports 10 closed-loop truth+achieved traces, runs the PCRLB
REM recursion over 4 measurement variants (B0-B3) x 2 sensor grades, and
REM produces the summary CSV + figures. Sequential stages -- each depends on
REM the previous one's output, so this does NOT parallelize across julia
REM processes the way hybrid_ctrl_v2's 5-seed launchers do.
REM
REM Usage: run_bound_analysis.bat
REM =============================================================================
setlocal enabledelayedexpansion
set ROOT=%~dp0
set PYTHON=C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe
set BA=%ROOT%bound_analysis

echo [1/4] Screening ellipse combos (select_ellipse_combos.jl)...
julia "%BA%\select_ellipse_combos.jl"
if errorlevel 1 (
    echo select_ellipse_combos.jl FAILED.
    exit /b 1
)

echo.
echo [2/4] Exporting truth + achieved traces (export_truth_traces.jl)...
julia "%BA%\export_truth_traces.jl"
if errorlevel 1 (
    echo export_truth_traces.jl FAILED.
    exit /b 1
)

echo.
echo [3/4] Running the PCRLB recursion (run_bound.py)...
"%PYTHON%" "%BA%\run_bound.py"
if errorlevel 1 (
    echo run_bound.py FAILED.
    exit /b 1
)

echo.
echo [4/4] Producing figures (plot_bound.py)...
"%PYTHON%" "%BA%\plot_bound.py"
if errorlevel 1 (
    echo plot_bound.py FAILED.
    exit /b 1
)

echo.
echo Done. See %BA%\reports\ for bound_summary.csv and figure*.png.
endlocal
