@echo off
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM Generate Arrow data for the new lemniscate_flat profile and the updated
REM spiral_orbit profile across the three mu config directories.
REM Runs each config directory sequentially with 8 Julia threads; resume-aware.
REM ---------------------------------------------------------------------------
set CODE_DIR=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe

cd /d "%CODE_DIR%"
if errorlevel 1 exit /b 1

echo [%date% %time%] Starting lemniscate_flat + spiral_orbit sweep

set MU_DIRS=trajectory_files_run_0p3_main trajectory_files_run_0p5_main trajectory_files_run_0p8_main
for %%D in (%MU_DIRS%) do (
    set CFG=%%D
    for /f "tokens=4 delims=_" %%M in ("%%D") do set MU=%%M
    echo.
    echo [%date% %time%] Running config dir: !CFG!  [mu=!MU!]
    "%JULIA%" --project=. -t 8 Data_Generation_Julia.jl --config-dir !CFG! --profiles lemniscate_flat_mu_!MU!.toml,spiral_orbit_mu_!MU!.toml --timeout 300 --progress-interval 60
    if errorlevel 1 (
        echo [%date% %time%] FAILED in !CFG!
        exit /b 1
    )
)

echo [%date% %time%] Sweep complete
endlocal
