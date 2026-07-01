@echo off
REM =============================================================================
REM run_a1_cross_study.bat — Cross-subset evaluation for Mamba ForceRecon v1.
REM
REM Runs cross_eval.py for every finished A1 run under
REM Mecanum_PINN_Mamba_ForceRecon_v1\checkpoints_mamba_v1\, then runs
REM cross_report.py to aggregate the results.
REM
REM Usage:
REM   double-click, or from cmd:
REM     bat_files\run_a1_cross_study.bat
REM   force re-evaluation:
REM     bat_files\run_a1_cross_study.bat --force
REM =============================================================================

setlocal enabledelayedexpansion

set CODE_ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
set ENV_NAME=Vsk_venv
set PKG=Mecanum_PINN_Mamba_ForceRecon_v1
set CKPT_DIR=%PKG%\checkpoints_mamba_v1

cd /d "%CODE_ROOT%" || (
    echo ERROR: could not cd to %CODE_ROOT%
    exit /b 1
)

call conda activate %ENV_NAME% 2>nul || call activate %ENV_NAME% 2>nul
if errorlevel 1 (
    echo ERROR: could not activate conda environment %ENV_NAME%
    exit /b 1
)

python %PKG%\run_cross_study.py --ckpt-dir %CKPT_DIR% %*
if errorlevel 1 (
    echo ERROR: cross-study run failed
    exit /b 1
)

conda deactivate
pause
