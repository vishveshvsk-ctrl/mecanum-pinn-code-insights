@echo off
REM =====================================================================
REM  launch_a1_24gb.bat  --  A1 (Mamba ForceRecon PINN) sweep on the 24 GB box
REM
cd "G:/Vishvesh Koranne PINN Robotics"

"MecanumPINN_venv/\\Scripts\activate.bat"

REM  >>> ADD YOUR VENV ACTIVATION ABOVE THIS LINE so that `python` = the torch
REM      venv, e.g.:   call C:\path\to\venv\Scripts\activate.bat
REM                    (or)  call conda activate <env>
REM
REM  Put this .bat in the folder that holds BOTH the venv and mecanum_pinn_head
REM  (your venv working dir). %~dp0 anchors paths to THIS file's location, so it
REM  works no matter where you launch it from.
REM =====================================================================

REM --- cd into code_insights (the scripts REQUIRE this as the working dir) ---
cd /d "%~dp0mecanum_pinn_head\code_insights"
if errorlevel 1 ( echo [error] could not cd into code_insights -- fix the path above & pause & exit /b 1 )

REM ------------------------- sweep knobs (EDIT) ------------------------
REM  MAXPAR: concurrent runs (MAXPAR x DLWORKERS = total loader procs; keep <= cores)
set MAXPAR=8
REM  WINDOWS: seq_len ablation (stride = 0.5 x window)
set WINDOWS=8,16,32
REM  REGIMES: S1/S2 only -- the chi (S3) axis is separate
set REGIMES=S1_train,S2_train
REM  BATCH: per-run batch (model is tiny; a throughput knob)
set BATCH=1024
REM  DLWORKERS: dataloader workers per run
set DLWORKERS=2
REM  EPOCHSCALE: 0.5 ~= 220 epochs total (fwd+inv); 1.0 = full 440
set EPOCHSCALE=0.5
REM ---------------------------------------------------------------------

REM  (add --dry-run to preview the job list without running)
python Mecanum_PINN_Mamba_ForceRecon_v1\launch_parallel.py ^
  --regimes %REGIMES% --windows %WINDOWS% --stride-frac 0.5 ^
  --max-parallel %MAXPAR% --per-run-batch %BATCH% --dl-workers %DLWORKERS% ^
  --epoch-scale %EPOCHSCALE% --no-lbfgs ^
  --warm-cache --heartbeat 300

echo.
echo [done] launcher exited. ranking CSV: Mecanum_PINN_Mamba_ForceRecon_v1\runs\sweep_results.csv
pause
