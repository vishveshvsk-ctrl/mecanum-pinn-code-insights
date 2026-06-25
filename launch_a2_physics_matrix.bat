@echo off
rem Launcher for the A2 physics-loss 2x2 matrix (residual + integrated, S1 + S2).
rem Run this from the Windows live tree root:
rem   C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\
rem
rem Assumes origin checkpoints already exist:
rem   observer_v1_py\runs\S1_train_w32_non_phys_max_norm_b1024\checkpoint.pt
rem   observer_v1_py\runs\S2_train_w32_non_phys_max_norm_b1024\checkpoint.pt
rem
rem Uses the decimated cache at C:\Users\vishv\mecanum_cache_decim (off OneDrive).
rem On the laptop run sequentially with --max-parallel 1; on the 24/128 box raise
rem --max-parallel until nvidia-smi GPU util saturates.

setlocal enabledelayedexpansion
set PYTHONUTF8=1
set PY=C:\Users\vishv\miniforge3\envs\myenv\python.exe
set CACHE=C:\Users\vishv\mecanum_cache_decim
set SCALER=..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov\variable_scaler_percentiles.csv
set COMMON=--windows 32 --per-run-batch 1024 --phase-epochs 60 --norm max --scaler-csv "%SCALER%" --cache-dir "%CACHE%" --dl-workers 2 --max-parallel 1 --heartbeat 120

echo === A2 physics-loss matrix ===
echo PYTHON=%PY%
echo CACHE=%CACHE%

rem ---- integrated variant, S1 ----
echo [1/4] integrated / S1
%PY% -u observer_v1_py\launch_parallel.py %COMMON% ^
  --regimes S1_train ^
  --physics-loss --physics-variant integrated ^
  --warm-from observer_v1_py\runs\S1_train_w32_non_phys_max_norm_b1024\checkpoint.pt ^
  --tag-suffix _phys_integrated_b1024 ^
  --log-dir observer_v1_py\runs\_parallel_logs_phys_integrated_S1 ^
  --csv observer_v1_py\runs\sweep_results_phys_integrated_S1.csv
if errorlevel 1 exit /b 1

rem ---- integrated variant, S2 ----
echo [2/4] integrated / S2
%PY% -u observer_v1_py\launch_parallel.py %COMMON% ^
  --regimes S2_train ^
  --physics-loss --physics-variant integrated ^
  --warm-from observer_v1_py\runs\S2_train_w32_non_phys_max_norm_b1024\checkpoint.pt ^
  --tag-suffix _phys_integrated_b1024 ^
  --log-dir observer_v1_py\runs\_parallel_logs_phys_integrated_S2 ^
  --csv observer_v1_py\runs\sweep_results_phys_integrated_S2.csv
if errorlevel 1 exit /b 1

rem ---- residual variant, S1 ----
echo [3/4] residual / S1
%PY% -u observer_v1_py\launch_parallel.py %COMMON% ^
  --regimes S1_train ^
  --physics-loss --physics-variant residual ^
  --warm-from observer_v1_py\runs\S1_train_w32_non_phys_max_norm_b1024\checkpoint.pt ^
  --tag-suffix _phys_residual_b1024 ^
  --log-dir observer_v1_py\runs\_parallel_logs_phys_residual_S1 ^
  --csv observer_v1_py\runs\sweep_results_phys_residual_S1.csv
if errorlevel 1 exit /b 1

rem ---- residual variant, S2 ----
echo [4/4] residual / S2
%PY% -u observer_v1_py\launch_parallel.py %COMMON% ^
  --regimes S2_train ^
  --physics-loss --physics-variant residual ^
  --warm-from observer_v1_py\runs\S2_train_w32_non_phys_max_norm_b1024\checkpoint.pt ^
  --tag-suffix _phys_residual_b1024 ^
  --log-dir observer_v1_py\runs\_parallel_logs_phys_residual_S2 ^
  --csv observer_v1_py\runs\sweep_results_phys_residual_S2.csv
if errorlevel 1 exit /b 1

echo === A2 physics-loss matrix complete ===
endlocal
