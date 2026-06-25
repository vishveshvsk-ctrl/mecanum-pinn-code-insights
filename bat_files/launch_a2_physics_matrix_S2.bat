@echo off
rem S2 fold of the A2 physics-loss matrix (integrated + residual, sequential).
rem Run this in one Windows terminal; run launch_a2_physics_matrix_S1.bat in
rem another terminal to execute S1 and S2 concurrently.
rem
rem Run from: C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\bat_files\
rem Assumes warm-start origin exists:
rem   ..\observer_v1_py\runs\S2_train_w32_non_phys_max_norm_b1024\checkpoint.pt

setlocal
set PYTHONUTF8=1
set PY=C:\Users\vishv\miniforge3\envs\myenv\python.exe
set CACHE=C:\Users\vishv\mecanum_cache_decim
set SCALER=..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov\variable_scaler_percentiles.csv
set WARM=..\observer_v1_py\runs\S2_train_w32_non_phys_max_norm_b1024\checkpoint.pt
set COMMON=--regimes S2_train --windows 32 --per-run-batch 1024 --phase-epochs 60 --norm max --scaler-csv "%SCALER%" --cache-dir "%CACHE%" --dl-workers 2 --max-parallel 1 --heartbeat 120

echo === A2 physics-loss matrix: S2 fold ===

echo [S2 1/2] integrated
%PY% -u ..\observer_v1_py\launch_parallel.py %COMMON% ^
  --physics-loss --physics-variant integrated ^
  --warm-from "%WARM%" ^
  --tag-suffix _phys_integrated_b1024 ^
  --log-dir ..\observer_v1_py\runs\_parallel_logs_phys_integrated_S2 ^
  --csv ..\observer_v1_py\runs\sweep_results_phys_integrated_S2.csv
if errorlevel 1 exit /b 1

echo [S2 2/2] residual
%PY% -u ..\observer_v1_py\launch_parallel.py %COMMON% ^
  --physics-loss --physics-variant residual ^
  --warm-from "%WARM%" ^
  --tag-suffix _phys_residual_b1024 ^
  --log-dir ..\observer_v1_py\runs\_parallel_logs_phys_residual_S2 ^
  --csv ..\observer_v1_py\runs\sweep_results_phys_residual_S2.csv
if errorlevel 1 exit /b 1

echo === S2 fold complete ===
endlocal
