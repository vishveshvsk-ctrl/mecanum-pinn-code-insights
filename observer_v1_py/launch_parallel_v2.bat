@echo off
REM launch_parallel_v2.bat — parallel S1+S2 γ-only observer v2 campaign (Windows).
REM Run from the project root: C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\
set "PY=C:\Users\vishv\miniforge3\envs\myenv\python.exe"
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

%PY% -u "%ROOT%\observer_v1_py\launch_parallel_v2.py" ^
  --warm-cache ^
  --max-parallel 2 ^
  --regimes S1_train,S2_train ^
  --windows 32 ^
  --per-run-batch 2048 ^
  --dl-workers 4 ^
  --cache-dir "C:/Users/vishv/mecanum_cache_decim" ^
  --scaler-csv "%ROOT%\..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov\variable_scaler_percentiles.csv" ^
  --physics-variant residual ^
  --w-roller 1.0 ^
  %*
