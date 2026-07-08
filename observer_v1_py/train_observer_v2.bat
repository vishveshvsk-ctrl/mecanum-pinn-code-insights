@echo off
REM train_observer_v2.bat — single γ-only observer v2 training run (Windows).
REM Run from the project root: C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\
set "PY=C:\Users\vishv\miniforge3\envs\myenv\python.exe"
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

%PY% -u "%ROOT%\observer_v1_py\train_observer_v2.py" ^
  --regime "%ROOT%\observer_v1_py\regimes\S1_train.toml" ^
  --window 32 ^
  --stride-frac 0.5 ^
  --cache-dir "C:/Users/vishv/mecanum_cache_decim" ^
  --scaler-csv "%ROOT%\..\data\Simulation_Data_MecanumSlipSpin_LugreAdamov\variable_scaler_percentiles.csv" ^
  --physics-variant residual ^
  --w-roller 1.0 ^
  --batch-size 2048 ^
  --jobs 4 ^
  %*
