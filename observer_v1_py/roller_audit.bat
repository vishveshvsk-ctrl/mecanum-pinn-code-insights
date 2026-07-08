@echo off
REM roller_audit.bat — roller residual headroom diagnostic against a v1 checkpoint (Windows).
REM Run from the project root: C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\
set "PY=C:\Users\vishv\miniforge3\envs\myenv\python.exe"
set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

%PY% -u "%ROOT%\observer_v1_py\roller_audit.py" ^
  --run-dir "%ROOT%\observer_v1_py\runs\S1_train_w32_non_phys_max_norm" ^
  --cache-dir "C:/Users/vishv/mecanum_cache_decim" ^
  %*
