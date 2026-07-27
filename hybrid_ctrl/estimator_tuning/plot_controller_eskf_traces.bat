@echo off
setlocal

cd /D "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

set "PYTHON=C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe"

%PYTHON% hybrid_ctrl\estimator_tuning\plot_controller_eskf_traces.py %*

endlocal
