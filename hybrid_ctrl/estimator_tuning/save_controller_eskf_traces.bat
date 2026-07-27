@echo off
setlocal

cd /D "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

set JULIA_NUM_THREADS=1
set "JULIA=C:\Users\vishv\AppData\Local\Microsoft\WindowsApps\julia.exe"

%JULIA% hybrid_ctrl\estimator_tuning\save_controller_eskf_traces.jl %*

endlocal
