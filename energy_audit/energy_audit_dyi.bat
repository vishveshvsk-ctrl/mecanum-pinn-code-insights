@echo off
REM energy_audit_dyi.bat — Windows launcher for the DYi energy-leak audit.
REM Mirrors energy_audit_dyi.py against the Windows-synced project tree.
setlocal
set VENV=C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe
set SCRIPT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\_tmp\energy_audit_dyi.py
set DATA=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\data\Simulation_Data_MecanumSlipSpin_LugreAdamov
set CFG=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\trajectory_files_run_0p3_main

"%VENV%" "%SCRIPT%" --data-dir "%DATA%" --config-dir "%CFG%" %*
endlocal
