@echo off
REM Windows launcher for compare_controllers_eskf.jl
REM {ASMC,PID,MPC} comparison on a frozen ESKF, two ellipse-excluded subset
REM variants (with/without coupled_vomega). See
REM instructions\controller-comparison-frozen-eskf.md

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no compare_controllers_eskf.jl ^
  --estimator-dir runs_eskf_noellipse_v2/eskf_dxnes ^
  --controllers asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,pid:runs_controller_pid_5seed/pid_FINAL_seed2.json,mpc:runs_controller/mpc_clean/best_config.json ^
  --run-dir trajectory_files_run_0p5_main ^
  --seeds 1,2,3,4,5,6,7,8,9,10 ^
  --out runs_controller_compare_eskf
