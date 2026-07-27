@echo off
setlocal

:: Controller comparison on frozen ESKF: ASMC vs PID (no MPC), single seed.
:: Uses the same velocity trajectory set as experiment_noise_eval.jl
:: (default_trajs_3 excluding ellipse), in two variants:
::   with_coupled  = octagon, spin_creep, coupled_vomega, spiral_orbit
::   no_coupled    = octagon, spin_creep, spiral_orbit

cd /d "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set SCRIPT=compare_controllers_eskf.jl
set LOG=_tmp\compare_controllers_eskf_asmc_pid_run.log

if not exist _tmp mkdir _tmp

:: Wake-lock: Modern Standby kills long Julia runs.
start "keep_awake" /min cmd /c "python keep_awake.py"

echo Starting %SCRIPT% at %date% %time% > %LOG%
%JULIA% -t 1 --project=. --startup-file=no %SCRIPT% ^
  --controllers asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,pid:runs_controller_pid_5seed/pid_FINAL_seed2.json ^
  --run-dir trajectory_files_run_0p5_main ^
  --seeds 1 ^
  --out runs_controller_compare_eskf_asmc_pid >> %LOG% 2>&1

echo Finished at %date% %time% >> %LOG%

endlocal
