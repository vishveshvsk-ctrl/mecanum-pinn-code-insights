@echo off
setlocal

:: Controller comparison on frozen ESKF, whitelist-sampled trajectories.
:: Runs ASMC + PID (no MPC) on 10 trajectories per velocity profile from
:: diagnostics_combined.csv (combined_reco == keep, mu=0.5, ellipse excluded).
:: Single sensor-noise seed: the frozen ESKF already provides estimator noise.

cd /d "C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"

set JULIA=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe
set SCRIPT=compare_controllers_eskf_whitelist.jl
set LOG=_tmp\compare_controllers_eskf_whitelist_run.log

if not exist _tmp mkdir _tmp

:: Wake-lock: Modern Standby kills long Julia runs. Start before the job.
start "keep_awake" /min cmd /c "python keep_awake.py"

echo Starting %SCRIPT% at %date% %time% > %LOG%
%JULIA% -t 1 --project=. --startup-file=no %SCRIPT% ^
  --controllers asmc:runs_controller_asmc_pin/asmc_FINAL_seed3.json,pid:runs_controller_pid_5seed/pid_FINAL_seed2.json ^
  --run-dir trajectory_files_run_0p5_main ^
  --whitelist diagnostics_combined.csv ^
  --mu 0.5 ^
  --n-per-profile 10 ^
  --whitelist-col combined_reco ^
  --whitelist-vals keep ^
  --exclude-profiles ellipse ^
  --sample-seed 1234 ^
  --seeds 1 ^
  --out runs_controller_compare_eskf_whitelist >> %LOG% 2>&1

echo Finished at %date% %time% >> %LOG%

:: Optional: terminate the wake-lock helper after the run.
:: taskkill /FI "WINDOWTITLE eq keep_awake" /F >nul 2>&1

endlocal
