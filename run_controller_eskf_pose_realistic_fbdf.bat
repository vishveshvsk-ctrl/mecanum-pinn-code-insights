@echo off
REM FBDF ablation: ASMC vs PID through frozen ESKF, realistic sensor/pose-fix noise.

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\compare_controllers_eskf_pose.jl ^
  --sensor-noise realistic ^
  --seeds 42,43,44,45,46 ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_realistic_fbdf
