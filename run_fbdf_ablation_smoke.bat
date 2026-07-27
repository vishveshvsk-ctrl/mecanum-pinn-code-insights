@echo off
REM Smoke test for FBDF solver ablation on controller/estimator evaluation.

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t 1 --project="%ROOT%" --startup-file=no hybrid_ctrl\estimator_tuning\compare_controllers_eskf_pose.jl ^
  --sensor-noise default ^
  --out hybrid_ctrl\estimator_tuning\reports\controller_eskf_pose_fbdf ^
  --smoke
