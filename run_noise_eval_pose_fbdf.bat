@echo off
REM FBDF ablation: pose-mode oracle noisy robustness (4 non-ellipse trajs).

set "ROOT=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights"
set "JULIA_EXE=C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe"

cd /d "%ROOT%"

"%JULIA_EXE%" -t 4 --project="%ROOT%" --startup-file=no hybrid_ctrl\controller_tuning\experiment_noise_eval_pose.jl ^
  --exclude-ellipse ^
  --out runs_controller\noise_eval_pose_10seed_fbdf
