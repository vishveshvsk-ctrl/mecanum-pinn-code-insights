@echo off
REM eval_docking_oracle.bat — docking (step-hold) controller comparison, oracle feedback
cd /d C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
"C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe" --project=. -t 1 hybrid_ctrl_v2\eval_docking_oracle.jl %*
