@echo off
REM tierAC_static_jacobian.bat — Tier A+C near-linearity analysis (see .jl header)
cd /d C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
"C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe" --project=. -t 1 nonlinearity_analysis\tierAC_static_jacobian.jl %*
