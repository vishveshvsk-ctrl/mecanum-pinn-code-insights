@echo off
REM tierF_gamma_baseline.bat — gamma baseline storing pass (see .jl header)
cd /d C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
"C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe" --project=. -t 2 nonlinearity_analysis\tierF_gamma_baseline.jl %*
