@echo off
REM tierB_multisine.bat — open-loop odd-multisine BLA analysis (see .jl header)
cd /d C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights
"C:\Users\vishv\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe" --project=. -t 6 nonlinearity_analysis\tierB_multisine.jl %*
