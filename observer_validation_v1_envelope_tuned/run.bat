@echo off
REM Run SMO validation from any directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
"%USERPROFILE%\.julia\juliaup\julia-1.12.5+0.x64.w64.mingw32\bin\julia.exe" --project="%PROJECT_DIR%" "%SCRIPT_DIR%validate_smo_v1.jl"
pause
