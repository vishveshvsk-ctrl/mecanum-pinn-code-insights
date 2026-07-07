@echo off
setlocal enabledelayedexpansion

set "LOGDIR=C:\Users\vishv\OneDrive\Desktop\Vishvesh_Data\VNIT\mecanum_pinn_head\code_insights\bat_files"
set "LOGFILE=%LOGDIR%\kimi_test.log"

echo === Kimi bat test started === > "%LOGFILE%"
echo Date: %date% Time: %time% >> "%LOGFILE%"
echo Current directory: %cd% >> "%LOGFILE%"
echo Windows username: %USERNAME% >> "%LOGFILE%"
echo Computer name: %COMPUTERNAME% >> "%LOGFILE%"
echo Arguments passed: %* >> "%LOGFILE%"
echo. >> "%LOGFILE%"
echo Hello from kimi_test.bat, launched from WSL. >> "%LOGFILE%"
echo This confirms bat scripts can be written, launched, and logged from Kimi. >> "%LOGFILE%"
echo. >> "%LOGFILE%"
echo === Kimi bat test completed === >> "%LOGFILE%"

echo Done. Log written to: %LOGFILE%
exit /b 0
