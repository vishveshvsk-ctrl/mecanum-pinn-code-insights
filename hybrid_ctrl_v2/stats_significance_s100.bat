@echo off
REM stats_significance_s100.bat — run the §9 significance tests from Windows.
REM Uses the no-torch venv (pandas+scipy only). CSV paths are relative to this dir.
cd /d %~dp0
C:\Users\vishv\claude-venv\mecanum\Scripts\python.exe stats_significance_s100.py
