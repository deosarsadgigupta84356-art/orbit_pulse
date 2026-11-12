@echo off
title ORBIT PULSE - Mission Control
cd /d "%~dp0"

echo.
echo ========================================
echo       ORBIT PULSE - MISSION CONTROL
echo ========================================
echo.
echo Starting dashboard...
echo Keep this window open while using ORBIT PULSE.
echo.

python -m streamlit run dashboard/app.py

echo.
echo ORBIT PULSE has stopped.
pause