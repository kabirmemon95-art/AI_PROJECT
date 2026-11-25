@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  CUSTOMER CLUSTERING - FIRST TIME SETUP AND RUN
echo ============================================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found.
    echo Install Python from https://www.python.org/downloads/
    echo During installation, enable the Python PATH option.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    py -m venv .venv
    if errorlevel 1 goto :error
)

echo.
echo Installing or updating required libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Starting the clustering project...
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto :error

echo.
echo The project finished. Check the results and plots folders.
pause
exit /b 0

:error
echo.
echo An error occurred. Read the message above, correct the issue, and run this file again.
pause
exit /b 1
