@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The project environment has not been created yet.
    echo Double-click setup_and_run.bat first.
    pause
    exit /b 1
)

echo Running Customer Clustering Project...
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo The project stopped because of an error. Read the message above.
)
pause
