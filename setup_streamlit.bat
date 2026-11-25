@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo CUSTOMER CLUSTERING - STREAMLIT SETUP
echo ================================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found.
    echo Install Python from https://www.python.org/downloads/
    echo During installation, enable the Python PATH option.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment with your default Python...
    py -m venv .venv
    if errorlevel 1 goto :error
)

call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Starting Streamlit dashboard...
python -m streamlit run streamlit_app.py
if errorlevel 1 goto :error

goto :end

:error
echo.
echo Setup or launch failed. Read the error above and check the deployment guide.
pause
exit /b 1

:end
endlocal
