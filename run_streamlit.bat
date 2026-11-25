@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No project virtual environment found. Run setup_streamlit.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run streamlit_app.py
pause
endlocal
