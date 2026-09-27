@echo off
cd /d "%~dp0"
python --version
if errorlevel 1 goto :fail
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto :fail
if exist "requirements-lock.txt" ".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m saathi download-models
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m saathi doctor
if errorlevel 1 goto :fail
echo Ready. Double-click start.bat.
pause
exit /b 0
:fail
echo Setup did not finish. Read the error above and README.md.
pause
exit /b 1
