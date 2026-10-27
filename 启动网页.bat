@echo off
chcp 65001 >nul
setlocal
title Leaf Disease Recognition
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "TEMP=%~dp0.venv\smoke_tmp"
set "TMP=%TEMP%"
set "TMPDIR=%TEMP%"
set "XDG_CACHE_HOME=%TEMP%"
set "PYTHONUTF8=1"
set "PYTHONPYCACHEPREFIX=%TEMP%\web_pycache"

if not exist "%PYTHON%" goto missing_python
if not exist "%~dp0app.py" goto missing_app
if not exist "%TEMP%\" mkdir "%TEMP%"
if not exist "%TEMP%\" goto missing_temp

"%PYTHON%" -c "import socket,sys; s=socket.socket(); s.settimeout(1); result=s.connect_ex(('127.0.0.1',8601)); s.close(); sys.exit(0 if result == 0 else 1)"
if not errorlevel 1 goto port_in_use

echo Starting the leaf disease recognition app...
echo Open in your browser: http://127.0.0.1:8601
echo Keep this window open. Press Ctrl+C to stop.
echo.
"%PYTHON%" -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8601 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 goto startup_failed
exit /b 0

:port_in_use
echo Port 8601 is already in use. No second server was started.
echo Please check: http://127.0.0.1:8601
echo If this is another app, check its process before starting this project.
pause
exit /b 0

:missing_python
echo Project Python was not found: %PYTHON%
echo Keep this launcher in the project folder with its .venv environment.
pause
exit /b 1

:missing_app
echo app.py was not found. Keep this launcher in the project root.
pause
exit /b 1

:missing_temp
echo Could not create the temporary folder: %TEMP%
echo Check that the project folder is writable.
pause
exit /b 1

:startup_failed
echo.
echo Startup failed. Keep this window open and check the error above.
pause
exit /b 1