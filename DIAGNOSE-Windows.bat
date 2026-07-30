@echo off
REM =====================================================================
REM  Collects everything needed to diagnose a failed start.
REM  Run this and send the resulting diagnose.txt (or a screenshot).
REM  Pure ASCII on purpose - see the note in START-Windows.bat.
REM =====================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "OUT=diagnose.txt"

echo ============================================================ > "%OUT%"
echo   LCM90 Visual Tracker - diagnostics                       >> "%OUT%"
echo ============================================================ >> "%OUT%"
echo. >> "%OUT%"

echo [date] >> "%OUT%"
date /t >> "%OUT%" 2>&1
time /t >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [windows version] >> "%OUT%"
ver >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [current folder] >> "%OUT%"
echo %CD% >> "%OUT%"
echo %CD%| findstr /R /C:"[^ -~]" >nul
if not errorlevel 1 (
    echo NOTE: path contains non-ASCII characters - this can break pip/venv >> "%OUT%"
)
echo. >> "%OUT%"

echo [codepage] >> "%OUT%"
chcp >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [py launcher] >> "%OUT%"
py -3 --version >> "%OUT%" 2>&1
echo exit code: %ERRORLEVEL% >> "%OUT%"
echo. >> "%OUT%"

echo [python on PATH] >> "%OUT%"
python --version >> "%OUT%" 2>&1
echo exit code: %ERRORLEVEL% >> "%OUT%"
where python >> "%OUT%" 2>&1
echo. >> "%OUT%"

echo [files present] >> "%OUT%"
if exist "serve.py"            (echo serve.py           OK >> "%OUT%") else (echo serve.py           MISSING >> "%OUT%")
if exist "config.yaml"         (echo config.yaml        OK >> "%OUT%") else (echo config.yaml        MISSING >> "%OUT%")
if exist "requirements.txt"    (echo requirements.txt   OK >> "%OUT%") else (echo requirements.txt   MISSING >> "%OUT%")
if exist "core\__init__.py" (echo core\              OK >> "%OUT%") else (echo core\              MISSING >> "%OUT%")
if exist ".venv\Scripts\python.exe" (echo .venv\             OK >> "%OUT%") else (echo .venv\             not created yet >> "%OUT%")
echo. >> "%OUT%"

if exist ".venv\Scripts\python.exe" (
    echo [venv python version] >> "%OUT%"
    ".venv\Scripts\python.exe" --version >> "%OUT%" 2>&1
    echo. >> "%OUT%"

    echo [installed packages] >> "%OUT%"
    ".venv\Scripts\python.exe" -m pip list >> "%OUT%" 2>&1
    echo. >> "%OUT%"

    echo [import test] >> "%OUT%"
    ".venv\Scripts\python.exe" -c "import sys;print('python',sys.version)" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import numpy;print('numpy',numpy.__version__)" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import cv2;print('opencv',cv2.__version__)" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import yaml;print('pyyaml ok')" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import fastapi,uvicorn;print('fastapi/uvicorn ok')" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import PIL;print('pillow ok')" >> "%OUT%" 2>&1
    ".venv\Scripts\python.exe" -c "import serial;print('pyserial ok')" >> "%OUT%" 2>&1
    echo. >> "%OUT%"

    echo [config load test] >> "%OUT%"
    ".venv\Scripts\python.exe" -c "import sys;sys.path.insert(0,'.');import yaml;cfg=yaml.safe_load(open('config.yaml',encoding='utf-8'));print('config ok, port',cfg.get('server',{}).get('port'))" >> "%OUT%" 2>&1
    echo. >> "%OUT%"

    echo [app import test] >> "%OUT%"
    ".venv\Scripts\python.exe" -c "import sys;sys.path.insert(0,'.');import app.main;print('app import ok')" >> "%OUT%" 2>&1
    echo. >> "%OUT%"
)

echo [serial ports] >> "%OUT%"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import serial.tools.list_ports as p;[print(x.device,x.description) for x in p.comports()] or print('(no COM ports found)')" >> "%OUT%" 2>&1
) else (
    echo venv not ready, skipped >> "%OUT%"
)
echo. >> "%OUT%"

echo ============================================================ >> "%OUT%"
echo   end of diagnostics >> "%OUT%"
echo ============================================================ >> "%OUT%"

type "%OUT%"
echo.
echo ============================================================
echo   Saved to: %CD%\diagnose.txt
echo   Please send that file (or a screenshot of this window).
echo ============================================================
pause
