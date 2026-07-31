@echo off
REM Diagnose why the QHY camera is not detected. Double-click to run.
REM ASCII-only on purpose - see START-Windows.bat.
setlocal
cd /d "%~dp0"

echo ============================================================
echo   QHY camera diagnostics
echo ============================================================
echo.

echo [1] Updating code...
git pull
echo.

echo [2] Looking for the QHY SDK dll...
if exist "C:\Windows\System32\qhyccd.dll" (echo   System32\qhyccd.dll   FOUND) else (echo   System32\qhyccd.dll   MISSING)
if exist "C:\Windows\SysWOW64\qhyccd.dll" (echo   SysWOW64\qhyccd.dll   FOUND) else (echo   SysWOW64\qhyccd.dll   MISSING)
echo.

echo [3] Loading SDK and scanning for cameras...
if not exist ".venv\Scripts\python.exe" (
    echo   [X] Environment not ready - run START-Windows.bat once first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe tools\qhy_probe.py
echo.

echo ============================================================
echo   Send a screenshot of this window.
echo ============================================================
pause
