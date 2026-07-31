@echo off
REM =====================================================================
REM  LCM90 Visual Tracker - Windows launcher
REM
REM  This file is intentionally pure ASCII.  A .bat containing Chinese
REM  characters is read by cmd.exe using the system ANSI codepage (GBK on
REM  Chinese Windows); a UTF-8 encoded file then gets mis-parsed and the
REM  script dies before it does anything.  Keep this file English-only.
REM  All Chinese instructions live in README-Windows.txt.
REM =====================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo   LCM90 Visual Tracker
echo ============================================================
echo.
echo Working folder: %CD%
echo.

REM ---- warn about non-ASCII path (breaks pip/venv on some systems) ----
echo %CD%| findstr /R /C:"[^ -~]" >nul
if not errorlevel 1 (
    echo [!] WARNING: the folder path contains non-English characters.
    echo     If anything fails below, move this folder to a simple path
    echo     such as  C:\telescope-tracker  and try again.
    echo.
)

REM ---- locate a working Python 3 ----
set "PY="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PY=py -3"

if not defined PY (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo [X] Python 3 was not found.
    echo.
    echo     Install it from:  https://www.python.org/downloads/
    echo.
    echo     IMPORTANT: on the first installer screen, tick the checkbox
    echo     "Add Python to PATH" at the bottom.  Without it Windows
    echo     cannot find Python and this script will keep failing.
    echo.
    echo     After installing, close this window and run the script again.
    echo.
    pause
    exit /b 1
)

echo Found Python:
%PY% --version
echo.

REM ---- create the virtual environment on first run ----
if not exist ".venv\Scripts\python.exe" (
    echo [1/2] First run - setting up. This downloads a few hundred MB
    echo       and takes roughly 3-10 minutes depending on your connection.
    echo       Only needed once; later starts are instant.
    echo.

    %PY% -m venv .venv
    if errorlevel 1 (
        echo.
        echo [X] Failed to create the virtual environment.
        echo     Try running:   %PY% -m pip install --upgrade virtualenv
        echo.
        pause
        exit /b 1
    )

    echo       Upgrading pip...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    echo.
    echo       Installing packages - this is the slow part...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [X] Package installation failed.
        echo     Most often this is a network problem. Try a China mirror:
        echo.
        echo     .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        echo.
        pause
        exit /b 1
    )
    echo.
    echo       Setup complete.
    echo.
)

echo [2/2] Starting. Your browser should open in a few seconds.
echo       To stop the program: press Ctrl+C here, or close this window.
echo.

".venv\Scripts\python.exe" serve.py
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo [X] The program exited with error code %RC%.
    echo     Please send a screenshot of this window.
) else (
    echo Program closed.
)
pause
