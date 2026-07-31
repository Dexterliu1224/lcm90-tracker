@echo off
REM Pull the latest code and restart. Requires Git for Windows (git-scm.com).
REM ASCII-only on purpose - see START-Windows.bat.
setlocal
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
    echo [X] Git not found. Install it once from https://git-scm.com/
    pause
    exit /b 1
)
echo Pulling latest code...
git pull --ff-only
if errorlevel 1 (
    echo [X] git pull failed. Check network / login, then retry.
    pause
    exit /b 1
)
echo Done. Starting...
call START-Windows.bat
