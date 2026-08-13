@echo off
REM ===========================================================================
REM  LaunchGuard AI - double-click this file to run the app.
REM
REM  Finds Python, then hands over to launch.py which does the real work:
REM  creates a private environment, installs dependencies once, asks for an
REM  API key if there isn't one, and opens the browser.
REM ===========================================================================

title LaunchGuard AI
cd /d "%~dp0"

echo.
echo   Starting LaunchGuard AI...
echo.

REM --- find a usable Python. "py" is the Windows launcher and is most reliable,
REM --- but a plain "python" install is common too.
set PYCMD=

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
    if %errorlevel%==0 set PYCMD=py -3
)

if "%PYCMD%"=="" (
    where python >nul 2>&1
    if %errorlevel%==0 (
        python -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
        if %errorlevel%==0 set PYCMD=python
    )
)

if "%PYCMD%"=="" (
    echo   ==========================================================
    echo    Python 3.9 or newer was not found on this computer.
    echo.
    echo    Install it from  https://www.python.org/downloads/
    echo    IMPORTANT: tick "Add Python to PATH" during installation,
    echo    then close this window and run START.bat again.
    echo   ==========================================================
    echo.
    pause
    exit /b 1
)

%PYCMD% launch.py

REM launch.py handles its own pause on exit, so nothing needed here.
