@echo off
setlocal enableextensions enabledelayedexpansion

REM Minimal runner for the clean FastAPI app
REM Usage:
REM   run.bat           - Create venv, install deps, run app
REM   set HOST/PORT/RELOAD=0 to customize

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

if not exist "%PYEXE%" (
  echo [INFO] Creating virtual environment...
  where py >nul 2>&1 && (
    py -m venv "%VENV%"
  ) || (
    where python >nul 2>&1 && (
      python -m venv "%VENV%"
    ) || (
      echo [ERROR] Python not found on PATH. Install Python or add it to PATH.
      exit /b 1
    )
  )
)

echo [INFO] Upgrading pip and installing requirements...
"%PYEXE%" -m pip install --upgrade pip
if exist requirements.txt (
  "%PYEXE%" -m pip install -r requirements.txt
) else (
  echo fastapi> requirements.txt
  echo uvicorn[standard]>> requirements.txt
  "%PYEXE%" -m pip install -r requirements.txt
)

set "HOST=%HOST%"
if not defined HOST set "HOST=127.0.0.1"
set "PORT=%PORT%"
if not defined PORT set "PORT=8000"
set "RELOAD_FLAG=--reload"
if "%RELOAD%"=="0" set "RELOAD_FLAG="

REM Auto-open Web UI unless OPEN=0
set "OPEN=%OPEN%"
if not defined OPEN set "OPEN=1"
if not "%OPEN%"=="0" (
  REM Open in background after a short delay; fallback if PowerShell not available
  where powershell >nul 2>&1 && (
    start "" powershell -NoProfile -Command "Start-Sleep -Seconds 2; Start-Process 'http://%HOST%:%PORT%/web'"
  ) || (
    start "" "http://%HOST%:%PORT%/web"
  )
)

echo [INFO] Launching ASGI at http://%HOST%:%PORT% %RELOAD_FLAG%
"%PYEXE%" -m uvicorn app.main:app %RELOAD_FLAG% --host %HOST% --port %PORT%

endlocal
