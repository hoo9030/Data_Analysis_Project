@echo off
setlocal enableextensions enabledelayedexpansion

REM Move to project root (script directory)
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM Derive APP_NAME from folder name if not provided
for %%I in (.) do set "APP_NAME_FALLBACK=%%~nxI"
if not defined APP_NAME (
  set "APP_NAME=!APP_NAME_FALLBACK!"
  set "APP_NAME=!APP_NAME:_= !"
  set "APP_NAME=!APP_NAME:-= !"
)

set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

REM Create venv if missing
if not exist "%PYEXE%" (
  echo [INFO] Creating virtual environment...
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -m venv "%VENV%"
  ) else (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 (
      python -m venv "%VENV%"
    ) else (
      echo [ERROR] Python not found on PATH. Install Python or add it to PATH.
      pause
      exit /b 1
    )
  )
)

if not exist "%PYEXE%" (
  echo [ERROR] Failed to create virtual environment. Please verify Python installation.
  pause
  exit /b 1
)

REM Optionally skip installs for quick dry-run: set SKIP_INSTALL=1
if defined SKIP_INSTALL (
  echo [INFO] SKIP_INSTALL detected. Skipping dependency installation.
) else (
  echo [INFO] Upgrading pip and installing requirements...
  "%PYEXE%" -m pip install --upgrade pip
  if exist requirements.txt (
    "%PYEXE%" -m pip install -r requirements.txt
  )
)

REM Launch Streamlit app (can skip with SKIP_RUN=1)
if defined SKIP_RUN (
  echo [INFO] SKIP_RUN detected. Not launching Streamlit.
) else (
  echo [INFO] Launching Streamlit...
  "%PYEXE%" -m streamlit run app.py
)

endlocal
