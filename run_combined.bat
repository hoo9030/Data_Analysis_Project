@echo off
setlocal enableextensions enabledelayedexpansion

REM Launch combined ASGI app (Django + FastAPI + Flask)
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo [ERROR] .venv not found. Create it and install requirements first.
  exit /b 1
)

set "DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1"
"%PYEXE%" -m uvicorn backend.asgi_combined:app --reload --host 127.0.0.1 --port 8000

endlocal
