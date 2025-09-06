@echo off
setlocal
cd /d "%~dp0\backend"

if not defined DJANGO_ALLOWED_HOSTS set DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

set "PYEXE=..\.venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo [ERROR] Python venv not found. Please create .venv first.
  exit /b 1
)

"%PYEXE%" manage.py runserver 127.0.0.1:8000

endlocal
