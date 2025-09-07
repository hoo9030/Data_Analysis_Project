@echo off
setlocal enableextensions enabledelayedexpansion

REM Unified runner for Studio (Streamlit), Django, and Combined (ASGI)
REM Usage:
REM   run.bat studio    - Launch Streamlit Studio
REM   run.bat django    - Launch Django dev server
REM   run.bat combined  - Launch combined ASGI (Django+FastAPI+Flask)
REM   run.bat install   - Upgrade pip and install requirements
REM   run.bat migrate   - Run Django migrations
REM   run.bat           - Interactive menu

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "HOST=%HOST%"
if not defined HOST set "HOST=127.0.0.1"
set "PORT=%PORT%"
if not defined PORT set "PORT=8000"

goto :dispatch

REM Ensure venv exists (create if missing)
:ensure_venv
if exist "%PYEXE%" goto :eof_ensure_venv
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
      exit /b 1
    )
  )
if not exist "%PYEXE%" (
  echo [ERROR] Failed to create virtual environment. Please verify Python installation.
  exit /b 1
)
:eof_ensure_venv
goto :eof

REM Install/upgrade deps unless SKIP_INSTALL is set
:ensure_deps
if defined SKIP_INSTALL (
  echo [INFO] SKIP_INSTALL detected. Skipping dependency installation.
) else (
  echo [INFO] Upgrading pip and installing requirements...
  "%PYEXE%" -m pip install --upgrade pip
  if exist requirements.txt (
    "%PYEXE%" -m pip install -r requirements.txt
  )
)
goto :eof

REM APP_NAME fallback (from folder name), replacing _ and - with spaces
:ensure_app_name
for %%I in (.) do set "APP_NAME_FALLBACK=%%~nxI"
if not defined APP_NAME (
  set "APP_NAME=!APP_NAME_FALLBACK!"
  set "APP_NAME=!APP_NAME:_= !"
  set "APP_NAME=!APP_NAME:-= !"
)
goto :eof

REM Default ALLOWED_HOSTS for Django
:ensure_hosts
if not defined DJANGO_ALLOWED_HOSTS set "DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1"
goto :eof

REM Subcommands ---------------------------------------------------------------
:cmd_install
call :ensure_venv
call :ensure_deps
exit /b %ERRORLEVEL%

:cmd_migrate
call :ensure_venv
call :ensure_deps
pushd backend
"%PYEXE%" manage.py migrate --noinput
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%

:cmd_studio
call :ensure_venv
call :ensure_deps
call :ensure_app_name
echo [INFO] Launching Streamlit Studio (APP_NAME="%APP_NAME%").
"%PYEXE%" -m streamlit run app.py
exit /b %ERRORLEVEL%

:cmd_django
call :ensure_venv
call :ensure_deps
call :ensure_hosts
echo [INFO] Launching Django dev server at http://%HOST%:%PORT%
pushd backend
"%PYEXE%" manage.py runserver %HOST%:%PORT%
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%

:cmd_combined
call :ensure_venv
call :ensure_deps
call :ensure_hosts
set "RELOAD_FLAG=--reload"
if "%RELOAD%"=="0" set "RELOAD_FLAG="
echo [INFO] Launching combined ASGI at http://%HOST%:%PORT% %RELOAD_FLAG%
"%PYEXE%" -m uvicorn backend.asgi_combined:app %RELOAD_FLAG% --host %HOST% --port %PORT%
exit /b %ERRORLEVEL%

REM Interactive menu ----------------------------------------------------------
:menu
echo.
echo Select mode:
echo   [1] Studio   (Streamlit)
echo   [2] Django   (runserver)
echo   [3] Combined (Uvicorn ASGI)
set /p CHOICE=Enter choice [1-3]: 
if "%CHOICE%"=="1" goto :cmd_studio
if "%CHOICE%"=="2" goto :cmd_django
if "%CHOICE%"=="3" goto :cmd_combined
echo Invalid choice.
exit /b 1

REM Argument routing ----------------------------------------------------------
:dispatch
if "%~1"=="" goto :cmd_combined
set ARG=%~1
if /i "%ARG%"=="studio"   goto :cmd_studio
if /i "%ARG%"=="django"   goto :cmd_django
if /i "%ARG%"=="combined" goto :cmd_combined
if /i "%ARG%"=="install"  goto :cmd_install
if /i "%ARG%"=="migrate"  goto :cmd_migrate
if /i "%ARG%"=="menu"     goto :menu
if /i "%ARG%"=="help"     goto :help

echo Usage: run.bat ^<studio^|django^|combined^|install^|migrate^|menu^|help^>
exit /b 1

endlocal

:help
echo.
echo Usage: run.bat ^<studio^|django^|combined^|install^|migrate^|menu^|help^>
echo   studio   - Launch Streamlit app
echo   django   - Launch Django runserver (HOST/PORT env supported)
echo   combined - Launch Uvicorn ASGI (HOST/PORT/RELOAD env supported)
echo   install  - Create venv and install requirements
echo   migrate  - Run Django migrations
echo   menu     - Interactive menu
echo.
echo Env:
echo   HOST=127.0.0.1  PORT=8000   RELOAD=1  DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
exit /b 0
