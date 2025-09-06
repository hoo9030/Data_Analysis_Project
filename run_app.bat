@echo off
setlocal enableextensions enabledelayedexpansion
REM UTF-8 콘솔 (한글 출력 개선)
chcp 65001 >nul 2>&1

REM 프로젝트 루트로 이동 (배치파일 위치 기준)
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

REM 가상환경이 없으면 생성
if not exist "%PYEXE%" (
  echo [INFO] 가상환경 생성 중...
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -m venv "%VENV%"
  ) else (
    python -m venv "%VENV%"
  )
)

if not exist "%PYEXE%" (
  echo [ERROR] 가상환경 생성 실패. Python 설치를 확인하세요.
  pause
  exit /b 1
)

REM pip 업데이트 및 의존성 설치
"%PYEXE%" -m pip install --upgrade pip
if exist requirements.txt (
  echo [INFO] 의존성 설치/확인 중...
  "%PYEXE%" -m pip install -r requirements.txt
)

REM Streamlit 앱 실행
echo [INFO] Streamlit 앱 실행 중...
"%PYEXE%" -m streamlit run app.py

endlocal
