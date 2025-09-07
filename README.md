# 데이터 분석 스튜디오 (ASGI)

데이터 탐색·시각화·모델링을 위한 Python 기반 프로젝트입니다. 분석 로직은 Python 그대로 유지하고, 웹은 Django + FastAPI 통합 ASGI 앱으로 제공합니다.

## 빠른 시작

1) 환경 준비
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) 실행 (ASGI 통합 모드 권장)
```
run.bat combined
```

## 통합 실행(run.bat)

다음 모드를 지원합니다:

- Django: `run.bat django` — Django 개발 서버 실행 (http://127.0.0.1:8000)
- Combined: `run.bat combined` — Uvicorn으로 Django + FastAPI(및 Flask) 통합 실행
- Install: `run.bat install` — 가상환경 생성/유지 및 의존성 설치
- Migrate: `run.bat migrate` — Django 데이터베이스 마이그레이션

공통 환경변수:
- `DJANGO_ALLOWED_HOSTS`: Django 허용 호스트 목록(기본: localhost,127.0.0.1)
- `SKIP_INSTALL=1`: 의존성 설치를 건너뜀(빠른 재실행용)

예시:
```
set DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 & run.bat combined
```

## 아키텍처

- `/` (Django): 업로드 폼 및 기본 페이지
- `/api` (FastAPI): 분석 API (예: `POST /api/eda/summary`)
- `/legacy` (Flask): 예시용 헬스 체크
- 정적 파일: `backend/staticfiles` (배포 전 `collectstatic` 필요)

## 배포 팁

- 정적 파일 수집
```
cd backend
..\.venv\Scripts\python.exe manage.py collectstatic --noinput
```
- 권장 Uvicorn 실행 옵션
```
.\.venv\Scripts\python.exe -m uvicorn backend.asgi_combined:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
```
- 환경변수
  - `DJANGO_SECRET_KEY`: 필수, 강한 랜덤 값
  - `DJANGO_DEBUG=false`
  - `DJANGO_ALLOWED_HOSTS=example.com,127.0.0.1`
  - `DJANGO_CSRF_TRUSTED_ORIGINS=https://example.com`

## 프로젝트 구조 (요약)
```
backend/
  asgi_combined.py     # Django+FastAPI(+Flask) 통합 ASGI 앱
  config/              # Django 설정/URL
  studio/              # Django 앱(뷰/템플릿)
  templates/           # 템플릿
  static/              # 개발용 정적
  staticfiles/         # collectstatic 출력(배포용)
src/
  data_ops.py, eda_ops.py, ...  # 분석 로직
.run.bat                         # 통합 실행 스크립트
requirements.txt
```




