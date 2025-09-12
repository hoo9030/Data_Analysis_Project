Data Analysis Studio — Clean Start (FastAPI)

Quick Start

1) Run the app (venv + deps auto-handled):

```
run.bat
```

2) Open in browser:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/api/healthz
- http://127.0.0.1:8000/web  (웹 UI)
  - 참고: 이제 루트(`/`)는 `/web`으로 리다이렉트됩니다.

Dataset APIs (CSV)

- POST `http://127.0.0.1:8000/api/datasets` (form field: `file`)
- GET  `http://127.0.0.1:8000/api/datasets`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/preview?nrows=20`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/describe?limit=5000&include_all=true`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/download`
 - DELETE `http://127.0.0.1:8000/api/datasets/{id}`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/nulls?limit=100000` (결측치 현황; limit 미지정 시 전체)
 - POST `http://127.0.0.1:8000/api/datasets/{id}/cast` JSON: `{ "column": "col", "to": "int|float|string|datetime|bool|category", "mode": "coerce|strict", "out_id": "(옵션)" }`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/distribution?column=col&bins=20&topk=20&limit=50000&dropna=true`
     - 숫자형: 히스토그램(bins)
     - 문자형: Top-K 빈도
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/corr?method=pearson&limit=50000` (수치 컬럼 상관행렬)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/sample.csv?rows=100` (상위 N행을 CSV로 다운로드)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/filter.csv?columns=col1,col2&limit=10000&query=price>10 and category=='A'`
    - 주의: query는 pandas DataFrame.query 문법 사용
 - GET  `http://127.0.0.1:8000/api/info` (앱 정보 JSON)

Web UI

- 경로: `/web` (정적 HTML/JS)
- 기능: 업로드, 목록, 미리보기, 요약 통계 확인
  - 추가: 결측치 현황 조회, 타입 변환(새 데이터셋 생성)
  - 추가: 분포(숫자형 히스토그램/문자형 Top-K), 상관관계(테이블)
  - 추가: 샘플/필터 CSV 다운로드 폼
- 팁: 새 기능을 추가할 때 `app/web/`의 UI 섹션을 함께 확장하세요.

Environment

- HOST default 127.0.0.1
- PORT default 8000
- RELOAD set 0 to disable auto-reload (default on)
- OPEN set 0 to prevent auto-opening the Web UI (default on)

Notes

- New code lives under `app/`.
- This scaffold intentionally includes only a health endpoint. Add more under `app/api/*` and include in `app/api/__init__.py`.
- Datasets are stored under `./data/datasets/<id>.csv` with metadata at `./data/metadata.json`.
- If you are deleting legacy files, keep `app/`, `run.bat`, and optionally `.gitignore`.

Dev Tips

- Launch directly with uvicorn if preferred:
  `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
