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
 - GET  `http://127.0.0.1:8000/api/info` (앱 정보 JSON)

Web UI

- 경로: `/web` (정적 HTML/JS)
- 기능: 업로드, 목록, 미리보기, 요약 통계 확인
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
