Data Analysis Studio (FastAPI)

This project provides a unified, FastAPI‑based data analysis studio. It offers CSV ingestion with smart encoding/delimiter detection, EDA summaries, interactive visualization (Plotly), profiling (ydata‑profiling), web crawling helpers, and lightweight model evaluation — all accessible via HTTP APIs and a simple HTML UI.

Quick Start

1) Setup virtualenv and install deps

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Run the app

```
run.bat combined
```

The app serves:
- `/`          HTML Home (served directly by FastAPI)
- `/api/*`     All APIs (EDA, visualization, profiling, crawling, modeling)
- `/static/*`  Static assets

Environment

- `HOST` default `127.0.0.1`
- `PORT` default `8000`
- `RELOAD` set `0` to disable auto‑reload (default on)

Key Endpoints (selection)

- EDA summary (upload): `POST /api/eda/summary`
  - Form: `file`, `sep`, `decimal`, `encoding`, `corr_method`, filters (`filter_query`, `include_cols`, `limit_rows`), aggregation (`group_by`, `agg`, `value_cols`, `pivot_col`, `pivot_fill`)
- EDA visualize (upload): `POST /api/eda/visualize`
- Profiling (upload): `POST /api/profile/html`
- Sample variants: `GET /api/sample/summary`, `GET /api/sample/visualize`, `GET /api/sample/csv`
- Crawling: `GET /api/crawl/csv`, `GET /api/crawl/table`, `GET /api/crawl/html`
- Modeling (upload): `POST /api/model/evaluate` (auto classify/regress + CV)
- Modeling (sample): `GET /api/sample/model/evaluate`
- Orchestrator: `POST /api/run` — compose multiple steps (filters/aggregation/summary/visualize/profile/model) in a single request via JSON `spec`.

Notes

- The app now runs FastAPI‑only; previous Django/Flask mounts are removed.
- Legacy files and caches may remain on disk (e.g., `backend/config`, `backend/studio` pyc, `backend/db.sqlite3`). They are unused and can be deleted safely if not needed.
- Plotly rendering uses client‑side JS; image downloads are generated in the browser.

Dev Tips

- Launch directly with uvicorn if preferred:
  `.\.venv\Scripts\python.exe -m uvicorn backend.asgi_combined:app --host 127.0.0.1 --port 8000 --reload`
- CORS is permissive for local development. Harden before production.

