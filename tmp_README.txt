Data Analysis Studio ??Clean Start (FastAPI)

Quick Start

1) Run the app (venv + deps auto-handled):

```
run.bat
```

2) Open in browser:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/api/healthz
- http://127.0.0.1:8000/web  (??UI)
  - ì°¸ê³ : ?´ì œ ë£¨íŠ¸(`/`)??`/web`?¼ë¡œ ë¦¬ë‹¤?´ë ‰?¸ë©?ˆë‹¤.

Dataset APIs (CSV)

- POST `http://127.0.0.1:8000/api/datasets` (form field: `file`)
- GET  `http://127.0.0.1:8000/api/datasets`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/preview?nrows=20`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/describe?limit=5000&include_all=true`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/download`
 - DELETE `http://127.0.0.1:8000/api/datasets/{id}`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/nulls?limit=100000` (ê²°ì¸¡ì¹??„í™©; limit ë¯¸ì??????„ì²´)
 - POST `http://127.0.0.1:8000/api/datasets/{id}/cast` JSON: `{ "column": "col", "to": "int|float|string|datetime|bool|category", "mode": "coerce|strict", "out_id": "(?µì…˜)" }`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/distribution?column=col&bins=20&topk=20&limit=50000&dropna=true`
     - ?«ì?? ?ˆìŠ¤? ê·¸??bins)
     - ë¬¸ì?? Top-K ë¹ˆë„
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/corr?method=pearson&limit=50000` (?˜ì¹˜ ì»¬ëŸ¼ ?ê??‰ë ¬)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/sample.csv?rows=100` (?ìœ„ N?‰ì„ CSVë¡??¤ìš´ë¡œë“œ)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/filter.csv?columns=col1,col2&limit=10000&query=price>10 and category=='A'`
    - ì£¼ì˜: query??pandas DataFrame.query ë¬¸ë²• ?¬ìš©
  - POST `http://127.0.0.1:8000/api/datasets/{id}/fillna` JSON: `{ "columns": [..](?µì…˜), "strategy": "value|mean|median|mode", "value": any(?µì…˜), "out_id": "(?µì…˜)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/drop` JSON: `{ "columns": ["col1", "col2"], "out_id": "(?µì…˜)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/rename` JSON: `{ "mapping": { "old": "new", ... }, "out_id": "(?µì…˜)" }`
 - GET  `http://127.0.0.1:8000/api/info` (???•ë³´ JSON)

Web UI

- ê²½ë¡œ: `/web` (?•ì  HTML/JS)
- ê¸°ëŠ¥: ?…ë¡œ?? ëª©ë¡, ë¯¸ë¦¬ë³´ê¸°, ?”ì•½ ?µê³„ ?•ì¸
  - ì¶”ê?: ê²°ì¸¡ì¹??„í™© ì¡°íšŒ, ?€??ë³€?????°ì´?°ì…‹ ?ì„±)
  - ì¶”ê?: ë¶„í¬ ?œê°??ìº”ë²„??ë°?ì°¨íŠ¸ + ?Œì´ë¸?, ?ê?ê´€ê³??œê°???ˆíŠ¸ë§?+ ?Œì´ë¸?
  - ì¶”ê?: ?˜í”Œ/?„í„° CSV ?¤ìš´ë¡œë“œ ??  - ì¶”ê?: ê¸°ì´ˆ ?„ì²˜ë¦?ê²°ì¸¡ì¹?ì±„ìš°ê¸? ì»¬ëŸ¼ ?? œ/?´ë¦„ë³€ê²?
- ?? ??ê¸°ëŠ¥??ì¶”ê?????`app/web/`??UI ?¹ì…˜???¨ê»˜ ?•ì¥?˜ì„¸??

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

