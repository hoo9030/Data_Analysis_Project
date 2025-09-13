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
  - 李멸퀬: ?댁젣 猷⑦듃(`/`)??`/web`?쇰줈 由щ떎?대젆?몃맗?덈떎.

Dataset APIs (CSV)

- POST `http://127.0.0.1:8000/api/datasets` (form field: `file`)
- GET  `http://127.0.0.1:8000/api/datasets`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/preview?nrows=20`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/describe?limit=5000&include_all=true`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/download`
 - DELETE `http://127.0.0.1:8000/api/datasets/{id}`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/nulls?limit=100000` (寃곗륫移??꾪솴; limit 誘몄??????꾩껜)
 - POST `http://127.0.0.1:8000/api/datasets/{id}/cast` JSON: `{ "column": "col", "to": "int|float|string|datetime|bool|category", "mode": "coerce|strict", "out_id": "(?듭뀡)" }`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/distribution?column=col&bins=20&topk=20&limit=50000&dropna=true`
     - ?レ옄?? ?덉뒪?좉렇??bins)
     - 臾몄옄?? Top-K 鍮덈룄
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/corr?method=pearson&limit=50000` (?섏튂 而щ읆 ?곴??됰젹)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/sample.csv?rows=100` (?곸쐞 N?됱쓣 CSV濡??ㅼ슫濡쒕뱶)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/filter.csv?columns=col1,col2&limit=10000&query=price>10 and category=='A'`
    - 二쇱쓽: query??pandas DataFrame.query 臾몃쾿 ?ъ슜
  - POST `http://127.0.0.1:8000/api/datasets/{id}/fillna` JSON: `{ "columns": [..](?듭뀡), "strategy": "value|mean|median|mode", "value": any(?듭뀡), "out_id": "(?듭뀡)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/drop` JSON: `{ "columns": ["col1", "col2"], "out_id": "(?듭뀡)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/rename` JSON: `{ "mapping": { "old": "new", ... }, "out_id": "(?듭뀡)" }`
 - GET  `http://127.0.0.1:8000/api/info` (???뺣낫 JSON)

Web UI

- 寃쎈줈: `/web` (?뺤쟻 HTML/JS)
- 湲곕뒫: ?낅줈?? 紐⑸줉, 誘몃━蹂닿린, ?붿빟 ?듦퀎 ?뺤씤
  - 異붽?: 寃곗륫移??꾪솴 議고쉶, ???蹂?????곗씠?곗뀑 ?앹꽦)
  - 異붽?: 遺꾪룷 ?쒓컖??罹붾쾭??諛?李⑦듃 + ?뚯씠釉?, ?곴?愿怨??쒓컖???덊듃留?+ ?뚯씠釉?
  - 異붽?: ?섑뵆/?꾪꽣 CSV ?ㅼ슫濡쒕뱶 ??  - 異붽?: 湲곗큹 ?꾩쿂由?寃곗륫移?梨꾩슦湲? 而щ읆 ??젣/?대쫫蹂寃?
- ?? ??湲곕뒫??異붽?????`app/web/`??UI ?뱀뀡???④퍡 ?뺤옣?섏꽭??

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

