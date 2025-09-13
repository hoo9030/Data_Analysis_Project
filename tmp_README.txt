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
  - 참고: ?�제 루트(`/`)??`/web`?�로 리다?�렉?�됩?�다.

Dataset APIs (CSV)

- POST `http://127.0.0.1:8000/api/datasets` (form field: `file`)
- GET  `http://127.0.0.1:8000/api/datasets`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/preview?nrows=20`
- GET  `http://127.0.0.1:8000/api/datasets/{id}/describe?limit=5000&include_all=true`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/download`
 - DELETE `http://127.0.0.1:8000/api/datasets/{id}`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/nulls?limit=100000` (결측�??�황; limit 미�??????�체)
 - POST `http://127.0.0.1:8000/api/datasets/{id}/cast` JSON: `{ "column": "col", "to": "int|float|string|datetime|bool|category", "mode": "coerce|strict", "out_id": "(?�션)" }`
 - GET  `http://127.0.0.1:8000/api/datasets/{id}/distribution?column=col&bins=20&topk=20&limit=50000&dropna=true`
     - ?�자?? ?�스?�그??bins)
     - 문자?? Top-K 빈도
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/corr?method=pearson&limit=50000` (?�치 컬럼 ?��??�렬)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/sample.csv?rows=100` (?�위 N?�을 CSV�??�운로드)
  - GET  `http://127.0.0.1:8000/api/datasets/{id}/filter.csv?columns=col1,col2&limit=10000&query=price>10 and category=='A'`
    - 주의: query??pandas DataFrame.query 문법 ?�용
  - POST `http://127.0.0.1:8000/api/datasets/{id}/fillna` JSON: `{ "columns": [..](?�션), "strategy": "value|mean|median|mode", "value": any(?�션), "out_id": "(?�션)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/drop` JSON: `{ "columns": ["col1", "col2"], "out_id": "(?�션)" }`
  - POST `http://127.0.0.1:8000/api/datasets/{id}/rename` JSON: `{ "mapping": { "old": "new", ... }, "out_id": "(?�션)" }`
 - GET  `http://127.0.0.1:8000/api/info` (???�보 JSON)

Web UI

- 경로: `/web` (?�적 HTML/JS)
- 기능: ?�로?? 목록, 미리보기, ?�약 ?�계 ?�인
  - 추�?: 결측�??�황 조회, ?�??변?????�이?�셋 ?�성)
  - 추�?: 분포 ?�각??캔버??�?차트 + ?�이�?, ?��?관�??�각???�트�?+ ?�이�?
  - 추�?: ?�플/?�터 CSV ?�운로드 ??  - 추�?: 기초 ?�처�?결측�?채우�? 컬럼 ??��/?�름변�?
- ?? ??기능??추�?????`app/web/`??UI ?�션???�께 ?�장?�세??

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

