"""Combined ASGI application: Django + FastAPI + Flask.

Run (dev):
  uvicorn backend.asgi_combined:app --reload
"""

from __future__ import annotations

import os
import sys
import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from django.core.asgi import get_asgi_application
from flask import Flask


# Django settings
# When launched from project root, ensure Django loads settings from backend/config
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.config.settings")

# Resolve paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import reusable analysis function
from eda_ops import basic_info, missing_summary, correlation_matrix  # type: ignore
from data_ops import generate_sample_data, detect_column_types  # type: ignore


# Django ASGI app mounted at '/'
django_asgi = get_asgi_application()


# FastAPI app mounted at '/api'
api = FastAPI(title="Studio API", version="0.1.0")


@api.post("/eda/summary")
async def eda_summary(
    file: UploadFile = File(...),
    sep: str = Form(","),
    decimal: str = Form("."),
    encoding: str = Form("utf-8"),
    max_corr_dims: int = Form(8),
):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data), sep=sep, decimal=decimal, encoding=encoding)
    info = basic_info(df)
    # Missing summary
    ms_df = missing_summary(df)
    ms_list = (
        ms_df.to_dict(orient="records")
        if hasattr(ms_df, "to_dict")
        else []
    )
    # Column types
    num_cols, cat_cols, dt_cols = detect_column_types(df)
    # Correlation (limit dimensions)
    corr_payload = None
    if len(num_cols) >= 2:
        use_cols = num_cols[: max(2, min(max_corr_dims, len(num_cols)))]
        corr = correlation_matrix(df[use_cols])
        corr_payload = {
            "labels": list(corr.columns),
            "matrix": corr.values.tolist(),
        }

    return {
        "rows": int(info.get("rows", len(df))),
        "columns": int(info.get("columns", df.shape[1] if not df.empty else 0)),
        "memory": str(info.get("memory", "")),
        "columns_info": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "datetime": dt_cols,
        },
        "missing": ms_list,
        "corr": corr_payload,
    }


@api.get("/sample/summary")
async def sample_summary(rows: int = 500, seed: int = 42, max_corr_dims: int = 8):
    df = generate_sample_data(rows=rows, seed=seed)
    info = basic_info(df)
    ms_df = missing_summary(df)
    ms_list = ms_df.to_dict(orient="records")
    num_cols, cat_cols, dt_cols = detect_column_types(df)
    corr_payload = None
    if len(num_cols) >= 2:
        use_cols = num_cols[: max(2, min(max_corr_dims, len(num_cols)))]
        corr = correlation_matrix(df[use_cols])
        corr_payload = {
            "labels": list(corr.columns),
            "matrix": corr.values.tolist(),
        }
    return {
        "rows": int(info.get("rows", len(df))),
        "columns": int(info.get("columns", df.shape[1] if not df.empty else 0)),
        "memory": str(info.get("memory", "")),
        "columns_info": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "datetime": dt_cols,
        },
        "missing": ms_list,
        "corr": corr_payload,
    }


@api.get("/sample/csv")
async def sample_csv(rows: int = 500, seed: int = 42):
    from starlette.responses import StreamingResponse
    import io as _io

    df = generate_sample_data(rows=rows, seed=seed)
    csv_text = df.to_csv(index=False)
    buf = _io.BytesIO(csv_text.encode("utf-8"))
    headers = {
        "Content-Disposition": f"attachment; filename=sample_{rows}_{seed}.csv"
    }
    return StreamingResponse(buf, media_type="text/csv; charset=utf-8", headers=headers)


# Optional: small Flask app mounted at '/legacy'
flask_app = Flask(__name__)


@flask_app.get("/health")
def health():  # pragma: no cover - tiny example
    return {"ok": True}


# Starlette root that mounts each framework
static_dir = BACKEND_DIR / "staticfiles"
if not static_dir.exists():
    # Fallback to app static dir for dev
    static_dir = BACKEND_DIR / "static"

routes = [
    Mount("/static", app=StaticFiles(directory=str(static_dir), html=False), name="static"),
    Mount("/api", app=api),
    Mount("/legacy", app=WSGIMiddleware(flask_app)),
    Mount("/", app=django_asgi),
]

app = Starlette(routes=routes)

# CORS (allow local dev tools; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
