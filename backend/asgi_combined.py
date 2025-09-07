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

# Import reusable analysis function
from eda_ops import basic_info  # type: ignore


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
):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data), sep=sep, decimal=decimal, encoding=encoding)
    info = basic_info(df)
    return {
        "rows": int(info.get("rows", len(df))),
        "columns": int(info.get("columns", df.shape[1] if not df.empty else 0)),
        "memory": str(info.get("memory", "")),
    }


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
