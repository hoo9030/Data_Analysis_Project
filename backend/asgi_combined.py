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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from django.core.asgi import get_asgi_application
from flask import Flask
import plotly.express as px
from starlette.responses import Response


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


# ----- CSV utilities ---------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard limit


def _detect_encoding(data: bytes) -> str | None:
    try:
        import chardet  # type: ignore

        res = chardet.detect(data)
        enc = res.get("encoding") if isinstance(res, dict) else None
        return enc
    except Exception:
        return None


def _smart_read_csv(
    data: bytes,
    *,
    sep: str | None = None,
    decimal: str | None = None,
    encoding: str | None = None,
):
    """Try to read CSV with best-effort detection for encoding and delimiter.

    Returns (df, used_encoding, used_sep).
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"File too large (>{MAX_UPLOAD_BYTES // (1024*1024)} MB)")

    # Build encoding candidates
    enc_candidates: list[str] = []
    if encoding:
        enc_candidates.append(encoding)
    detected = _detect_encoding(data)
    if detected and detected.lower() not in {e.lower() for e in enc_candidates}:
        enc_candidates.append(detected)
    # Common fallbacks
    for e in ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]:
        if e.lower() not in {x.lower() for x in enc_candidates}:
            enc_candidates.append(e)

    last_err: Exception | None = None
    used_enc: str | None = None
    used_sep: str | None = None

    for enc in enc_candidates:
        try:
            # If sep not provided, let pandas infer using python engine
            if sep:
                df = pd.read_csv(io.BytesIO(data), sep=sep, decimal=decimal or ".", encoding=enc)
                used_sep = sep
            else:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", decimal=decimal or ".", encoding=enc)
                used_sep = None
            used_enc = enc
            # If header collapsed to single wide column, try csv.Sniffer
            if df.shape[1] == 1 and not sep:
                import csv
                try:
                    preview = data[:4096].decode(enc, errors="ignore")
                    dialect = csv.Sniffer().sniff(preview)
                    delim = dialect.delimiter
                    df = pd.read_csv(io.BytesIO(data), sep=delim, decimal=decimal or ".", encoding=enc)
                    used_sep = delim
                except Exception:
                    pass
            return df, used_enc, used_sep
        except Exception as e:
            last_err = e
            continue

    # Final fallback: decode ignoring errors and try generic read
    try:
        txt = data.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(txt), sep=sep or None, engine="python")
        return df, used_enc or "utf-8", used_sep
    except Exception:
        pass

    msg = "Failed to parse CSV. Try specifying encoding or delimiter."
    if last_err:
        msg += f" Last error: {type(last_err).__name__}: {last_err}"
    raise HTTPException(status_code=400, detail=msg)


@api.post("/eda/summary")
async def eda_summary(
    file: UploadFile = File(...),
    sep: str = Form(","),
    decimal: str = Form("."),
    encoding: str = Form("utf-8"),
    max_corr_dims: int = Form(8),
):
    data = await file.read()
    # Blank values act as auto-detect
    auto_sep = sep or None
    auto_enc = encoding or None
    df, used_enc, used_sep = _smart_read_csv(data, sep=auto_sep, decimal=decimal or ".", encoding=auto_enc)
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
        "detected": {"encoding": used_enc, "sep": used_sep},
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


def _build_figure(df: pd.DataFrame, chart: str, x: str | None, y: str | None, color: str | None, bins: int = 30):
    chart = (chart or "").lower()
    if chart in ("histogram", "hist"):
        if not x:
            raise HTTPException(400, detail="x is required for histogram")
        fig = px.histogram(df, x=x, color=color, nbins=bins)
    elif chart in ("bar", "bar_count", "count"):
        if not x:
            raise HTTPException(400, detail="x is required for bar_count")
        fig = px.histogram(df, x=x, color=color)
    elif chart in ("scatter", "point"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for scatter")
        fig = px.scatter(df, x=x, y=y, color=color)
    elif chart == "box":
        if not y:
            raise HTTPException(400, detail="y is required for box")
        fig = px.box(df, x=x, y=y, color=color)
    elif chart in ("line", "timeseries", "time"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for line")
        fig = px.line(df.sort_values(by=x), x=x, y=y, color=color)
    elif chart in ("kde", "density", "density_contour"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for density")
        fig = px.density_contour(df, x=x, y=y, color=color)
    elif chart in ("hexbin", "hist2d", "density_heatmap"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for 2D density")
        fig = px.density_heatmap(df, x=x, y=y, nbinsx=bins, nbinsy=bins, color_continuous_scale="Viridis")
    else:
        raise HTTPException(400, detail="Unsupported chart. Use histogram|bar_count|scatter|box|line|density|hist2d")
    return fig.to_dict()


@api.post("/eda/visualize")
async def eda_visualize(
    file: UploadFile = File(...),
    chart: str = Form(...),
    x: str | None = Form(None),
    y: str | None = Form(None),
    color: str | None = Form(None),
    bins: int = Form(30),
):
    try:
        data = await file.read()
        df, _, _ = _smart_read_csv(data)
        spec = _build_figure(df, chart=chart, x=x, y=y, color=color, bins=bins)
        return {"figure": spec}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@api.get("/sample/visualize")
async def sample_visualize(
    chart: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    bins: int = 30,
    rows: int = 500,
    seed: int = 42,
):
    df = generate_sample_data(rows=rows, seed=seed)
    spec = _build_figure(df, chart=chart, x=x, y=y, color=color, bins=bins)
    return {"figure": spec}


@api.post("/profile/html")
async def profile_html(
    file: UploadFile = File(...),
    minimal: bool = Form(True),
    sample_n: int | None = Form(2000),
):
    try:
        from profile_ops import generate_profile_html  # type: ignore
    except Exception as e:
        raise HTTPException(500, detail=f"Profiling import error: {e}")

    try:
        data = await file.read()
        df = pd.read_csv(io.BytesIO(data))
        html = generate_profile_html(df, minimal=bool(minimal), sample_n=sample_n)
        return Response(content=html, media_type="text/html; charset=utf-8")
    except Exception as e:
        raise HTTPException(400, detail=str(e))


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


