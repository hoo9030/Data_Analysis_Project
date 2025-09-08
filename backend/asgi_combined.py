"""Combined ASGI application: Django + FastAPI + Flask.

Run (dev):
  uvicorn backend.asgi_combined:app --reload
"""

from __future__ import annotations

import os
import sys
import io
from pathlib import Path
import socket
import ipaddress
from urllib.parse import urlparse

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
import httpx
from bs4 import BeautifulSoup


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


# ----- Simple SSRF guard and fetch helpers ----------------------------------
def _is_private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
        for family, _, _, _, sockaddr in infos:
            ip = ipaddress.ip_address(sockaddr[0] if isinstance(sockaddr, tuple) else sockaddr)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
    except Exception:
        return True
    return False


async def _fetch_bytes(url: str, *, timeout: float = 15.0) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(400, detail="Only http/https schemes are allowed")
    if not parsed.netloc:
        raise HTTPException(400, detail="Invalid URL")
    host = parsed.hostname or ""
    if _is_private_host(host):
        raise HTTPException(400, detail="Refusing to fetch private/loopback addresses")

    headers = {"User-Agent": "StudioBot/1.0 (+https://example.local)"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


# ----- DataFrame filtering helpers ------------------------------------------
def _parse_cols(cols: str | None, df: pd.DataFrame) -> list[str] | None:
    if not cols:
        return None
    items = [c.strip() for c in cols.split(",") if c.strip()]
    return [c for c in items if c in df.columns]


def _apply_filters(
    df: pd.DataFrame,
    *,
    include_cols: str | None = None,
    filter_query: str | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    dfx = df
    # include columns first
    cols = _parse_cols(include_cols, dfx)
    if cols:
        dfx = dfx[cols]
    # apply query if provided
    if filter_query:
        q = str(filter_query).strip()
        if len(q) > 0:
            try:
                # Try numexpr first (safer), fallback to python for strings
                dfx = dfx.query(q, engine="numexpr")
            except Exception:
                dfx = dfx.query(q, engine="python")
    # limit rows
    if isinstance(limit_rows, int) and limit_rows and limit_rows > 0:
        dfx = dfx.head(limit_rows)
    return dfx


ALLOWED_AGG = {"count", "mean", "sum", "min", "max", "median", "nunique"}


def _apply_aggregation(
    df: pd.DataFrame,
    *,
    group_by: str | None = None,
    agg: str | None = None,
    value_cols: str | None = None,
    pivot_col: str | None = None,
    pivot_fill: float | int | None = None,
) -> pd.DataFrame:
    # No-op if nothing requested
    if not group_by and not pivot_col:
        return df
    agg_func = (agg or "mean").lower()
    if agg_func not in ALLOWED_AGG:
        agg_func = "mean"
    gb_cols = _parse_cols(group_by, df) or []
    val_cols = _parse_cols(value_cols, df)
    if val_cols is None:
        # default to numeric columns
        val_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not val_cols:
        return df

    # If pivot requested, compute directly from base df to retain pivot column
    if pivot_col:
        if pivot_col not in df.columns:
            return df
        val = val_cols[0]
        idx = gb_cols if gb_cols else None
        try:
            pt = df.pivot_table(
                index=idx,
                columns=pivot_col,
                values=val,
                aggfunc=agg_func,
                fill_value=pivot_fill,
            )
            try:
                pt.columns = [str(c) for c in pt.columns.to_flat_index()]
            except Exception:
                pt.columns = [str(c) for c in pt.columns]
            return pt.reset_index() if idx is not None else pt.reset_index(drop=True)
        except Exception:
            return df

    # Otherwise plain groupby if requested
    if gb_cols:
        try:
            out = (
                df.groupby(gb_cols, dropna=False)[val_cols]
                .agg(agg_func)
                .reset_index()
            )
            return out
        except Exception:
            return df

    return df


def _preview_records(df: pd.DataFrame, max_rows: int = 50):
    try:
        n = max(1, min(int(max_rows or 50), len(df)))
    except Exception:
        n = min(50, len(df))
    if n <= 0:
        n = 1
    pre = df.head(n).copy()
    # Convert datetime-like columns to strings for JSON safety
    try:
        dt_cols = pre.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
        if dt_cols:
            pre[dt_cols] = pre[dt_cols].astype(str)
    except Exception:
        pass
    # Replace NaN with None
    try:
        pre = pre.where(pd.notna(pre), None)
    except Exception:
        pass
    try:
        return pre.to_dict(orient="records")
    except Exception:
        # Fallback: stringify all
        return pre.astype(str).to_dict(orient="records")


def _safe_float(x):
    try:
        import numpy as _np  # type: ignore
        if x is None:
            return None
        if pd.isna(x):
            return None
        if isinstance(x, (_np.floating, _np.integer)):
            return float(x)
        return float(x) if isinstance(x, (int, float)) else None
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def _column_summaries(df: pd.DataFrame, top_n: int = 5) -> dict:
    out: dict = {"numeric": [], "categorical": [], "datetime": []}
    try:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for col in num_cols:
            s = df[col]
            try:
                desc = s.describe()
            except Exception:
                desc = {}
            rec = {
                "column": col,
                "count": int((~s.isna()).sum()),
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "p25": _safe_float(desc.get("25%")),
                "median": _safe_float(desc.get("50%")),
                "p75": _safe_float(desc.get("75%")),
                "max": _safe_float(desc.get("max")),
                "unique": int(s.nunique(dropna=True)),
                "missing": int(s.isna().sum()),
            }
            out["numeric"].append(rec)

        # Treat non-number as categorical/datetime split
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
        for col in dt_cols:
            s = df[col]
            try:
                vmin = s.min()
                vmax = s.max()
            except Exception:
                vmin = vmax = None
            rec = {
                "column": col,
                "min": str(vmin) if vmin is not None else None,
                "max": str(vmax) if vmax is not None else None,
                "unique": int(s.nunique(dropna=True)),
                "missing": int(s.isna().sum()),
            }
            out["datetime"].append(rec)

        # Categorical: everything else excluding numeric/datetime
        exclude = set(num_cols) | set(dt_cols)
        cat_cols = [c for c in df.columns if c not in exclude]
        for col in cat_cols:
            s = df[col].astype(str, errors="ignore") if hasattr(df[col], "astype") else df[col]
            vc = s.value_counts(dropna=True)
            top = []
            total = max(1, len(s))
            for v, cnt in vc.head(top_n).items():
                try:
                    pct = round(float(cnt) / float(total), 4)
                except Exception:
                    pct = None
                top.append({"value": str(v), "count": int(cnt), "pct": pct})
            rec = {
                "column": col,
                "unique": int(s.nunique(dropna=True)),
                "missing": int(pd.isna(df[col]).sum()) if hasattr(pd, "isna") else 0,
                "top": top,
            }
            out["categorical"].append(rec)
    except Exception:
        pass
    return out


ALLOWED_CORR = {"pearson", "spearman", "kendall"}


@api.post("/eda/summary")
async def eda_summary(
    file: UploadFile = File(...),
    sep: str = Form(","),
    decimal: str = Form("."),
    encoding: str = Form("utf-8"),
    max_corr_dims: int = Form(8),
    corr_method: str = Form("pearson"),
    filter_query: str | None = Form(None),
    include_cols: str | None = Form(None),
    limit_rows: int | None = Form(None),
    group_by: str | None = Form(None),
    agg: str | None = Form(None),
    value_cols: str | None = Form(None),
    pivot_col: str | None = Form(None),
    pivot_fill: float | None = Form(None),
):
    data = await file.read()
    # Blank values act as auto-detect
    auto_sep = sep or None
    auto_enc = encoding or None
    df, used_enc, used_sep = _smart_read_csv(data, sep=auto_sep, decimal=decimal or ".", encoding=auto_enc)
    # Apply optional filters then aggregation
    df = _apply_filters(df, include_cols=include_cols, filter_query=filter_query, limit_rows=limit_rows)
    df = _apply_aggregation(df, group_by=group_by, agg=agg, value_cols=value_cols, pivot_col=pivot_col, pivot_fill=pivot_fill)
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
    method = (corr_method or "pearson").lower()
    if method not in ALLOWED_CORR:
        raise HTTPException(400, detail=f"Invalid corr_method. Use one of {sorted(ALLOWED_CORR)}")
    corr_payload = None
    if len(num_cols) >= 2:
        use_cols = num_cols[: max(2, min(max_corr_dims, len(num_cols)))]
        corr = correlation_matrix(df[use_cols], method=method)
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
        "columns_stats": _column_summaries(df, top_n=5),
        "missing": ms_list,
        "corr": corr_payload,
        "corr_method": method,
        "detected": {"encoding": used_enc, "sep": used_sep},
        "preview": _preview_records(df, max_rows=50),
    }


@api.get("/sample/summary")
async def sample_summary(
    rows: int = 500,
    seed: int = 42,
    max_corr_dims: int = 8,
    corr_method: str = "pearson",
    filter_query: str | None = None,
    include_cols: str | None = None,
    limit_rows: int | None = None,
    group_by: str | None = None,
    agg: str | None = None,
    value_cols: str | None = None,
    pivot_col: str | None = None,
    pivot_fill: float | None = None,
):
    df = generate_sample_data(rows=rows, seed=seed)
    df = _apply_filters(df, include_cols=include_cols, filter_query=filter_query, limit_rows=limit_rows)
    df = _apply_aggregation(df, group_by=group_by, agg=agg, value_cols=value_cols, pivot_col=pivot_col, pivot_fill=pivot_fill)
    info = basic_info(df)
    ms_df = missing_summary(df)
    ms_list = ms_df.to_dict(orient="records")
    num_cols, cat_cols, dt_cols = detect_column_types(df)
    method = (corr_method or "pearson").lower()
    if method not in ALLOWED_CORR:
        raise HTTPException(400, detail=f"Invalid corr_method. Use one of {sorted(ALLOWED_CORR)}")
    corr_payload = None
    if len(num_cols) >= 2:
        use_cols = num_cols[: max(2, min(max_corr_dims, len(num_cols)))]
        corr = correlation_matrix(df[use_cols], method=method)
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
        "columns_stats": _column_summaries(df, top_n=5),
        "missing": ms_list,
        "corr": corr_payload,
        "corr_method": method,
        "preview": _preview_records(df, max_rows=50),
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


def _build_figure(
    df: pd.DataFrame,
    chart: str,
    x: str | None,
    y: str | None,
    color: str | None,
    bins: int = 30,
    *,
    log_x: bool = False,
    log_y: bool = False,
    facet_row: str | None = None,
    facet_col: str | None = None,
    norm: str | None = None,
    barmode: str | None = None,
):
    chart = (chart or "").lower()
    if chart in ("histogram", "hist"):
        if not x:
            raise HTTPException(400, detail="x is required for histogram")
        fig = px.histogram(
            df,
            x=x,
            color=color,
            nbins=bins,
            facet_row=facet_row,
            facet_col=facet_col,
            histnorm=norm if norm in ("percent", "probability", "probability density", "density") else None,
            log_x=log_x,
            log_y=log_y,
        )
        if barmode:
            fig.update_layout(barmode=barmode)
    elif chart in ("bar", "bar_count", "count"):
        if not x:
            raise HTTPException(400, detail="x is required for bar_count")
        fig = px.histogram(
            df,
            x=x,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            histnorm=norm if norm in ("percent", "probability", "probability density", "density") else None,
            log_x=log_x,
            log_y=log_y,
        )
        if barmode:
            fig.update_layout(barmode=barmode)
    elif chart in ("bar_value", "bar_y"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for bar_value")
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
        )
    elif chart in ("scatter", "point"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for scatter")
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            log_x=log_x,
            log_y=log_y,
        )
    elif chart == "box":
        if not y:
            raise HTTPException(400, detail="y is required for box")
        fig = px.box(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            log_x=log_x,
            log_y=log_y,
        )
    elif chart in ("line", "timeseries", "time"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for line")
        fig = px.line(
            df.sort_values(by=x),
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            log_x=log_x,
            log_y=log_y,
        )
    elif chart in ("kde", "density", "density_contour"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for density")
        fig = px.density_contour(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
        )
    elif chart in ("hexbin", "hist2d", "density_heatmap"):
        if not (x and y):
            raise HTTPException(400, detail="x and y are required for 2D density")
        fig = px.density_heatmap(
            df,
            x=x,
            y=y,
            nbinsx=bins,
            nbinsy=bins,
            facet_row=facet_row,
            facet_col=facet_col,
            color_continuous_scale="Viridis",
        )
    elif chart in ("violin", "violinplot"):
        if not y:
            raise HTTPException(400, detail="y is required for violin")
        fig = px.violin(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            box=True,
            points=False,
        )
    elif chart in ("scatter_matrix", "splom"):
        # Dimensions: parse from x if comma-separated, else auto-pick top N numeric
        dims: list[str]
        if x and "," in x:
            dims = [c.strip() for c in x.split(",") if c.strip() in df.columns]
        else:
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            n = max(2, min(int(bins or 4), len(num_cols)))
            dims = num_cols[:n]
        if len(dims) < 2:
            raise HTTPException(400, detail="Need at least 2 numeric columns for scatter_matrix")
        fig = px.scatter_matrix(
            df,
            dimensions=dims,
            color=color if (color and color in df.columns) else None,
            opacity=0.7,
        )
        fig.update_traces(diagonal_visible=True)
    elif chart in ("feature_importance", "feat_imp", "importance"):
        # Compute simple feature importances using RandomForest.
        # y (target) is required.
        if not y or y not in df.columns:
            raise HTTPException(400, detail="y (target) is required for feature_importance")
        try:
            import numpy as np  # type: ignore
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # type: ignore
            from sklearn.preprocessing import LabelEncoder  # type: ignore
        except Exception as e:
            raise HTTPException(500, detail=f"scikit-learn import error: {e}")

        dfx = df.copy()
        # Select numeric features, exclude target
        feat_cols = dfx.select_dtypes(include=["number"]).columns.tolist()
        if y in feat_cols:
            feat_cols = [c for c in feat_cols if c != y]
        # If no numeric features, try encoding categoricals (simple label encoding)
        if not feat_cols:
            for col in dfx.columns:
                if col == y:
                    continue
                if dfx[col].dtype == object:
                    try:
                        dfx[col] = LabelEncoder().fit_transform(dfx[col].astype(str))
                    except Exception:
                        pass
            feat_cols = [c for c in dfx.columns if c != y]
        if len(feat_cols) < 1:
            raise HTTPException(400, detail="No usable feature columns for importance computation")

        dfx = dfx[feat_cols + [y]].dropna()
        if dfx.empty:
            raise HTTPException(400, detail="No rows left after dropping NA for importance computation")
        X = dfx[feat_cols].values
        yv = dfx[y].values

        # Choose model type
        is_classification = dfx[y].dtype == object or str(dfx[y].dtype).startswith("bool") or dfx[y].nunique() < max(20, int(len(dfx) * 0.05))
        if is_classification:
            # Encode target labels
            try:
                y_enc = LabelEncoder().fit_transform(yv.astype(str))
            except Exception:
                # Fallback: coerce
                y_enc = LabelEncoder().fit_transform(yv.astype("category"))
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X, y_enc)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X, yv)

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise HTTPException(500, detail="Model did not provide feature_importances_")

        pairs = sorted(zip(feat_cols, importances), key=lambda t: t[1], reverse=True)
        top_n = max(1, min(int(bins or 20), len(pairs)))
        pairs = pairs[:top_n]
        f_names = [p[0] for p in pairs][::-1]
        f_vals = [float(p[1]) for p in pairs][::-1]
        fig = px.bar(x=f_vals, y=f_names, orientation="h", labels={"x": "importance", "y": "feature"})
    else:
        raise HTTPException(400, detail="Unsupported chart. Use histogram|bar_count|scatter|box|line|density|hist2d|violin|scatter_matrix|feature_importance")
    return fig.to_dict()


@api.post("/eda/visualize")
async def eda_visualize(
    file: UploadFile = File(...),
    chart: str = Form(...),
    x: str | None = Form(None),
    y: str | None = Form(None),
    color: str | None = Form(None),
    bins: int = Form(30),
    log_x: bool = Form(False),
    log_y: bool = Form(False),
    facet_row: str | None = Form(None),
    facet_col: str | None = Form(None),
    norm: str | None = Form(None),
    barmode: str | None = Form(None),
    filter_query: str | None = Form(None),
    include_cols: str | None = Form(None),
    limit_rows: int | None = Form(None),
    group_by: str | None = Form(None),
    agg: str | None = Form(None),
    value_cols: str | None = Form(None),
    pivot_col: str | None = Form(None),
    pivot_fill: float | None = Form(None),
):
    try:
        data = await file.read()
        df, _, _ = _smart_read_csv(data)
        df = _apply_filters(df, include_cols=include_cols, filter_query=filter_query, limit_rows=limit_rows)
        df = _apply_aggregation(df, group_by=group_by, agg=agg, value_cols=value_cols, pivot_col=pivot_col, pivot_fill=pivot_fill)
        spec = _build_figure(
            df,
            chart=chart,
            x=x,
            y=y,
            color=color,
            bins=bins,
            log_x=log_x,
            log_y=log_y,
            facet_row=facet_row,
            facet_col=facet_col,
            norm=norm,
            barmode=barmode,
        )
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
    log_x: bool = False,
    log_y: bool = False,
    facet_row: str | None = None,
    facet_col: str | None = None,
    norm: str | None = None,
    barmode: str | None = None,
    filter_query: str | None = None,
    include_cols: str | None = None,
    limit_rows: int | None = None,
    group_by: str | None = None,
    agg: str | None = None,
    value_cols: str | None = None,
    pivot_col: str | None = None,
    pivot_fill: float | None = None,
):
    df = generate_sample_data(rows=rows, seed=seed)
    df = _apply_filters(df, include_cols=include_cols, filter_query=filter_query, limit_rows=limit_rows)
    df = _apply_aggregation(df, group_by=group_by, agg=agg, value_cols=value_cols, pivot_col=pivot_col, pivot_fill=pivot_fill)
    spec = _build_figure(
        df,
        chart=chart,
        x=x,
        y=y,
        color=color,
        bins=bins,
        log_x=log_x,
        log_y=log_y,
        facet_row=facet_row,
        facet_col=facet_col,
        norm=norm,
        barmode=barmode,
    )
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


# ----- Crawling endpoints ----------------------------------------------------

@api.get("/crawl/csv")
async def crawl_csv(
    url: str,
    sep: str | None = None,
    decimal: str | None = None,
    encoding: str | None = None,
    max_corr_dims: int = 8,
    corr_method: str = "pearson",
    filter_query: str | None = None,
    include_cols: str | None = None,
    limit_rows: int | None = None,
    group_by: str | None = None,
    agg: str | None = None,
    value_cols: str | None = None,
    pivot_col: str | None = None,
    pivot_fill: float | None = None,
):
    data = await _fetch_bytes(url)
    df, used_enc, used_sep = _smart_read_csv(data, sep=sep, decimal=decimal or ".", encoding=encoding)
    # Apply optional filters then aggregation
    df = _apply_filters(df, include_cols=include_cols, filter_query=filter_query, limit_rows=limit_rows)
    df = _apply_aggregation(df, group_by=group_by, agg=agg, value_cols=value_cols, pivot_col=pivot_col, pivot_fill=pivot_fill)
    info = basic_info(df)
    ms_df = missing_summary(df)
    ms_list = ms_df.to_dict(orient="records") if hasattr(ms_df, "to_dict") else []
    num_cols, cat_cols, dt_cols = detect_column_types(df)
    method = (corr_method or "pearson").lower()
    if method not in ALLOWED_CORR:
        raise HTTPException(400, detail=f"Invalid corr_method. Use one of {sorted(ALLOWED_CORR)}")
    corr_payload = None
    if len(num_cols) >= 2:
        use_cols = num_cols[: max(2, min(max_corr_dims, len(num_cols)))]
        corr = correlation_matrix(df[use_cols], method=method)
        corr_payload = {"labels": list(corr.columns), "matrix": corr.values.tolist()}
    return {
        "source_url": url,
        "rows": int(info.get("rows", len(df))),
        "columns": int(info.get("columns", df.shape[1] if not df.empty else 0)),
        "memory": str(info.get("memory", "")),
        "columns_info": {"numeric": num_cols, "categorical": cat_cols, "datetime": dt_cols},
        "columns_stats": _column_summaries(df, top_n=5),
        "missing": ms_list,
        "corr": corr_payload,
        "corr_method": method,
        "detected": {"encoding": used_enc, "sep": used_sep},
        "preview": _preview_records(df, max_rows=50),
    }


@api.get("/crawl/table")
async def crawl_table(url: str, index: int = 0, max_rows: int = 50):
    data = await _fetch_bytes(url)
    try:
        # pandas.read_html requires lxml
        tables = pd.read_html(io.BytesIO(data), flavor="lxml")
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to parse HTML tables: {e}")
    if not tables:
        raise HTTPException(404, detail="No tables found")
    if not (0 <= index < len(tables)):
        raise HTTPException(400, detail=f"Index out of range (found {len(tables)} tables)")
    df = tables[index]
    preview = df.head(max(1, max_rows)).to_dict(orient="records")
    return {"source_url": url, "table_index": index, "rows": len(df), "columns": df.shape[1], "preview": preview}


@api.get("/crawl/html")
async def crawl_html(url: str, selector: str | None = None, attr: str | None = None, max_items: int = 50):
    data = await _fetch_bytes(url)
    try:
        soup = BeautifulSoup(data, "lxml")
    except Exception as e:
        raise HTTPException(400, detail=f"HTML parse error: {e}")
    out: dict = {"source_url": url}
    if selector:
        nodes = soup.select(selector)[:max_items]
        if attr:
            items = [n.get(attr) for n in nodes]
        else:
            items = [n.get_text(strip=True) for n in nodes]
        out["selector"] = selector
        out["items"] = items
    else:
        out["title"] = (soup.title.string.strip() if soup.title and soup.title.string else None)
        out["links"] = [a.get("href") for a in soup.select("a[href]")[:max_items]]
        desc = soup.select_one('meta[name="description"]')
        out["description"] = desc.get("content") if desc else None
    return out


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


