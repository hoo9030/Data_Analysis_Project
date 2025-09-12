import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import shutil
from pydantic import BaseModel, Field
import io


router = APIRouter(prefix="/datasets", tags=["datasets"])


# Simple local storage under ./data
DATA_ROOT = os.path.join(os.getcwd(), "data")
DATASET_DIR = os.path.join(DATA_ROOT, "datasets")
META_PATH = os.path.join(DATA_ROOT, "metadata.json")


def _ensure_dirs() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)


def _load_meta() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {"datasets": {}}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt or unreadable metadata -> reset
        return {"datasets": {}}


def _save_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(DATA_ROOT, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _csv_path(dataset_id: str) -> str:
    fname = f"{dataset_id}.csv"
    return os.path.join(DATASET_DIR, fname)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("")
async def upload_dataset(file: UploadFile = File(...), dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a CSV file. Stores to ./data/datasets/<id>.csv and records metadata.
    Returns dataset_id and basic metadata.
    """
    _ensure_dirs()

    # Basic validation
    original_name = file.filename or "uploaded.csv"
    if not (original_name.lower().endswith(".csv") or (file.content_type or "").endswith("csv")):
        raise HTTPException(status_code=400, detail="Only CSV upload is supported.")

    ds_id = (dataset_id or str(uuid.uuid4())).strip()
    if not ds_id:
        ds_id = str(uuid.uuid4())

    target = _csv_path(ds_id)
    if os.path.exists(target):
        raise HTTPException(status_code=409, detail=f"Dataset id already exists: {ds_id}")

    # Save file to disk
    try:
        with open(target, "wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                out.write(chunk)
    finally:
        await file.close()

    # Attempt to read some rows for quick metadata
    row_count: Optional[int] = None
    columns: List[str] = []
    dtypes: Dict[str, str] = {}
    preview_rows: int = 0
    try:
        # Read a small sample to infer schema
        sample = pd.read_csv(target, nrows=50)
        columns = list(sample.columns)
        dtypes = {c: str(t) for c, t in sample.dtypes.items()}
        preview_rows = len(sample)
    except Exception:
        # Keep metadata minimal if parsing fails
        pass

    meta = _load_meta()
    created_at = _now_iso()
    meta["datasets"][ds_id] = {
        "id": ds_id,
        "filename": os.path.basename(target),
        "original_name": original_name,
        "path": os.path.relpath(target, os.getcwd()),
        "size_bytes": os.path.getsize(target),
        "columns": columns,
        "dtypes": dtypes,
        "sampled_rows": preview_rows,
        "created_at": created_at,
        "updated_at": created_at,
    }
    _save_meta(meta)

    return {"dataset_id": ds_id, "metadata": meta["datasets"][ds_id]}


@router.get("")
def list_datasets() -> Dict[str, Any]:
    meta = _load_meta()
    items = list(meta.get("datasets", {}).values())
    # Sort by updated_at desc
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"items": items, "count": len(items)}


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, nrows: int = 20) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(path, nrows=max(1, min(nrows, 1000)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Convert to JSON-friendly structures
    data = df.to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "rows": data,
        "row_count": len(df),
    }


@router.get("/{dataset_id}/describe")
def describe_dataset(dataset_id: str, limit: int = 5000, include_all: bool = True) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(path, nrows=max(10, min(limit, 200000)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        desc = df.describe(include="all" if include_all else None).fillna(value=None)
    except Exception:
        # As a fallback, compute simple stats manually for numeric cols
        numeric_cols = df.select_dtypes("number")
        desc = numeric_cols.describe().fillna(value=None)

    # dtypes
    dtypes = {c: str(t) for c, t in df.dtypes.items()}

    # Convert describe dataframe to mapping: index -> {col -> value}
    stats = {}
    for idx, row in desc.iterrows():
        stats[str(idx)] = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "stats": stats,
        "sample_rows": len(df),
    }


@router.get("/{dataset_id}/nulls")
def nulls_overview(dataset_id: str, limit: Optional[int] = None, chunksize: int = 50000) -> Dict[str, Any]:
    """
    Compute per-column null counts and percentages.
    - If limit is provided, reads up to `limit` rows using chunks.
    - Otherwise processes the entire file in chunks.
    """
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    total_rows = 0
    null_counts: Dict[str, int] = {}
    try:
        rows_left = limit if limit is not None else None
        for chunk in pd.read_csv(path, chunksize=max(1000, chunksize)):
            if rows_left is not None:
                # Trim chunk to rows_left
                if rows_left <= 0:
                    break
                if len(chunk) > rows_left:
                    chunk = chunk.iloc[:rows_left]
                rows_left -= len(chunk)

            total_rows += len(chunk)
            isna = chunk.isna().sum()
            for col, cnt in isna.items():
                null_counts[col] = null_counts.get(col, 0) + int(cnt)
        if total_rows == 0:
            # Empty file or zero rows processed
            df = pd.read_csv(path, nrows=0)
            columns = list(df.columns)
            items = [
                {"column": c, "total_rows": 0, "nulls": 0, "null_pct": 0.0}
                for c in columns
            ]
            return {"dataset_id": dataset_id, "items": items, "total_rows": 0}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Build response
    columns = list(null_counts.keys())
    items = []
    for c in columns:
        n = null_counts.get(c, 0)
        pct = (n / total_rows * 100.0) if total_rows else 0.0
        items.append({"column": c, "total_rows": total_rows, "nulls": n, "null_pct": round(pct, 3)})

    # Keep items sorted by null percentage desc
    items.sort(key=lambda x: x["null_pct"], reverse=True)
    return {"dataset_id": dataset_id, "items": items, "total_rows": total_rows}


class CastRequest(BaseModel):
    column: str = Field(..., description="Column to cast")
    to: str = Field(..., description="Target type: int|float|string|datetime|bool|category")
    out_id: Optional[str] = Field(None, description="New dataset id; generated if omitted")
    mode: str = Field("coerce", description="coerce|strict for parsing errors")


@router.post("/{dataset_id}/cast")
def cast_column(dataset_id: str, req: CastRequest) -> Dict[str, Any]:
    """
    Cast a column to a target dtype and save as a new dataset.
    - mode=coerce: invalid values become NaN/NaT
    - mode=strict: fails on invalid conversion
    """
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column not found: {col}")

    target = req.to.lower().strip()
    mode = req.mode.lower().strip() if req.mode else "coerce"
    if target not in {"int", "float", "string", "datetime", "bool", "category"}:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {target}")
    if mode not in {"coerce", "strict"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    errors = "coerce" if mode == "coerce" else "raise"
    before_nulls = int(df[col].isna().sum())

    def _cast_series(s: pd.Series) -> pd.Series:
        if target == "int":
            s2 = pd.to_numeric(s, errors=errors)
            # Prefer nullable integer dtype
            try:
                return s2.astype("Int64")
            except Exception:
                return s2.astype("int64")
        if target == "float":
            s2 = pd.to_numeric(s, errors=errors)
            return s2.astype("float64")
        if target == "string":
            # Pandas nullable string dtype
            try:
                return s.astype("string")
            except Exception:
                return s.astype(str)
        if target == "datetime":
            s2 = pd.to_datetime(s, errors=errors, utc=False)
            return s2
        if target == "bool":
            base = s
            try:
                tmp = base.astype(str).str.strip().str.lower()
                mapping = {
                    "true": True, "1": True, "yes": True, "y": True, "t": True,
                    "false": False, "0": False, "no": False, "n": False, "f": False,
                }
                s2 = tmp.map(mapping)
                if mode == "strict" and s2.isna().any():
                    raise ValueError("Invalid boolean tokens present")
                return s2.astype("boolean")
            except Exception as e:
                if mode == "strict":
                    raise
                # coerce: non-mapped remain NaN
                return s2.astype("boolean")
        if target == "category":
            return s.astype("category")
        return s

    try:
        df[col] = _cast_series(df[col])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cast failed: {e}")

    after_nulls = int(df[col].isna().sum())
    coerced_new_nulls = max(0, after_nulls - before_nulls)

    out_id = (req.out_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    out_path = _csv_path(out_id)
    if os.path.exists(out_path):
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_id}")

    try:
        _ensure_dirs()
        df.to_csv(out_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save new dataset: {e}")

    # Update metadata
    meta = _load_meta()
    created_at = _now_iso()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    meta["datasets"][out_id] = {
        "id": out_id,
        "filename": os.path.basename(out_path),
        "original_name": f"derive:{dataset_id}",
        "path": os.path.relpath(out_path, os.getcwd()),
        "size_bytes": os.path.getsize(out_path),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "sampled_rows": min(50, len(df)),
        "created_at": created_at,
        "updated_at": created_at,
    }
    _save_meta(meta)

    return {
        "status": "ok",
        "source_id": dataset_id,
        "dataset_id": out_id,
        "column": col,
        "target_type": target,
        "before_nulls": before_nulls,
        "after_nulls": after_nulls,
        "coerced_new_nulls": coerced_new_nulls,
        "metadata": meta["datasets"][out_id],
    }


class FillNARequest(BaseModel):
    columns: Optional[List[str]] = Field(None, description="Target columns; default all")
    strategy: str = Field("value", description="value|mean|median|mode")
    value: Optional[Any] = Field(None, description="Fill value when strategy=value")
    out_id: Optional[str] = None


@router.post("/{dataset_id}/fillna")
def fillna_dataset(dataset_id: str, req: FillNARequest) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    cols = req.columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise HTTPException(status_code=400, detail="No valid columns to fill")

    strategy = req.strategy.lower().strip() if req.strategy else "value"
    before_nulls = {c: int(df[c].isna().sum()) for c in cols}

    try:
        if strategy == "value":
            df[cols] = df[cols].fillna(req.value)
        elif strategy in ("mean", "median"):
            num = df[cols].select_dtypes(include="number").columns.tolist()
            if not num:
                raise HTTPException(status_code=400, detail="No numeric columns for mean/median")
            if strategy == "mean":
                fillmap = {c: df[c].mean() for c in num}
            else:
                fillmap = {c: df[c].median() for c in num}
            for c, v in fillmap.items():
                df[c] = df[c].fillna(v)
        elif strategy == "mode":
            for c in cols:
                try:
                    m = df[c].mode(dropna=True)
                    if not m.empty:
                        df[c] = df[c].fillna(m.iloc[0])
                except Exception:
                    # skip column if mode fails
                    pass
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported strategy: {strategy}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fillna failed: {e}")

    after_nulls = {c: int(df[c].isna().sum()) for c in cols}
    filled_total = int(sum(before_nulls[c] - after_nulls[c] for c in cols))

    out_id = (req.out_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    out_path = _csv_path(out_id)
    if os.path.exists(out_path):
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_id}")

    try:
        _ensure_dirs()
        df.to_csv(out_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save new dataset: {e}")

    meta = _load_meta()
    created_at = _now_iso()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    meta["datasets"][out_id] = {
        "id": out_id,
        "filename": os.path.basename(out_path),
        "original_name": f"derive:{dataset_id}",
        "path": os.path.relpath(out_path, os.getcwd()),
        "size_bytes": os.path.getsize(out_path),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "sampled_rows": min(50, len(df)),
        "created_at": created_at,
        "updated_at": created_at,
    }
    _save_meta(meta)

    return {
        "status": "ok",
        "source_id": dataset_id,
        "dataset_id": out_id,
        "strategy": strategy,
        "filled_total": filled_total,
        "before_nulls": before_nulls,
        "after_nulls": after_nulls,
        "metadata": meta["datasets"][out_id],
    }


class DropColsRequest(BaseModel):
    columns: List[str]
    out_id: Optional[str] = None


@router.post("/{dataset_id}/drop")
def drop_columns(dataset_id: str, req: DropColsRequest) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    cols = [c for c in (req.columns or []) if c in df.columns]
    if not cols:
        raise HTTPException(status_code=400, detail="No valid columns to drop")

    df2 = df.drop(columns=cols)

    out_id = (req.out_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    out_path = _csv_path(out_id)
    if os.path.exists(out_path):
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_id}")
    try:
        _ensure_dirs()
        df2.to_csv(out_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save new dataset: {e}")

    meta = _load_meta()
    created_at = _now_iso()
    dtypes = {c: str(t) for c, t in df2.dtypes.items()}
    meta["datasets"][out_id] = {
        "id": out_id,
        "filename": os.path.basename(out_path),
        "original_name": f"derive:{dataset_id}",
        "path": os.path.relpath(out_path, os.getcwd()),
        "size_bytes": os.path.getsize(out_path),
        "columns": list(df2.columns),
        "dtypes": dtypes,
        "sampled_rows": min(50, len(df2)),
        "created_at": created_at,
        "updated_at": created_at,
    }
    _save_meta(meta)

    return {
        "status": "ok",
        "source_id": dataset_id,
        "dataset_id": out_id,
        "dropped": cols,
        "metadata": meta["datasets"][out_id],
    }


class RenameRequest(BaseModel):
    mapping: Dict[str, str]
    out_id: Optional[str] = None


@router.post("/{dataset_id}/rename")
def rename_columns(dataset_id: str, req: RenameRequest) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    mapping = {k: v for k, v in (req.mapping or {}).items() if k in df.columns and v}
    if not mapping:
        raise HTTPException(status_code=400, detail="No valid rename mapping")
    df2 = df.rename(columns=mapping)
    # Validate no duplicate columns
    if len(set(df2.columns)) != len(df2.columns):
        raise HTTPException(status_code=400, detail="Duplicate column names after rename")

    out_id = (req.out_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    out_path = _csv_path(out_id)
    if os.path.exists(out_path):
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_id}")
    try:
        _ensure_dirs()
        df2.to_csv(out_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save new dataset: {e}")

    meta = _load_meta()
    created_at = _now_iso()
    dtypes = {c: str(t) for c, t in df2.dtypes.items()}
    meta["datasets"][out_id] = {
        "id": out_id,
        "filename": os.path.basename(out_path),
        "original_name": f"derive:{dataset_id}",
        "path": os.path.relpath(out_path, os.getcwd()),
        "size_bytes": os.path.getsize(out_path),
        "columns": list(df2.columns),
        "dtypes": dtypes,
        "sampled_rows": min(50, len(df2)),
        "created_at": created_at,
        "updated_at": created_at,
    }
    _save_meta(meta)

    return {
        "status": "ok",
        "source_id": dataset_id,
        "dataset_id": out_id,
        "renamed": mapping,
        "metadata": meta["datasets"][out_id],
    }


@router.get("/{dataset_id}/distribution")
def distribution(
    dataset_id: str,
    column: str,
    bins: int = 20,
    topk: int = 20,
    limit: Optional[int] = 50000,
    dropna: bool = True,
) -> Dict[str, Any]:
    """
    Compute a simple distribution for a column.
    - Numeric: histogram with `bins` buckets
    - Non-numeric: top-K value counts
    """
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        nrows = None if (limit is None or limit <= 0) else int(limit)
        df = pd.read_csv(path, nrows=nrows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column not found: {column}")

    s = df[column]
    na_count = int(s.isna().sum())
    total = int(len(s))

    # Decide if numeric
    is_numeric = pd.api.types.is_numeric_dtype(s)
    if not is_numeric:
        try:
            s_num = pd.to_numeric(s, errors="coerce")
            numeric_ratio = float(s_num.notna().mean()) if len(s_num) else 0.0
            is_numeric = numeric_ratio >= 0.8
        except Exception:
            is_numeric = False

    if is_numeric:
        # Numeric histogram
        s_num = pd.to_numeric(s, errors="coerce")
        if dropna:
            s_num = s_num.dropna()
        if len(s_num) == 0:
            return {
                "dataset_id": dataset_id,
                "column": column,
                "type": "numeric",
                "total": total,
                "na_count": na_count,
                "items": [],
                "bins": bins,
            }
        try:
            cut = pd.cut(s_num, bins=max(1, int(bins)), include_lowest=True)
            counts = cut.value_counts().sort_index()
            items = []
            for interval, cnt in counts.items():
                try:
                    left = float(interval.left)
                    right = float(interval.right)
                except Exception:
                    left = None
                    right = None
                items.append({
                    "left": left,
                    "right": right,
                    "count": int(cnt),
                    "label": str(interval),
                })
            return {
                "dataset_id": dataset_id,
                "column": column,
                "type": "numeric",
                "total": total,
                "na_count": na_count,
                "bins": bins,
                "min": float(s_num.min()) if len(s_num) else None,
                "max": float(s_num.max()) if len(s_num) else None,
                "items": items,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to compute histogram: {e}")
    else:
        # Categorical top-K
        try:
            counts = s.value_counts(dropna=True)
            items = []
            for val, cnt in counts.head(max(1, int(topk))).items():
                items.append({
                    "value": None if pd.isna(val) else str(val),
                    "count": int(cnt),
                })
            return {
                "dataset_id": dataset_id,
                "column": column,
                "type": "categorical",
                "total": total,
                "na_count": na_count,
                "topk": topk,
                "unique": int(counts.size),
                "items": items,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to compute value counts: {e}")


@router.get("/{dataset_id}/corr")
def correlation(dataset_id: str, method: str = "pearson", limit: Optional[int] = 50000) -> Dict[str, Any]:
    """
    Compute correlation matrix across numeric columns using pandas DataFrame.corr.
    method: pearson|spearman|kendall
    """
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    if method not in {"pearson", "spearman", "kendall"}:
        raise HTTPException(status_code=400, detail=f"Unsupported method: {method}")

    try:
        nrows = None if (limit is None or limit <= 0) else int(limit)
        df = pd.read_csv(path, nrows=nrows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {"dataset_id": dataset_id, "columns": list(num.columns), "matrix": {}, "rows": int(num.shape[0])}

    try:
        corr = num.corr(method=method)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to compute correlation: {e}")

    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for r in corr.index:
        row = {}
        for c in corr.columns:
            v = corr.loc[r, c]
            row[c] = None if pd.isna(v) else float(v)
        matrix[r] = row

    return {
        "dataset_id": dataset_id,
        "method": method,
        "columns": list(corr.columns),
        "matrix": matrix,
        "rows": int(num.shape[0]),
    }


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str):
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    filename = os.path.basename(path)
    return FileResponse(path, media_type="text/csv", filename=filename)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    path = _csv_path(dataset_id)
    meta = _load_meta()
    existed = False

    # Remove file if exists
    if os.path.exists(path):
        try:
            os.remove(path)
            existed = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    # Remove metadata
    if meta.get("datasets") and dataset_id in meta["datasets"]:
        del meta["datasets"][dataset_id]
        _save_meta(meta)
        existed = True or existed

    if not existed:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"status": "deleted", "dataset_id": dataset_id}


def _read_filtered_df(path: str, query: Optional[str], columns: Optional[List[str]], limit: int, chunksize: int = 50000) -> pd.DataFrame:
    acc: List[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, chunksize=max(1000, chunksize)):
        dfc = chunk
        if query:
            try:
                dfc = dfc.query(query, engine="python")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid query: {e}")
        if columns:
            cols = [c for c in columns if c in dfc.columns]
            if cols:
                dfc = dfc[cols]
        if limit > 0 and total + len(dfc) > limit:
            dfc = dfc.iloc[: max(0, limit - total)]
        acc.append(dfc)
        total += len(dfc)
        if limit > 0 and total >= limit:
            break
    if not acc:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(acc, ignore_index=True)


@router.get("/{dataset_id}/sample.csv")
def download_sample_csv(dataset_id: str, rows: int = 100) -> StreamingResponse:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        n = max(1, min(int(rows), 100000))
        df = pd.read_csv(path, nrows=n)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"{dataset_id}_sample_{n}.csv"
    return StreamingResponse(buf, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/{dataset_id}/filter.csv")
def download_filter_csv(
    dataset_id: str,
    query: Optional[str] = None,
    columns: Optional[str] = None,
    limit: int = 10000,
) -> StreamingResponse:
    path = _csv_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    cols_list: Optional[List[str]] = None
    if columns:
        cols_list = [c.strip() for c in columns.split(",") if c.strip()]
    try:
        lim = max(0, min(int(limit), 500000))
        df = _read_filtered_df(path, query=query, columns=cols_list, limit=lim)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to filter CSV: {e}")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"{dataset_id}_filter.csv"
    return StreamingResponse(buf, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
