import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import shutil


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
