import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .datasets import _csv_path as ds_csv_path, _load_meta as load_ds_meta, _save_meta as save_ds_meta, _now_iso

# Optional heavy deps (scikit-learn)
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for ML endpoints. Add 'scikit-learn' to requirements and install.")


router = APIRouter(prefix="/ml", tags=["ml"])


DATA_ROOT = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
MODELS_META_PATH = os.path.join(DATA_ROOT, "models.json")


def _ensure_model_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)


def _load_models_meta() -> Dict[str, Any]:
    if not os.path.exists(MODELS_META_PATH):
        return {"models": {}}
    try:
        with open(MODELS_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"models": {}}


def _save_models_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(DATA_ROOT, exist_ok=True)
    tmp = MODELS_META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MODELS_META_PATH)


def _kind_from_model(model: str) -> str:
    m = model.lower().strip()
    if m in {"linreg", "rf_reg", "linear_regression", "random_forest_regressor"}:
        return "regression"
    if m in {"logreg", "rf_clf", "logistic_regression", "random_forest_classifier"}:
        return "classification"
    return "unknown"


def _build_estimator(model: str):
    m = model.lower().strip()
    if m in {"linreg", "linear_regression"}:
        return LinearRegression()
    if m in {"logreg", "logistic_regression"}:
        return LogisticRegression(max_iter=1000)
    if m in {"rf_reg", "random_forest_regressor"}:
        return RandomForestRegressor(n_estimators=200, random_state=42)
    if m in {"rf_clf", "random_forest_classifier"}:
        return RandomForestClassifier(n_estimators=200, random_state=42)
    raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")


def _build_preprocess(df: pd.DataFrame, features: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    feats = [c for c in features if c in df.columns]
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre, num_cols, cat_cols


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    model: str = Field("linreg", description="linreg|logreg|rf_reg|rf_clf")
    features: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: Optional[int] = 42
    model_id: Optional[str] = None
    sample_rows: Optional[int] = Field(None, description="Train on first N rows (optional)")
    sample_frac: Optional[float] = Field(None, description="Random fraction 0-1 (optional)")


@router.post("/train")
def train_model(req: TrainRequest) -> Dict[str, Any]:
    path = ds_csv_path(req.dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        # Read whole dataset first for flexibility (sampling applied after load)
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if req.target not in df.columns:
        raise HTTPException(status_code=404, detail=f"Target not found: {req.target}")

    features = req.features or [c for c in df.columns if c != req.target]
    features = [c for c in features if c in df.columns]
    if not features:
        raise HTTPException(status_code=400, detail="No valid features")

    # Optional sampling
    df2 = df
    if req.sample_rows is not None and int(req.sample_rows) > 0:
        df2 = df2.head(int(req.sample_rows))
    elif req.sample_frac is not None:
        try:
            frac = max(0.0, min(1.0, float(req.sample_frac)))
            if frac > 0 and frac < 1:
                df2 = df2.sample(frac=frac, random_state=req.random_state)
        except Exception:
            pass
    # Drop NA target
    df2 = df2.dropna(subset=[req.target])
    X = df2[features]
    y = df2[req.target]

    pre, num_cols, cat_cols = _build_preprocess(df2, features)
    est = _build_estimator(req.model)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", est)])

    test_size = max(0.05, min(0.9, float(req.test_size)))
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=req.random_state)
        pipe.fit(X_train, y_train)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")

    kind = _kind_from_model(req.model)
    metrics: Dict[str, Any] = {"kind": kind}
    try:
        y_pred = pipe.predict(X_test)
        if kind == "regression":
            metrics.update({
                "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            })
        else:
            metrics.update({
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            })
            # Optional AUC for binary classification when possible
            try:
                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            except Exception:
                pass
    except Exception:
        pass

    # Persist model
    _ensure_model_dirs()
    model_id = (req.model_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail=f"Model already exists: {model_id}")
    try:
        joblib.dump({
            "pipeline": pipe,
            "features": features,
            "target": req.target,
            "kind": kind,
            "model": req.model,
        }, model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {e}")

    # Save metadata
    meta = _load_models_meta()
    now = _now_iso()
    meta_entry = {
        "id": model_id,
        "model": req.model,
        "kind": kind,
        "dataset_id": req.dataset_id,
        "features": features,
        "target": req.target,
        "metrics": metrics,
        "path": os.path.relpath(model_path, os.getcwd()),
        "created_at": now,
        "updated_at": now,
        "test_size": test_size,
        "random_state": req.random_state,
    }
    meta.setdefault("models", {})[model_id] = meta_entry
    _save_models_meta(meta)

    return {"status": "ok", "model_id": model_id, "metrics": metrics, "model": meta_entry}


class PredictDatasetRequest(BaseModel):
    dataset_id: str
    out_id: Optional[str] = None
    proba: Optional[bool] = False


@router.post("/{model_id}/predict_dataset")
def predict_dataset(model_id: str, req: PredictDatasetRequest) -> Dict[str, Any]:
    meta = _load_models_meta()
    m = meta.get("models", {}).get(model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_path = os.path.join(os.getcwd(), m["path"]) if not os.path.isabs(m["path"]) else m["path"]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    try:
        payload = joblib.load(model_path)
        pipe = payload["pipeline"]
        features: List[str] = payload["features"]
        kind: str = payload.get("kind", "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    path = ds_csv_path(req.dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    X = df[[c for c in features if c in df.columns]].copy()
    try:
        if kind == "classification" and req.proba and hasattr(pipe.named_steps["model"], "predict_proba"):
            pred = pipe.predict_proba(X)
            # Use positive class probability when binary, otherwise max probability
            if pred.shape[1] == 2:
                df["prediction_proba"] = pred[:, 1]
            df["prediction"] = pipe.predict(X)
        else:
            df["prediction"] = pipe.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    out_id = (req.out_id or str(uuid.uuid4())).strip() or str(uuid.uuid4())
    out_path = ds_csv_path(out_id)
    if os.path.exists(out_path):
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_id}")
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save predicted dataset: {e}")

    # Update dataset metadata
    ds_meta = load_ds_meta()
    created_at = _now_iso()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    ds_meta.setdefault("datasets", {})[out_id] = {
        "id": out_id,
        "filename": os.path.basename(out_path),
        "original_name": f"predict:{req.dataset_id}:model:{model_id}",
        "path": os.path.relpath(out_path, os.getcwd()),
        "size_bytes": os.path.getsize(out_path),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "sampled_rows": min(50, len(df)),
        "created_at": created_at,
        "updated_at": created_at,
    }
    save_ds_meta(ds_meta)

    return {"status": "ok", "model_id": model_id, "source_id": req.dataset_id, "dataset_id": out_id, "rows": int(len(df))}


@router.get("/models")
def list_models() -> Dict[str, Any]:
    meta = _load_models_meta()
    items = list(meta.get("models", {}).values())
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"items": items, "count": len(items)}


@router.delete("/models/{model_id}")
def delete_model(model_id: str) -> Dict[str, Any]:
    meta = _load_models_meta()
    existed = False
    m = meta.get("models", {}).get(model_id)
    if m:
        model_path = os.path.join(os.getcwd(), m["path"]) if not os.path.isabs(m["path"]) else m["path"]
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete model file: {e}")
        del meta["models"][model_id]
        _save_models_meta(meta)
        existed = True
    if not existed:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": model_id}
