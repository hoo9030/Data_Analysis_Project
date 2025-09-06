from __future__ import annotations

import io
import pickle
from typing import Literal, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


ProblemType = Literal["regression", "classification"]


def infer_problem_type(y: pd.Series) -> ProblemType:
    if pd.api.types.is_numeric_dtype(y):
        unique = y.dropna().unique()
        uniq_cnt = len(unique)
        if uniq_cnt <= 15 and uniq_cnt / max(len(y), 1) < 0.1:
            return "classification"
        return "regression"
    else:
        return "classification"


def split_features(df: pd.DataFrame, target: str) -> Tuple[list[str], list[str]]:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # bool은 범주형으로 취급
    cat_cols = X.columns.difference(num_cols).tolist()
    return num_cols, cat_cols


def build_pipeline(problem: ProblemType, num_cols: list[str], cat_cols: list[str], model_name: str) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    if problem == "regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            raise ValueError("Unknown regression model")
    else:
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            raise ValueError("Unknown classification model")

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"Accuracy": acc, "Precision(w)": precision, "Recall(w)": recall, "F1(w)": f1}


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    problem: ProblemType,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, Any]]:
    X = df.drop(columns=[target])
    y = df[target]

    num_cols, cat_cols = split_features(df, target)
    pipe = build_pipeline(problem, num_cols, cat_cols, model_name)

    stratify = y if problem == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    figs: Dict[str, Any] = {}
    if problem == "regression":
        metrics = evaluate_regression(y_test.values, y_pred)
        try:
            import plotly.express as px

            dfp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
            fig_scatter = px.scatter(dfp, x="y_true", y="y_pred", opacity=0.7)
            min_v = float(np.nanmin([dfp["y_true"].min(), dfp["y_pred"].min()]))
            max_v = float(np.nanmax([dfp["y_true"].max(), dfp["y_pred"].max()]))
            fig_scatter.add_shape(
                type="line",
                x0=min_v,
                y0=min_v,
                x1=max_v,
                y1=max_v,
                line=dict(color="#888", dash="dash"),
            )
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            figs["pred_vs_actual"] = fig_scatter

            res = dfp["y_pred"] - dfp["y_true"]
            fig_res = px.histogram(res, nbins=40, opacity=0.85)
            fig_res.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            figs["residuals_hist"] = fig_res
        except Exception:
            pass
    else:
        metrics = evaluate_classification(y_test.values, y_pred)
        try:
            import plotly.express as px

            labels = np.unique(np.concatenate([np.array(y_test), np.array(y_pred)]))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            figs["confusion_matrix"] = fig_cm
        except Exception:
            pass

    return pipe, metrics, figs


def serialize_model_to_bytes(model: Pipeline) -> bytes:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    buf.seek(0)
    return buf.read()


def get_param_grid(problem: ProblemType, model_name: str) -> Dict[str, list]:
    if problem == "regression":
        if model_name == "Linear Regression":
            return {
                "model__fit_intercept": [True, False],
            }
        elif model_name == "Random Forest":
            return {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["auto", "sqrt", 0.5],
            }
    else:
        if model_name == "Logistic Regression":
            return {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            }
        elif model_name == "Random Forest":
            return {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["auto", "sqrt", "log2"],
                "model__class_weight": [None, "balanced"],
            }
    raise ValueError("Unsupported model/problem for param grid")


def tune_and_evaluate(
    df: pd.DataFrame,
    target: str,
    problem: ProblemType,
    model_name: str,
    search: Literal["grid", "random"] = "grid",
    scoring: str | None = None,
    cv: int = 5,
    n_iter: int = 25,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float], Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Run CV-based hyperparameter tuning, then evaluate on a hold-out split.

    Returns: (best_model, metrics, figs, cv_results_df, best_params)
    """
    X = df.drop(columns=[target])
    y = df[target]

    num_cols, cat_cols = split_features(df, target)
    base = build_pipeline(problem, num_cols, cat_cols, model_name)
    param_grid = get_param_grid(problem, model_name)

    # CV Search
    common_kwargs = dict(scoring=scoring, cv=cv, n_jobs=-1, refit=True, return_train_score=True)
    if search == "grid":
        searcher = GridSearchCV(base, param_grid=param_grid, **{k: v for k, v in common_kwargs.items() if k != "n_jobs"} , n_jobs=-1)
    else:
        searcher = RandomizedSearchCV(base, param_distributions=param_grid, n_iter=n_iter, **common_kwargs)

    # Hold-out split for final evaluation
    stratify = y if problem == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    searcher.fit(X_train, y_train)
    best_model: Pipeline = searcher.best_estimator_
    y_pred = best_model.predict(X_test)

    figs: Dict[str, Any] = {}
    if problem == "regression":
        metrics = evaluate_regression(y_test.values, y_pred)
        try:
            import plotly.express as px
            dfp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
            fig_scatter = px.scatter(dfp, x="y_true", y="y_pred", opacity=0.7)
            min_v = float(np.nanmin([dfp["y_true"].min(), dfp["y_pred"].min()]))
            max_v = float(np.nanmax([dfp["y_true"].max(), dfp["y_pred"].max()]))
            fig_scatter.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#888", dash="dash"))
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            figs["pred_vs_actual"] = fig_scatter

            res = dfp["y_pred"] - dfp["y_true"]
            fig_res = px.histogram(res, nbins=40, opacity=0.85)
            fig_res.update_layout(margin=dict(l=10, r=10, t=30, b=10))
            figs["residuals_hist"] = fig_res
        except Exception:
            pass
    else:
        metrics = evaluate_classification(y_test.values, y_pred)
        try:
            import plotly.express as px
            labels = np.unique(np.concatenate([np.array(y_test), np.array(y_pred)]))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", margin=dict(l=10, r=10, t=30, b=10))
            figs["confusion_matrix"] = fig_cm
        except Exception:
            pass

    # CV results as DataFrame
    results = pd.DataFrame(searcher.cv_results_)
    cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "params",
    ]
    results = results[[c for c in cols if c in results.columns]].sort_values("rank_test_score")

    return best_model, metrics, figs, results, searcher.best_params_


def _get_output_feature_names(pipe: Pipeline) -> List[str]:
    """Return output feature names after preprocess step.

    Names include prefixes like 'num__' and 'cat__'.
    """
    if "preprocess" not in pipe.named_steps:
        return []
    pre = pipe.named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out()
        names = [str(x) for x in names]
    except Exception:
        # Fallback: approximate with original columns
        try:
            # Attempt to reconstruct from transformers list
            names = []
            for name, trans, cols in pre.transformers_:
                if hasattr(trans, "get_feature_names_out"):
                    sub_names = trans.get_feature_names_out(cols)
                    names.extend([f"{name}__{sn}" for sn in sub_names])
                else:
                    names.extend([f"{name}__{c}" for c in cols])
        except Exception:
            names = []
    return names


def extract_feature_importances(pipe: Pipeline) -> pd.DataFrame:
    """Extract feature importances/coefficients from a fitted pipeline.

    Returns DataFrame with columns: feature, importance, signed (optional for linear models).
    """
    model = pipe.named_steps.get("model")
    if model is None:
        return pd.DataFrame()

    feat_names = _get_output_feature_names(pipe)
    # Clean prefixes for readability
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names]

    imp = None
    signed = None

    if hasattr(model, "feature_importances_"):
        arr = np.asarray(model.feature_importances_).ravel()
        imp = arr
    elif hasattr(model, "coef_"):
        co = np.asarray(model.coef_)
        if co.ndim == 1:
            signed = co
            imp = np.abs(co)
        else:
            # Multi-class: aggregate across classes
            signed = np.mean(co, axis=0)
            imp = np.mean(np.abs(co), axis=0)
    else:
        return pd.DataFrame()

    # Align lengths if transformers dropped columns
    if feat_names and len(imp) != len(clean_names):
        # Trim or pad (pad with zeros) to match lengths safely
        n = min(len(imp), len(clean_names))
        imp = imp[:n]
        signed = signed[:n] if signed is not None else None
        clean_names = clean_names[:n]

    df_imp = pd.DataFrame({
        "feature": clean_names if clean_names else [f"f{i}" for i in range(len(imp))],
        "importance": imp,
    })
    if signed is not None:
        df_imp["signed"] = signed

    # Sort by absolute importance desc
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def get_expected_columns(pipe: Pipeline) -> Tuple[List[str], List[str]]:
    """Return (num_cols, cat_cols) that the preprocessor expects."""
    pre = pipe.named_steps.get("preprocess")
    num_cols: List[str] = []
    cat_cols: List[str] = []
    try:
        for name, trans, cols in pre.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
    except Exception:
        pass
    return num_cols, cat_cols


def align_columns_for_inference(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Ensure X has all columns expected by the pipeline's preprocessor.

    Adds missing columns with NaN so imputers can handle them, and reorders/filters
    columns to expected set.
    """
    num_cols, cat_cols = get_expected_columns(pipe)
    expected: List[str] = list(dict.fromkeys((num_cols or []) + (cat_cols or [])))
    if not expected:
        return X
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = np.nan
    # Keep only expected columns in stable order
    return X2[expected]
