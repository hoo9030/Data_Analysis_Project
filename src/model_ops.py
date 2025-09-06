from __future__ import annotations

import io
import pickle
from typing import Literal, Tuple, Dict, Any

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

