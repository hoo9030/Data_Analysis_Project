import io
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def generate_sample_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")

    df = pd.DataFrame(
        {
            "feature_1": rng.normal(loc=0.0, scale=1.0, size=rows),
            "feature_2": rng.normal(loc=5.0, scale=2.0, size=rows),
            "target": rng.normal(loc=10.0, scale=3.0, size=rows),
            "category": rng.choice(["A", "B", "C"], size=rows, p=[0.4, 0.4, 0.2]),
            "flag": rng.choice([True, False], size=rows),
            "date": dates,
        }
    )
    # 약간의 결측치 추가
    nan_idx = rng.choice(rows, size=max(1, rows // 20), replace=False)
    df.loc[nan_idx, "feature_2"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_csv(file, sep: str = ",", decimal: str = ".", encoding: str = "utf-8") -> pd.DataFrame:
    try:
        # Streamlit 업로드 객체 또는 파일 경로 모두 처리
        if hasattr(file, "read"):
            data = file.read()
            buf = io.BytesIO(data)
            df = pd.read_csv(buf, sep=sep, decimal=decimal, encoding=encoding)
        else:
            df = pd.read_csv(file, sep=sep, decimal=decimal, encoding=encoding)
    except Exception as e:
        st.error(f"CSV 로드 오류: {e}")
        return pd.DataFrame()

    # 날짜형 자동 추정 (이름/형태 기반)
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if any(s.count("-") == 2 or s.count("/") == 2 for s in sample):
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    pass
    return df


def detect_column_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number", "datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
    return num_cols, cat_cols, dt_cols

