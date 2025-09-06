import hashlib
from typing import Optional

import pandas as pd
import streamlit as st


def _df_fingerprint(df: pd.DataFrame) -> str:
    # Lightweight fingerprint: shape + dtypes + first/last 100 rows hash
    sig = str((df.shape, tuple(map(str, df.dtypes))))
    head = df.head(100).to_json(date_unit="ns", orient="split")
    tail = df.tail(100).to_json(date_unit="ns", orient="split")
    h = hashlib.md5()
    h.update(sig.encode("utf-8", errors="ignore"))
    h.update(head.encode("utf-8", errors="ignore"))
    h.update(tail.encode("utf-8", errors="ignore"))
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def generate_profile_html(
    df: pd.DataFrame,
    *,
    minimal: bool = True,
    sample_n: Optional[int] = None,
    title: str = "Data Profiling Report",
) -> str:
    try:
        from ydata_profiling import ProfileReport
    except Exception as e:
        raise ImportError("ydata-profiling 패키지가 필요합니다. requirements.txt를 설치하세요.") from e

    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        dfx = df.sample(n=sample_n, random_state=42)
    else:
        dfx = df

    profile = ProfileReport(dfx, title=title, minimal=minimal, explorative=True)
    html = profile.to_html()
    # Include fingerprint to make cache key sensitive to df changes
    _ = _df_fingerprint(df)
    return html

