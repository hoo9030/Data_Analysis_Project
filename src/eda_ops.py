import pandas as pd


def basic_info(df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum()
    if mem < 1024:
        mem_str = f"{mem:.0f} B"
    elif mem < 1024 ** 2:
        mem_str = f"{mem / 1024:.1f} KB"
    elif mem < 1024 ** 3:
        mem_str = f"{mem / 1024 ** 2:.1f} MB"
    else:
        mem_str = f"{mem / 1024 ** 3:.1f} GB"

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory": mem_str,
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    ms = df.isna().sum().rename("missing_count").to_frame()
    ms["missing_pct"] = (ms["missing_count"] / len(df)).round(4)
    ms.index.name = "column"
    return ms.reset_index().sort_values("missing_count", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)

