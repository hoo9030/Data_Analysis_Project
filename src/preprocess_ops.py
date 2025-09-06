import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def iqr_filter(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rows outside [Q1-k*IQR, Q3+k*IQR] for each given column.

    Returns: (filtered_df, stats_df)
    stats_df columns: column, q1, q3, iqr, low, high, removed
    """
    if not cols:
        return df, pd.DataFrame(columns=["column", "q1", "q3", "iqr", "low", "high", "removed"])

    mask = pd.Series(True, index=df.index)
    rows_before = len(df)
    stats_rows = []

    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        valid = (s >= low) & (s <= high) | s.isna()
        mask &= valid
        stats_rows.append({
            "column": c,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "low": low,
            "high": high,
        })

    filtered = df[mask].copy()
    removed = rows_before - len(filtered)
    for r in stats_rows:
        r["removed"] = removed

    stats_df = pd.DataFrame(stats_rows)
    return filtered, stats_df


def remove_duplicates(df: pd.DataFrame, subset: List[str] | None = None, keep: str | bool = "first") -> pd.DataFrame:
    """Remove duplicate rows.

    keep: 'first', 'last', or False (drop all duplicates)
    subset: columns to consider for identifying duplicates; None=all
    """
    return df.drop_duplicates(subset=subset, keep=keep)

