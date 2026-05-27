"""Pandas 2/4 compatible dtype selection helpers."""

from __future__ import annotations

import pandas as pd


def categorical_columns(frame: pd.DataFrame) -> list:
    """Return object/category/string columns without pandas 4 deprecation noise."""
    cols = frame.select_dtypes(include=["category"]).columns.tolist()
    for col in frame.columns:
        if col in cols:
            continue
        dtype = frame[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            cols.append(col)
    return cols


def numeric_columns(frame: pd.DataFrame) -> list:
    return frame.select_dtypes(include=["number"]).columns.tolist()
