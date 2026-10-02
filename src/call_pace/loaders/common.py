"""Helpers shared by every loader."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..contract import parse_date_column


def filter_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows whose `date` falls in [start, end], both ends included."""
    days = parse_date_column(df["date"])
    keep = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    return df[keep].reset_index(drop=True)
