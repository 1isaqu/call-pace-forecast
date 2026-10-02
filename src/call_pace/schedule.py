"""Operating-day and time-of-day helpers shared by features, simulation and serving."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd


def weekday_position(day: date | datetime | pd.Timestamp, weekdays: Sequence[int]) -> int | None:
    """Position of `day` inside the operating weekdays, or None when the site is closed.

    With the default Monday..Friday schedule Monday is 0 and Friday is 4.
    """
    try:
        return list(weekdays).index(day.weekday())
    except ValueError:
        return None


def weekday_positions(dates: pd.Series, weekdays: Sequence[int]) -> pd.Series:
    """Vectorised `weekday_position`. Closed days become NaN."""
    mapping = {wd: pos for pos, wd in enumerate(weekdays)}
    return pd.to_datetime(dates).dt.weekday.map(mapping)


def cyclic_encoding(position: np.ndarray | float, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Sine and cosine of a position on a cycle, so the last day sits next to the first."""
    angle = 2.0 * np.pi * np.asarray(position, dtype=float) / period
    return np.sin(angle), np.cos(angle)


def operating_days(start: date, end: date, weekdays: Iterable[int]) -> list[date]:
    """Every operating date in [start, end], both ends included."""
    allowed = set(weekdays)
    span = (end - start).days
    days = (start + timedelta(days=offset) for offset in range(span + 1))
    return [d for d in days if d.weekday() in allowed]


def next_operating_days(after: date, count: int, weekdays: Iterable[int]) -> list[date]:
    """The `count` operating dates strictly after `after`."""
    allowed = set(weekdays)
    found: list[date] = []
    cursor = after
    while len(found) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() in allowed:
            found.append(cursor)
    return found
