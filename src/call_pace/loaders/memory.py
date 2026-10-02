"""A loader over data already in memory. Used by tests and by notebooks."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..errors import ContractError
from .common import filter_dates


class InMemoryLoader:
    """Serves hourly counts (and optionally event amounts) from DataFrames."""

    def __init__(self, hourly: pd.DataFrame, amounts: pd.DataFrame | None = None) -> None:
        self._hourly = hourly
        self._amounts = amounts

    def load_hourly(self, start: date, end: date) -> pd.DataFrame:
        return filter_dates(self._hourly, start, end)

    def load_amounts(self, start: date, end: date) -> pd.DataFrame:
        if self._amounts is None:
            raise ContractError("this loader was created without amounts")
        return filter_dates(self._amounts, start, end)
