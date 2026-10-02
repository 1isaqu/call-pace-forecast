"""Read hourly counts and event amounts from CSV files you provide.

The files must follow docs/data-contract.md: a header row and the columns
`bucket,date,hour,count` (hourly) and `bucket,date,hour,amount` (events). Extra columns, such as
covariates, are passed through. This repository ships no data files; point the loader at yours.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..errors import ContractError, DataSourceError
from .common import filter_dates


class CsvLoader:
    """Loads from CSV on every call, so edits to the file are picked up immediately."""

    def __init__(
        self,
        hourly_path: str | Path,
        amounts_path: str | Path | None = None,
        sep: str = ",",
    ) -> None:
        self.hourly_path = Path(hourly_path)
        self.amounts_path = Path(amounts_path) if amounts_path else None
        self.sep = sep

    def load_hourly(self, start: date, end: date) -> pd.DataFrame:
        return filter_dates(self._read(self.hourly_path), start, end)

    def load_amounts(self, start: date, end: date) -> pd.DataFrame:
        if self.amounts_path is None:
            raise ContractError("no amounts file was configured")
        return filter_dates(self._read(self.amounts_path), start, end)

    def _read(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep=self.sep)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataSourceError(f"cannot read {path}: {exc}") from exc
