"""Loaders: anything that can return rows for a date range.

`InMemoryLoader` and `CsvLoader` cover files and tests. `SqlServerLoader` (in `sqlserver`)
builds read-only T-SQL against the generic schema in docs/data-contract.md. To connect your own
source, implement `HourlyLoader` (and optionally `AmountLoader`) from `call_pace.contract`.
"""

from __future__ import annotations

from .common import filter_dates
from .csv_loader import CsvLoader
from .memory import InMemoryLoader

__all__ = ["CsvLoader", "InMemoryLoader", "filter_dates"]
