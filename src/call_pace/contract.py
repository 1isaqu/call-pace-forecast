"""The time-series data contract and the loader protocol.

A loader is anything that can return rows for a date range. This module defines what those
rows must look like and validates them once, at the boundary, so everything downstream can
assume clean typed columns. See docs/data-contract.md for the human-readable version.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .config import Config
from .errors import ContractError

log = logging.getLogger(__name__)

HOURLY_COLUMNS = ("bucket", "date", "hour", "count")
AMOUNT_COLUMNS = ("bucket", "date", "hour", "amount")


@runtime_checkable
class HourlyLoader(Protocol):
    """Returns hourly counts. `start` and `end` are both inclusive calendar dates."""

    def load_hourly(self, start: date, end: date) -> pd.DataFrame: ...


@runtime_checkable
class AmountLoader(Protocol):
    """Returns one row per event with its amount (used to estimate a per-event ticket)."""

    def load_amounts(self, start: date, end: date) -> pd.DataFrame: ...


def empty_hourly(covariates: Iterable[str] = ()) -> pd.DataFrame:
    """A zero-row frame with the contract columns, for loaders that found nothing."""
    return pd.DataFrame({name: [] for name in (*HOURLY_COLUMNS, *covariates)})


def validate_hourly(
    df: pd.DataFrame, config: Config, covariates: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a clean copy of `df` or raise ContractError.

    Rules: required columns exist; `count` is finite and non-negative; rows outside the
    operating hours or operating weekdays are dropped (and logged); the key
    (bucket, date, hour) is unique; covariates are numeric and constant within a bucket-day.
    """
    names = tuple(config.covariates if covariates is None else covariates)
    _require_columns(df, (*HOURLY_COLUMNS, *names), "hourly")
    out = _normalise_keys(df[[*HOURLY_COLUMNS, *names]], "hourly")
    out["count"] = _non_negative(out["count"], "count")
    out = _restrict_to_schedule(out, config)
    _reject_duplicate_keys(out)
    _check_covariates(out, names)
    return out.sort_values(["bucket", "date", "hour"]).reset_index(drop=True)


def validate_amounts(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Same checks for event-level amounts. Several events per hour are expected."""
    _require_columns(df, AMOUNT_COLUMNS, "amounts")
    out = _normalise_keys(df[list(AMOUNT_COLUMNS)], "amounts")
    out["amount"] = _non_negative(out["amount"], "amount")
    out = _restrict_to_schedule(out, config)
    return out.sort_values(["bucket", "date", "hour"]).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{what} data is missing required columns: {missing}")


def _normalise_keys(df: pd.DataFrame, what: str) -> pd.DataFrame:
    out = df.copy()
    out["bucket"] = out["bucket"].astype(str).str.strip()
    out["date"] = _parse_dates(out["date"], what)
    out["hour"] = _parse_hours(out["hour"], what)
    return out


def parse_date_column(values: pd.Series) -> pd.Series:
    """Parse a column of dates or timestamps to naive midnight, whatever mix of formats it has.

    Raises ValueError (or TypeError) when a value cannot be read as a date.
    """
    try:
        parsed = pd.to_datetime(values)
    except ValueError:
        parsed = pd.to_datetime(values, format="mixed")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def _parse_dates(values: pd.Series, what: str) -> pd.Series:
    try:
        parsed = parse_date_column(values)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"{what} data has an unparseable `date` value: {exc}") from exc
    if parsed.isna().any():
        raise ContractError(f"{what} data has null `date` values")
    return parsed


def _parse_hours(values: pd.Series, what: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ContractError(f"{what} data has non-numeric or null `hour` values")
    if (numeric % 1 != 0).any():
        raise ContractError(f"{what} data has non-integer `hour` values")
    return numeric.astype("int64")


def _non_negative(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ContractError(f"`{name}` must be numeric and finite (no nulls)")
    if (numeric < 0).any():
        raise ContractError(f"`{name}` must be non-negative")
    return numeric.astype("float64")


def _restrict_to_schedule(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    in_hours = df["hour"].between(config.start_hour, config.end_hour)
    open_day = df["date"].dt.weekday.isin(config.operating_weekdays)
    keep = in_hours & open_day
    if not keep.all():
        log.warning("dropped %d rows outside the operating schedule", int((~keep).sum()))
    return df[keep].copy()


def _reject_duplicate_keys(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(["bucket", "date", "hour"], keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, ["bucket", "date", "hour"]].head(3).to_dict("records")
        raise ContractError(f"duplicate (bucket, date, hour) keys, for example {sample}")


def _check_covariates(df: pd.DataFrame, names: Sequence[str]) -> None:
    for name in names:
        values = pd.to_numeric(df[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"covariate `{name}` must be numeric and finite")
        df[name] = values.astype("float64")
        spread = df.groupby(["bucket", "date"])[name].nunique()
        if (spread > 1).any():
            raise ContractError(f"covariate `{name}` must be constant within a bucket-day")
