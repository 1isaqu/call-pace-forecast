"""Feature engineering shared by training, validation, simulation and serving.

The same functions build the training frame and the single-day frame used at serving time.
That keeps the two paths identical, which is the main defence against train/serve skew.

Leakage rules baked in here:
  * `cumulative_lag` is the total BEFORE the hour being predicted (a shift by one inside the
    day), so predicting hour H only uses what is known after hour H-1.
  * `first_window_total` (sum of the first hours of the day) is zero until that window has
    closed, so it is never visible while it is still being filled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .errors import ConfigError, ContractError
from .schedule import cyclic_encoding, weekday_position, weekday_positions

KEYS = ["bucket", "date"]


@dataclass(frozen=True)
class FeatureSpec:
    """Everything a fitted model needs to rebuild its feature vector. Persisted with it."""

    buckets: tuple[str, ...]
    covariates: tuple[str, ...] = ()
    start_hour: int = 8
    end_hour: int = 19
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    first_window_hours: int = 2

    @classmethod
    def from_config(cls, config: Config, buckets: Sequence[str]) -> FeatureSpec:
        return cls(
            buckets=tuple(sorted(set(buckets))),
            covariates=tuple(config.covariates),
            start_hour=config.start_hour,
            end_hour=config.end_hour,
            weekdays=tuple(config.operating_weekdays),
            first_window_hours=config.first_window_hours,
        )

    @classmethod
    def from_data(cls, hourly: pd.DataFrame, config: Config) -> FeatureSpec:
        return cls.from_config(config, hourly["bucket"].unique())

    @property
    def hours(self) -> range:
        return range(self.start_hour, self.end_hour + 1)

    @property
    def window_end_hour(self) -> int:
        """First hour at which `first_window_total` becomes available."""
        return self.start_hour + self.first_window_hours

    @property
    def bucket_columns(self) -> tuple[str, ...]:
        """One column for two buckets, one-hot for more, none for a single bucket."""
        count = len(self.buckets)
        if count == 1:
            return ()
        if count == 2:
            return ("bucket_code",)
        return tuple(f"bucket_is_{name}" for name in self.buckets)

    @property
    def names(self) -> tuple[str, ...]:
        """Model input columns, in the order every model is fitted with."""
        return (
            "hour",
            "dow_sin",
            "dow_cos",
            "cumulative_lag",
            *self.bucket_columns,
            "first_window_total",
            *self.covariates,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSpec:
        try:
            return cls(
                buckets=tuple(data["buckets"]),
                covariates=tuple(data.get("covariates", ())),
                start_hour=int(data["start_hour"]),
                end_hour=int(data["end_hour"]),
                weekdays=tuple(int(d) for d in data["weekdays"]),
                first_window_hours=int(data["first_window_hours"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid feature spec: {exc}") from exc


def bucket_values(spec: FeatureSpec, bucket: str) -> list[float]:
    """Encoded bucket features, matching `spec.bucket_columns`."""
    if bucket not in spec.buckets:
        raise ContractError(f"unknown bucket {bucket!r}; known buckets: {list(spec.buckets)}")
    if len(spec.buckets) == 1:
        return []
    if len(spec.buckets) == 2:
        return [float(spec.buckets.index(bucket))]
    return [1.0 if name == bucket else 0.0 for name in spec.buckets]


def complete_grid(
    hourly: pd.DataFrame, spec: FeatureSpec, *, fill_missing_days: bool = False
) -> pd.DataFrame:
    """Expand sparse rows to every operating hour of every bucket-day, filling zeros.

    A bucket-day is "present" when the loader returned at least one row for it. With
    `fill_missing_days` every operating weekday between a bucket's first and last date is
    added as well, which treats a silent day as zero activity instead of a holiday.
    """
    pairs = _bucket_days(hourly, spec)
    if fill_missing_days:
        pairs = _add_silent_days(pairs, spec)
    hours = pd.DataFrame({"hour": list(spec.hours)})
    grid = pairs.merge(hours, how="cross")
    counts = hourly[[*KEYS, "hour", "count"]]
    grid = grid.merge(counts, on=[*KEYS, "hour"], how="left")
    grid["count"] = grid["count"].fillna(0.0)
    return grid.sort_values([*KEYS, "hour"]).reset_index(drop=True)


def _bucket_days(hourly: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if spec.covariates:
        return hourly.groupby(KEYS, as_index=False)[list(spec.covariates)].first()
    return hourly[KEYS].drop_duplicates().reset_index(drop=True)


def _add_silent_days(pairs: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if spec.covariates:
        raise ConfigError("fill_missing_days cannot be combined with covariates")
    frames = []
    for bucket, group in pairs.groupby("bucket"):
        span = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        days = span[span.weekday.isin(spec.weekdays)]
        frames.append(pd.DataFrame({"bucket": bucket, "date": days}))
    return pd.concat(frames, ignore_index=True)


def build_frame(
    hourly: pd.DataFrame, spec: FeatureSpec, *, fill_missing_days: bool = False
) -> pd.DataFrame:
    """The modelling frame: one row per bucket-day-hour with every engineered column.

    `hourly` must already satisfy the contract (see `contract.validate_hourly`).
    """
    frame = complete_grid(hourly, spec, fill_missing_days=fill_missing_days)
    by_day = frame.groupby(KEYS)["count"]
    frame["cumulative"] = by_day.cumsum()
    frame["cumulative_lag"] = frame["cumulative"] - frame["count"]
    frame["day_total"] = by_day.transform("sum")
    frame["share"] = (frame["cumulative"] / frame["day_total"]).where(frame["day_total"] > 0)
    frame["first_window_total"] = _first_window_total(frame, spec)
    _add_weekday_features(frame, spec)
    _add_bucket_features(frame, spec)
    return frame


def _first_window_total(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    inside = frame["hour"] < spec.window_end_hour
    window_sum = frame["count"].where(inside, 0.0).groupby([frame["bucket"], frame["date"]])
    return window_sum.transform("sum").where(~inside, 0.0)


def _add_weekday_features(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    position = weekday_positions(frame["date"], spec.weekdays)
    if position.isna().any():
        raise ContractError("data contains dates outside the operating weekdays")
    frame["weekday_pos"] = position.astype("int64")
    frame["dow_sin"], frame["dow_cos"] = cyclic_encoding(frame["weekday_pos"], len(spec.weekdays))


def _add_bucket_features(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    unknown = sorted(set(frame["bucket"]) - set(spec.buckets))
    if unknown:
        raise ContractError(f"unknown buckets {unknown}; known buckets: {list(spec.buckets)}")
    if not spec.bucket_columns:
        return
    encoded = np.array([bucket_values(spec, b) for b in frame["bucket"]], dtype=float)
    for position, name in enumerate(spec.bucket_columns):
        frame[name] = encoded[:, position]


def feature_matrix(frame: pd.DataFrame, names: Sequence[str] | FeatureSpec) -> np.ndarray:
    """Model input as a float64 array in the column order of `names`."""
    columns = list(names.names if isinstance(names, FeatureSpec) else names)
    return frame[columns].to_numpy(dtype="float64")


def serving_frame(
    spec: FeatureSpec,
    bucket: str,
    day: date,
    observed: Mapping[int, float],
    current_hour: int,
    covariates: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Features for every hour of `day`, as they are known when `current_hour` is running.

    `observed` maps hour to the count seen so far; only hours before `current_hour` are used.
    Hours that have not started yet are conditioned on the cumulative reached so far, they are
    not rolled forward with predictions.
    """
    position = weekday_position(day, spec.weekdays)
    if position is None:
        raise ContractError(f"{day} is not an operating weekday")
    extra = _covariate_values(spec, covariates)
    done = {h: float(observed.get(h, 0.0)) for h in spec.hours if h < current_hour}
    window_total = sum(v for h, v in done.items() if h < spec.window_end_hour)
    rows = []
    for hour in spec.hours:
        lag = sum(v for h, v in done.items() if h < hour)
        rows.append(
            [hour, position, lag, window_total if hour >= spec.window_end_hour else 0.0, *extra]
        )
    return _serving_to_frame(spec, bucket, day, rows)


def _covariate_values(spec: FeatureSpec, covariates: Mapping[str, float] | None) -> list[float]:
    given = covariates or {}
    missing = [name for name in spec.covariates if name not in given]
    if missing:
        raise ContractError(f"serving needs covariate values for {missing}")
    return [float(given[name]) for name in spec.covariates]


def _serving_to_frame(spec: FeatureSpec, bucket: str, day: date, rows: list[list]) -> pd.DataFrame:
    columns = ["hour", "weekday_pos", "cumulative_lag", "first_window_total", *spec.covariates]
    frame = pd.DataFrame(rows, columns=columns)
    frame["dow_sin"], frame["dow_cos"] = cyclic_encoding(frame["weekday_pos"], len(spec.weekdays))
    for name, value in zip(spec.bucket_columns, bucket_values(spec, bucket), strict=True):
        frame[name] = value
    frame.insert(0, "bucket", bucket)
    frame.insert(1, "date", pd.Timestamp(day))
    return frame
