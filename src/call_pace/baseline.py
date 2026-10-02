"""Phase 1: the lookup baseline.

The expected count for an hour is a robust central value of what that (bucket, hour) cell
produced over a trailing window of recent days. There is nothing to tune beyond the window,
which is why it is the model every other model has to beat.

Design points:
  * The hour is a PARTITION KEY (exact match), never a distance dimension. Used as a distance
    feature it gets diluted by the other features; used as a key it is exact.
  * The median is the default estimator because daily volumes have outliers.
  * A trailing window instead of all history, because volume regimes shift.
  * Per-bucket maturity cut-off: the most recent days of a bucket can be incomplete (records
    that arrive late), so they can be excluded before fitting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LOOKUP_STATS, LOOKUP_TREATMENTS
from .errors import ArtifactError, ConfigError, InsufficientDataError
from .metrics import apply_postprocess

FORMAT_VERSION = 1


def central_estimate(
    values: Sequence[float], stat: str = "median", treatment: str = "none"
) -> float:
    """Central value of `values` after an optional outlier treatment.

    Treatments: `winsor` clips to the 5th and 95th percentiles, `clip3sd` caps the top at
    mean + 3 standard deviations, `log1p` averages on the log scale and maps back.
    """
    if stat not in LOOKUP_STATS:
        raise ConfigError(f"stat must be one of {LOOKUP_STATS}")
    if treatment not in LOOKUP_TREATMENTS:
        raise ConfigError(f"treatment must be one of {LOOKUP_TREATMENTS}")
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise InsufficientDataError("cannot estimate a central value from no observations")
    if treatment == "winsor":
        data = np.clip(data, *np.percentile(data, [5, 95]))
    elif treatment == "clip3sd":
        data = np.minimum(data, data.mean() + 3.0 * data.std(ddof=1 if data.size > 1 else 0))
    elif treatment == "log1p":
        data = np.log1p(data)
    centre = float(np.median(data) if stat == "median" else np.mean(data))
    return float(np.expm1(centre)) if treatment == "log1p" else centre


def keyed_lookup_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    keys: Sequence[str],
    stat: str = "median",
    treatment: str = "none",
    postprocess: str = "ceil",
) -> np.ndarray:
    """Lookup keyed by any columns, falling back to the hour alone for unseen cells.

    This is the experiment form: `keys` may be ("hour",) to pool every bucket or
    ("hour", "bucket") to separate them. The production model is `LookupForecaster`.
    """
    keys = list(keys)

    def estimator(series: pd.Series) -> float:
        return central_estimate(series.to_numpy(), stat, treatment)

    cell = train.groupby(keys)["count"].apply(estimator).rename("cell").reset_index()
    fallback = train.groupby("hour")["count"].apply(estimator).rename("fallback").reset_index()
    merged = test[keys].merge(cell, on=keys, how="left").merge(fallback, on="hour", how="left")
    raw = merged["cell"].where(merged["cell"].notna(), merged["fallback"]).fillna(0.0)
    return apply_postprocess(raw.to_numpy(), postprocess)


class LookupForecaster:
    """Median (or other robust) count per (bucket, hour) over a trailing window of days."""

    name = "lookup"

    def __init__(
        self,
        window_days: int | None = 60,
        maturity_cutoff_days: Mapping[str, int] | None = None,
        stat: str = "median",
        treatment: str = "none",
        postprocess: str = "ceil",
    ) -> None:
        self.window_days = window_days
        self.maturity_cutoff_days = dict(maturity_cutoff_days or {})
        self.stat = stat
        self.treatment = treatment
        self.postprocess = postprocess
        self.table: dict[str, dict[int, float]] = {}
        self.fallback: dict[int, float] = {}
        self.trained: dict[str, dict[str, Any]] = {}

    def fit(self, frame: pd.DataFrame) -> LookupForecaster:
        """Fit on a frame with columns bucket, date, hour, count and a complete hour grid."""
        self.table, self.trained = {}, {}
        pooled: list[pd.DataFrame] = []
        for bucket, group in frame.groupby("bucket"):
            window = self._window(bucket, group)
            self.table[bucket] = self._by_hour(window)
            self.trained[bucket] = self._trained_range(window)
            pooled.append(window)
        self.fallback = self._by_hour(pd.concat(pooled, ignore_index=True))
        return self

    def _window(self, bucket: str, group: pd.DataFrame) -> pd.DataFrame:
        dates = np.sort(group["date"].unique())
        cutoff = int(self.maturity_cutoff_days.get(bucket, 0))
        if cutoff:
            dates = dates[:-cutoff] if cutoff < len(dates) else dates[:0]
        if self.window_days is not None:
            dates = dates[-self.window_days :]
        if len(dates) == 0:
            raise InsufficientDataError(f"no usable days left for bucket {bucket!r}")
        return group[group["date"].isin(dates)]

    def _by_hour(self, window: pd.DataFrame) -> dict[int, float]:
        return {
            int(hour): central_estimate(group["count"].to_numpy(), self.stat, self.treatment)
            for hour, group in window.groupby("hour")
        }

    @staticmethod
    def _trained_range(window: pd.DataFrame) -> dict[str, Any]:
        return {
            "from": pd.Timestamp(window["date"].min()).date().isoformat(),
            "to": pd.Timestamp(window["date"].max()).date().isoformat(),
            "days": int(window["date"].nunique()),
        }

    def predict_raw_one(self, bucket: str, hour: int) -> float:
        """Unrounded expectation. Unseen bucket falls back to the pooled hour, then to zero."""
        cell = self.table.get(bucket, {})
        if hour in cell:
            return cell[hour]
        return self.fallback.get(hour, 0.0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = [
            self.predict_raw_one(b, int(h))
            for b, h in zip(frame["bucket"], frame["hour"], strict=True)
        ]
        return apply_postprocess(np.asarray(raw), self.postprocess)

    def expected_for_day(self, bucket: str, hours: Sequence[int]) -> np.ndarray:
        """Post-processed expectation for each hour of a day (a lookup ignores the day so far)."""
        raw = [self.predict_raw_one(bucket, int(h)) for h in hours]
        return apply_postprocess(np.asarray(raw), self.postprocess)

    def nonzero_hours(self, bucket: str) -> int:
        """How many hours have a post-processed expectation above zero for `bucket`."""
        hours = sorted(self.table.get(bucket, {}))
        return int(np.sum(self.expected_for_day(bucket, hours) > 0)) if hours else 0

    def is_informative(self, bucket: str, min_nonzero_hours: int = 6) -> bool:
        """Review trigger for routing: a lookup whose median is zero almost everywhere shows a
        flat zero band that can never flag a shortfall, so a sparse bucket should be served by
        a model that reacts to the day so far until its volume grows."""
        return self.nonzero_hours(bucket) >= min_nonzero_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "lookup",
            "format_version": FORMAT_VERSION,
            "stat": self.stat,
            "treatment": self.treatment,
            "postprocess": self.postprocess,
            "window_days": self.window_days,
            "maturity_cutoff_days": self.maturity_cutoff_days,
            "table": {b: {str(h): v for h, v in cells.items()} for b, cells in self.table.items()},
            "fallback": {str(h): v for h, v in self.fallback.items()},
            "trained": self.trained,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LookupForecaster:
        if data.get("model") != "lookup" or data.get("format_version") != FORMAT_VERSION:
            raise ArtifactError("not a lookup artifact of a supported version")
        model = cls(
            window_days=data["window_days"],
            maturity_cutoff_days=data["maturity_cutoff_days"],
            stat=data["stat"],
            treatment=data["treatment"],
            postprocess=data["postprocess"],
        )
        model.table = {
            b: {int(h): float(v) for h, v in c.items()} for b, c in data["table"].items()
        }
        model.fallback = {int(h): float(v) for h, v in data["fallback"].items()}
        model.trained = dict(data["trained"])
        return model

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LookupForecaster:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ArtifactError(f"cannot read lookup artifact {path}: {exc}") from exc
