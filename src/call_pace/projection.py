"""Projecting the day's final total from the volume so far.

Two ways to answer "how many will we have by closing time?":

  * `ProportionProjector`: the formula. If by this hour a bucket typically has completed a
    share `s` of its day, then projected total = cumulative so far / s.
  * `NeighborProjector`: a similarity model. Find past days that looked like today at this
    point and take the median of their final totals.

The comparison is the interesting part. By construction, share = cumulative / total, so the
formula inverts exactly the identity the similarity model has to learn indirectly. A model that
approximates a relation cannot beat a formula that states it unless it brings information the
formula ignores (weekday effects, seasonality, external events). `evaluate_projection` lets you
check that on your own data instead of taking it on faith.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .errors import ConfigError, InsufficientDataError
from .metrics import mae

MODES = ("hour_feature", "hour_partition")
_CHUNK = 512


class ProportionProjector:
    """Projected total = cumulative / median share of the day completed by that hour."""

    name = "proportion"

    def __init__(self, window_days: int | None = None) -> None:
        self.window_days = window_days
        self.shares: dict[tuple[str, int], float] = {}
        self.fallback: dict[int, float] = {}

    def fit(self, frame: pd.DataFrame) -> ProportionProjector:
        """Fit on a modelling frame (needs `share`); silent days have no share and are skipped."""
        usable = frame[frame["share"].notna()]
        if usable.empty:
            raise InsufficientDataError("no day with events to learn a share curve from")
        if self.window_days is not None:
            usable = self._trailing(usable)
        grouped = usable.groupby(["bucket", "hour"])["share"].median()
        self.shares = {(b, int(h)): float(v) for (b, h), v in grouped.items()}
        self.fallback = {
            int(h): float(v) for h, v in usable.groupby("hour")["share"].median().items()
        }
        return self

    def _trailing(self, usable: pd.DataFrame) -> pd.DataFrame:
        keep = (
            usable.groupby("bucket")["date"].transform(
                lambda s: s.rank(method="dense", ascending=False)
            )
            <= self.window_days
        )
        return usable[keep]

    def share_for(self, bucket: str, hour: int) -> float:
        return self.shares.get((bucket, hour), self.fallback.get(hour, float("nan")))

    def project_one(self, bucket: str, hour: int, cumulative: float) -> float:
        """Projected day total, or NaN when the typical share at this hour is zero or unknown."""
        share = self.share_for(bucket, hour)
        return cumulative / share if share and share > 0 else float("nan")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows = zip(frame["bucket"], frame["hour"], frame["cumulative"], strict=True)
        return np.array([self.project_one(b, int(h), c) for b, h, c in rows], dtype=float)


class NeighborProjector:
    """Median final total of the k most similar past days, in a min-max normalised space.

    Candidates are always restricted to the same bucket. With mode "hour_partition" they are
    also restricted to the same hour (exact match, distance on the rest). With mode
    "hour_feature" the hour is one more dimension of the distance and nothing is filtered.
    """

    name = "neighbors"

    def __init__(
        self,
        k: int = 10,
        mode: str = "hour_feature",
        covariates: Sequence[str] = (),
        start_hour: int = 8,
        end_hour: int = 19,
    ) -> None:
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        if k < 1:
            raise ConfigError("k must be positive")
        self.k, self.mode = k, mode
        self.covariates = tuple(covariates)
        self.start_hour, self.end_hour = start_hour, end_hour
        self.ood_count = 0
        self._ranges: dict[str, tuple[float, float]] = {}
        self._train: pd.DataFrame | None = None
        self._fallback = ProportionProjector()

    @property
    def _numeric(self) -> list[str]:
        return ["cumulative", *self.covariates]

    def fit(self, frame: pd.DataFrame) -> NeighborProjector:
        usable = frame[frame["day_total"] > 0]
        if usable.empty:
            raise InsufficientDataError("no day with events to learn from")
        self._ranges = {c: (float(usable[c].min()), float(usable[c].max())) for c in self._numeric}
        self._train = usable.reset_index(drop=True)
        self._fallback.fit(frame)
        return self

    def _normalise(self, frame: pd.DataFrame) -> np.ndarray:
        span = max(self.end_hour - self.start_hour, 1)
        columns = []
        if self.mode == "hour_feature":
            columns.append((frame["hour"].to_numpy(float) - self.start_hour) / span)
        for name in self._numeric:
            low, high = self._ranges[name]
            scaled = (
                (frame[name].to_numpy(float) - low) / (high - low)
                if high > low
                else np.zeros(len(frame))
            )
            self.ood_count += int(np.sum((scaled < 0) | (scaled > 1)))
            columns.append(np.clip(scaled, 0.0, 1.0))
        return np.column_stack(columns)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Projected day total per row. Cells without k candidates fall back to the formula."""
        if self._train is None:
            raise InsufficientDataError("the neighbour projector has not been fitted")
        self.ood_count = 0
        out = np.full(len(frame), np.nan)
        test_x, train_x = self._normalise(frame), self._normalise(self._train)
        keys = ["bucket", "hour"] if self.mode == "hour_partition" else ["bucket"]
        train_keys = self._train[keys]
        train_y = self._train["day_total"].to_numpy(float)
        by = keys if len(keys) > 1 else keys[0]
        for key, rows in frame[keys].reset_index(drop=True).groupby(by).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            same_cell = np.logical_and.reduce(
                [train_keys[c].to_numpy() == v for c, v in zip(keys, key, strict=True)]
            )
            if same_cell.sum() < self.k:
                out[rows] = self._fallback.predict(frame.iloc[rows])
            else:
                out[rows] = _nearest_median(
                    train_x[same_cell], train_y[same_cell], test_x[rows], self.k
                )
        return out


def _nearest_median(
    train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """Median target of the k nearest training rows for each query row (Euclidean)."""
    result = np.empty(len(query))
    for start in range(0, len(query), _CHUNK):
        block = query[start : start + _CHUNK]
        distances = np.sqrt(((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        result[start : start + len(block)] = np.median(train_y[nearest], axis=1)
    return result


def evaluate_projection(
    frame: pd.DataFrame,
    holdout_days: int = 20,
    k: int = 10,
    covariates: Sequence[str] = (),
    start_hour: int = 8,
    end_hour: int = 19,
) -> pd.DataFrame:
    """MAE of each projector per hour on the last `holdout_days` days (earlier days train).

    Rows where any projector has no prediction (NaN) are left out for ALL projectors, so every
    column is computed on exactly the same observations. The last row, hour "all", pools them.
    """
    dates = sorted(frame["date"].unique())
    if len(dates) <= holdout_days:
        raise InsufficientDataError(f"need more than {holdout_days} days, found {len(dates)}")
    test_dates = dates[-holdout_days:]
    train, test = frame[~frame["date"].isin(test_dates)], frame[frame["date"].isin(test_dates)]
    test = test[test["day_total"] > 0].reset_index(drop=True)
    projections = {"proportion": ProportionProjector().fit(train).predict(test)}
    for mode in MODES:
        model = NeighborProjector(k, mode, covariates, start_hour, end_hour).fit(train)
        projections[mode] = model.predict(test)
    valid = np.all([np.isfinite(p) for p in projections.values()], axis=0)
    scored = test[valid].assign(**{name: p[valid] for name, p in projections.items()})
    rows = [_projection_row(str(hour), group) for hour, group in scored.groupby("hour")]
    rows.append(_projection_row("all", scored))
    return pd.DataFrame(rows)


def _projection_row(label: str, group: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"hour": label, "n": len(group)}
    for name in ("proportion", *MODES):
        row[f"mae_{name}"] = mae(group["day_total"], group[name])
    return row
