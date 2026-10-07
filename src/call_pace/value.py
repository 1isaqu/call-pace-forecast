"""Expected value as expected count times a typical ticket.

Forecasting an amount directly is hard: amounts are heavy tailed and dominated by a few large
events. The decomposition reuses the count model and needs only a ticket lookup:

    expected value of an hour = expected count of that hour x typical ticket

The count comes from any of the count models. The ticket is a plain lookup (the median of
recent event amounts, or a winsorised mean), the same family as the baseline, with no
trainable model.

Design choices:
  * the median is the default estimator; a winsorised mean is the alternative for heavy tails;
  * the window is measured in OPERATING days of a calendar built from the first to the last
    day with events, so silent days still age the window;
  * a per-bucket maturity cut-off drops the most recent days, which can be incomplete;
  * a sparse (bucket, hour) cell falls back to the bucket-wide ticket;
  * a bucket with no events in the window is an error. A made-up placeholder ticket would
    silently turn into made-up money on a dashboard.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import ArtifactError, ConfigError, InsufficientDataError
from .metrics import mae, mean_error, wape
from .schedule import operating_days
from .validation import Fold, make_folds

ESTIMATORS = ("median", "winsor")
GRANULARITIES = ("bucket", "hour_bucket")
FORMAT_VERSION = 1


def ticket_estimate(
    values: Sequence[float], estimator: str = "median", limits: tuple[float, float] = (5.0, 95.0)
) -> float:
    """Typical amount: the median, or the mean after clipping to two percentiles."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise InsufficientDataError("cannot estimate a ticket from no events")
    if estimator == "median":
        return float(np.median(data))
    if estimator == "winsor":
        low, high = np.percentile(data, limits)
        return float(np.clip(data, low, high).mean())
    raise ConfigError(f"estimator must be one of {ESTIMATORS}")


@dataclass(frozen=True)
class TicketTable:
    """Tickets of one bucket: a bucket-wide value and optional per-hour values."""

    overall: float
    hours: Mapping[int, float]

    def for_hour(self, hour: int) -> float:
        return self.hours.get(hour, self.overall)


def estimate_tickets(
    window: pd.DataFrame,
    estimator: str,
    granularity: str,
    min_cell_obs: int,
    allow_empty: bool = False,
) -> TicketTable:
    """Tickets from the events of one bucket inside the training window."""
    if granularity not in GRANULARITIES:
        raise ConfigError(f"granularity must be one of {GRANULARITIES}")
    if window.empty:
        if allow_empty:
            return TicketTable(0.0, {})
        raise InsufficientDataError("no events in the training window: refusing to invent a ticket")
    overall = ticket_estimate(window["amount"].to_numpy(), estimator)
    if granularity == "bucket":
        return TicketTable(overall, {})
    cells = {
        int(hour): ticket_estimate(group["amount"].to_numpy(), estimator)
        for hour, group in window.groupby("hour")
        if len(group) >= min_cell_obs
    }
    return TicketTable(overall, cells)


def _calendar(amounts: pd.DataFrame, weekdays: Sequence[int]) -> list[pd.Timestamp]:
    """Every operating day between the first and last day this bucket had events."""
    first, last = amounts["date"].min().date(), amounts["date"].max().date()
    return [pd.Timestamp(d) for d in operating_days(first, last, weekdays)]


def _trim(days: list[pd.Timestamp], cutoff: int) -> list[pd.Timestamp]:
    """Drop the most recent `cutoff` days (they may still be incomplete)."""
    return days[: max(len(days) - cutoff, 0)] if cutoff > 0 else days


class TicketLookup:
    """Typical ticket per bucket (and optionally per hour), fitted on a trailing window."""

    name = "ticket_lookup"

    def __init__(
        self,
        estimator: str = "median",
        granularity: str = "bucket",
        window_days: int = 60,
        maturity_cutoff_days: Mapping[str, int] | None = None,
        min_cell_obs: int = 5,
        weekdays: Sequence[int] = (0, 1, 2, 3, 4),
    ) -> None:
        if estimator not in ESTIMATORS:
            raise ConfigError(f"estimator must be one of {ESTIMATORS}")
        if granularity not in GRANULARITIES:
            raise ConfigError(f"granularity must be one of {GRANULARITIES}")
        if window_days < 1 or min_cell_obs < 1:
            raise ConfigError("window_days and min_cell_obs must be positive")
        self.estimator, self.granularity = estimator, granularity
        self.window_days, self.min_cell_obs = window_days, min_cell_obs
        self.maturity_cutoff_days = dict(maturity_cutoff_days or {})
        self.weekdays = tuple(weekdays)
        self.tables: dict[str, TicketTable] = {}
        self.trained: dict[str, dict[str, Any]] = {}

    def fit(self, amounts: pd.DataFrame) -> TicketLookup:
        """Fit on validated event amounts (columns bucket, date, hour, amount)."""
        self.tables, self.trained = {}, {}
        for bucket, events in amounts.groupby("bucket"):
            days = _trim(_calendar(events, self.weekdays), self.maturity_cutoff_days.get(bucket, 0))
            kept = days[-self.window_days :]
            if not kept:
                raise InsufficientDataError(f"no usable days left for bucket {bucket!r}")
            window = events[events["date"].isin(kept)]
            self.tables[bucket] = estimate_tickets(
                window, self.estimator, self.granularity, self.min_cell_obs
            )
            self.trained[bucket] = {
                "from": kept[0].date().isoformat(),
                "to": kept[-1].date().isoformat(),
                "events": int(len(window)),
            }
        return self

    def covers(self, buckets: Sequence[str]) -> bool:
        """True when a ticket was fitted for every bucket in `buckets`."""
        return all(bucket in self.tables for bucket in buckets)

    def ticket_for(self, bucket: str, hour: int) -> float:
        """Ticket for a bucket-hour, rounded to cents. Unknown buckets are an error."""
        if bucket not in self.tables:
            raise ArtifactError(f"no ticket fitted for bucket {bucket!r}")
        return round(max(0.0, self.tables[bucket].for_hour(hour)), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "ticket_lookup",
            "format_version": FORMAT_VERSION,
            "estimator": self.estimator,
            "granularity": self.granularity,
            "window_days": self.window_days,
            "min_cell_obs": self.min_cell_obs,
            "maturity_cutoff_days": self.maturity_cutoff_days,
            "weekdays": list(self.weekdays),
            "tables": {
                b: {"overall": t.overall, "hours": {str(h): v for h, v in t.hours.items()}}
                for b, t in self.tables.items()
            },
            "trained": self.trained,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TicketLookup:
        if data.get("model") != "ticket_lookup" or data.get("format_version") != FORMAT_VERSION:
            raise ArtifactError("not a ticket lookup artifact of a supported version")
        model = cls(
            data["estimator"],
            data["granularity"],
            data["window_days"],
            data["maturity_cutoff_days"],
            data["min_cell_obs"],
            data["weekdays"],
        )
        model.tables = {
            b: TicketTable(float(t["overall"]), {int(h): float(v) for h, v in t["hours"].items()})
            for b, t in data["tables"].items()
        }
        model.trained = dict(data["trained"])
        return model

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TicketLookup:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ArtifactError(f"cannot read ticket artifact {path}: {exc}") from exc


def expected_value(expected_counts: Sequence[float], tickets: Sequence[float]) -> list[float]:
    """Expected value per hour: count times ticket, to cents."""
    return [round(float(c) * float(t), 2) for c, t in zip(expected_counts, tickets, strict=True)]


@dataclass(frozen=True)
class TicketConfig:
    estimator: str
    granularity: str
    window_days: int


@dataclass(frozen=True)
class TicketGrid:
    """Candidate configurations, listed so that the earlier candidate is the more parsimonious.

    Ties on mean error are resolved by that order: bucket-wide before per-hour, longer window
    before shorter, median before winsorised mean.
    """

    estimators: tuple[str, ...] = ESTIMATORS
    granularities: tuple[str, ...] = GRANULARITIES
    windows: tuple[int, ...] = (60, 40)

    def candidates(self) -> list[TicketConfig]:
        return [
            TicketConfig(estimator, granularity, window)
            for granularity in self.granularities
            for window in self.windows
            for estimator in self.estimators
        ]


@dataclass
class TicketSelection:
    """Result of `select_ticket_config`."""

    results: pd.DataFrame
    winners: dict[str, TicketConfig]
    holdout: pd.DataFrame
    baselines: pd.DataFrame


def select_ticket_config(
    amounts: pd.DataFrame,
    hours: Sequence[int],
    weekdays: Sequence[int] = (0, 1, 2, 3, 4),
    grid: TicketGrid | None = None,
    n_folds: int = 4,
    fold_size_days: int = 10,
    min_train_days: int = 30,
    min_cell_obs: int = 5,
    maturity_cutoff_days: Mapping[str, int] | None = None,
) -> TicketSelection:
    """Walk-forward choice of ticket estimator, granularity and window for each bucket.

    Predicted value of a band is its ACTUAL event count times the estimated ticket. Using the
    real count isolates the ticket error from the count-model error. The numbers are therefore
    a best case for the whole decomposition, not an estimate of production error.

    Configurations are ranked by mean band MAE on folds 1..n-1; the last fold only confirms.
    Two naive references are scored on the same folds: (a) the median band value per hour and
    (b) a flat daily total (median of recent daily totals), judged by daily WAPE.
    """
    cutoffs = dict(maturity_cutoff_days or {})
    candidates = (grid or TicketGrid()).candidates()
    rows: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    winners: dict[str, TicketConfig] = {}
    for bucket, events in amounts.groupby("bucket"):
        days = _trim(_calendar(events, weekdays), cutoffs.get(bucket, 0))
        folds = make_folds(days, n_folds, fold_size_days, min_train_days)
        band = _band_table(events, days, hours)
        scored = _score_candidates(bucket, candidates, folds, events, band, min_cell_obs)
        rows += scored
        winner = _pick_winner(candidates, scored, n_folds)
        winners[bucket] = winner
        holdout += [r for r in scored if _same(r, winner) and r["fold"] == n_folds]
        baselines += _baseline_rows(bucket, winner, folds, band, scored)
    return TicketSelection(
        pd.DataFrame(rows), winners, pd.DataFrame(holdout), pd.DataFrame(baselines)
    )


def _band_table(
    events: pd.DataFrame, days: list[pd.Timestamp], hours: Sequence[int]
) -> pd.DataFrame:
    """Operating day x hour grid with actual value and event count (zeros included)."""
    grid = pd.MultiIndex.from_product([days, list(hours)], names=["date", "hour"]).to_frame(
        index=False
    )
    sums = events.groupby(["date", "hour"])["amount"].agg(value="sum", n="count").reset_index()
    merged = grid.merge(sums, on=["date", "hour"], how="left")
    merged[["value", "n"]] = merged[["value", "n"]].fillna(0.0)
    return merged


def _score_candidates(
    bucket: str,
    candidates: list[TicketConfig],
    folds: list[Fold],
    events: pd.DataFrame,
    band: pd.DataFrame,
    min_cell_obs: int,
) -> list[dict[str, Any]]:
    rows = []
    for config in candidates:
        for fold in folds:
            window_days = list(fold.train_dates[-config.window_days :])
            window = events[events["date"].isin(window_days)]
            table = estimate_tickets(
                window, config.estimator, config.granularity, min_cell_obs, allow_empty=True
            )
            test = band[band["date"].isin(fold.test_dates)]
            predicted = test["n"].to_numpy() * np.array(
                [table.for_hour(int(h)) for h in test["hour"]]
            )
            rows.append(_metric_row(bucket, config, fold.index, test, predicted))
    return rows


def _metric_row(
    bucket: str, config: TicketConfig, fold_index: int, test: pd.DataFrame, predicted: np.ndarray
) -> dict[str, Any]:
    daily = pd.DataFrame({"date": test["date"], "actual": test["value"], "pred": predicted})
    per_day = daily.groupby("date")[["actual", "pred"]].sum()
    return {
        "bucket": bucket,
        "fold": fold_index,
        "estimator": config.estimator,
        "granularity": config.granularity,
        "window_days": config.window_days,
        "mae_band": mae(test["value"], predicted),
        "me_band": mean_error(test["value"], predicted),
        "wape_day": wape(per_day["actual"], per_day["pred"]),
    }


def _same(row: Mapping[str, Any], config: TicketConfig) -> bool:
    return (row["estimator"], row["granularity"], row["window_days"]) == (
        config.estimator,
        config.granularity,
        config.window_days,
    )


def _pick_winner(
    candidates: list[TicketConfig], scored: list[dict[str, Any]], n_folds: int
) -> TicketConfig:
    """Lowest mean band MAE on folds 1..n-1. `min` keeps the first of equals (most parsimonious)."""

    def mean_mae(config: TicketConfig) -> float:
        values = [r["mae_band"] for r in scored if _same(r, config) and r["fold"] < n_folds]
        return float(np.mean(values))

    return min(candidates, key=mean_mae)


def _baseline_rows(
    bucket: str,
    winner: TicketConfig,
    folds: list[Fold],
    band: pd.DataFrame,
    scored: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for fold in folds:
        window_days = list(fold.train_dates[-winner.window_days :])
        train = band[band["date"].isin(window_days)]
        test = band[band["date"].isin(fold.test_dates)]
        by_hour = train.groupby("hour")["value"].median()
        pred_a = test["hour"].map(by_hour).fillna(float(train["value"].median())).to_numpy()
        daily_train = train.groupby("date")["value"].sum()
        flat = float(daily_train.median()) if len(daily_train) else 0.0
        daily_test = test.groupby("date")["value"].sum()
        own = next(r for r in scored if _same(r, winner) and r["fold"] == fold.index)
        rows.append(
            {
                "bucket": bucket,
                "fold": fold.index,
                "mae_band": own["mae_band"],
                "mae_band_baseline_a": mae(test["value"], pred_a),
                "wape_day": own["wape_day"],
                "wape_day_baseline_b": wape(daily_test, np.full(len(daily_test), flat)),
            }
        )
    return rows


def amounts_to_counts(amounts: pd.DataFrame) -> pd.DataFrame:
    """Event count per (bucket, date, hour) from event-level rows, in the hourly contract shape."""
    counts = amounts.groupby(["bucket", "date", "hour"]).size().rename("count").reset_index()
    counts["count"] = counts["count"].astype(float)
    return counts


def observed_value_by_hour(amounts: pd.DataFrame, day: date) -> dict[str, dict[int, float]]:
    """Sum of amounts per bucket and hour for one day (the "actual" side of value bands)."""
    today = amounts[amounts["date"] == pd.Timestamp(day)]
    sums = today.groupby(["bucket", "hour"])["amount"].sum()
    out: dict[str, dict[int, float]] = {}
    for (bucket, hour), total in sums.items():
        out.setdefault(str(bucket), {})[int(hour)] = float(total)
    return out
