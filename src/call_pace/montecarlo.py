"""Monte Carlo stress test of a retraining policy under volume growth.

Walk-forward validation asks "how well did the model do on the past". It cannot answer
"what happens if volume keeps growing and we retrain every month". This module simulates that:

  1. Learn from history: the hourly share profile per (bucket, weekday) and the empirical pool
     of daily totals per (bucket, weekday).
  2. For each scenario, draw a monthly growth rate from a triangular distribution and build the
     next months: sample a daily total from the pool, scale it by the accumulated growth, spread
     it with the hourly profile, and add negative-binomial noise (variance = mu + mu^2 / r, so
     counts are over-dispersed relative to a Poisson).
  3. Replay the policy: after each simulated month refit on a rolling window of the most recent
     days (real tail plus simulated days) and score the model on the following month.

What this does and does not tell you: it measures sensitivity of the model's error and of its
advantage over the baseline to growth, under the stated assumptions (a stable intraday and
weekly shape, independent days, a fixed dispersion). It is not a forecast of real future error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .baseline import LookupForecaster
from .config import Config
from .errors import ConfigError, InsufficientDataError
from .features import FeatureSpec, build_frame
from .knn import KnnForecaster
from .metrics import directional_accuracy, mae
from .schedule import next_operating_days

_KEYS = ["bucket", "date"]


@dataclass(frozen=True)
class MonteCarloConfig:
    """Scenario parameters.

    Every default was chosen for the synthetic example; set them from your own expectations.
    """

    n_scenarios: int = 200
    n_months: int = 6
    days_per_month: int = 21
    growth_low: float = 0.0
    growth_mode: float = 0.03
    growth_high: float = 0.06
    dispersion: float = 12.0
    window_days: int = 60
    k: int = 10
    seed: int = 0
    directional_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not self.growth_low <= self.growth_mode <= self.growth_high:
            raise ConfigError("growth must satisfy low <= mode <= high")
        if self.growth_low == self.growth_high:
            raise ConfigError("growth_low and growth_high must differ")
        if min(self.n_scenarios, self.n_months, self.days_per_month, self.window_days, self.k) < 1:
            raise ConfigError("counts, window and k must be positive")
        if self.dispersion <= 0:
            raise ConfigError("dispersion must be positive")


class ScenarioGenerator:
    """Builds synthetic months from the shape of historical days."""

    def __init__(self, history: pd.DataFrame, config: Config, mc: MonteCarloConfig) -> None:
        self.config, self.mc = config, mc
        self.hours = list(config.hours)
        self.profiles = self._profiles(history)
        self.pools = self._pools(history)
        self.covariates = {c: history.groupby("bucket")[c].mean() for c in config.covariates}
        if not self.pools:
            raise InsufficientDataError("history has no usable day to learn a shape from")

    def _profiles(self, history: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
        mean = history.groupby(["bucket", "weekday_pos", "hour"])["count"].mean().unstack("hour")
        mean = mean.reindex(columns=self.hours, fill_value=0.0)
        totals = mean.sum(axis=1)
        shares = mean.div(totals.where(totals > 0), axis=0).fillna(0.0)
        return {(b, int(w)): row.to_numpy() for (b, w), row in shares.iterrows()}

    def _pools(self, history: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
        days = history.groupby([*_KEYS, "weekday_pos"], as_index=False)["day_total"].first()
        return {
            (b, int(w)): g["day_total"].to_numpy(float)
            for (b, w), g in days.groupby(["bucket", "weekday_pos"])
            if (b, int(w)) in self.profiles
        }

    def month(self, after: date, scale: float, rng: np.random.Generator) -> pd.DataFrame:
        """`days_per_month` operating days after `after`, with daily totals scaled by `scale`."""
        weekdays = self.config.operating_weekdays
        days = next_operating_days(after, self.mc.days_per_month, weekdays)
        rows = [(b, d, w) for d in days for (b, w) in self._keys_for(d, weekdays)]
        totals = np.array([rng.choice(self.pools[(b, w)]) * scale for b, _, w in rows])
        shares = np.array([self.profiles[(b, w)] for b, _, w in rows])
        counts = _negative_binomial(rng, totals[:, None] * shares, self.mc.dispersion)
        return self._to_frame(rows, counts)

    def _keys_for(self, day: date, weekdays: tuple[int, ...]) -> list[tuple[str, int]]:
        position = weekdays.index(day.weekday())
        return sorted((b, w) for (b, w) in self.pools if w == position)

    def _to_frame(self, rows: list[tuple[str, date, int]], counts: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "bucket": np.repeat([b for b, _, _ in rows], len(self.hours)),
                "date": np.repeat(pd.to_datetime([d for _, d, _ in rows]), len(self.hours)),
                "hour": np.tile(self.hours, len(rows)),
                "count": counts.ravel(),
            }
        )
        for name, by_bucket in self.covariates.items():
            frame[name] = frame["bucket"].map(by_bucket).astype(float)
        return frame


def _negative_binomial(rng: np.random.Generator, mean: np.ndarray, dispersion: float) -> np.ndarray:
    mean = np.maximum(mean, 1e-3)
    return rng.negative_binomial(dispersion, dispersion / (dispersion + mean)).astype(float)


def _simulate_horizon(
    generator: ScenarioGenerator, last_day: date, mc: MonteCarloConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[list[pd.Timestamp]]]:
    """n_months + 1 simulated months (the last one is only ever a test month)."""
    scale, after, months, month_dates = 1.0, last_day, [], []
    for _ in range(mc.n_months + 1):
        scale *= 1.0 + rng.triangular(mc.growth_low, mc.growth_mode, mc.growth_high)
        month = generator.month(after, scale, rng)
        months.append(month)
        dates = sorted(month["date"].unique())
        month_dates.append(dates)
        after = dates[-1].date()
    return pd.concat(months, ignore_index=True), month_dates


@dataclass
class MonteCarloResult:
    """One row per (scenario, month) with the model's and the baseline's error."""

    runs: pd.DataFrame
    mc: MonteCarloConfig

    def summary(self, percentiles: tuple[int, ...] = (10, 50, 90)) -> pd.DataFrame:
        """Percentiles of MAE, baseline MAE and directional accuracy for each month."""
        rows = []
        for month, group in self.runs.groupby("month"):
            row: dict[str, float] = {"month": int(month)}
            for metric in ("mae", "mae_lookup", "dir_acc"):
                values = np.nanpercentile(group[metric], percentiles)
                row.update(
                    {f"{metric}_p{p}": float(v) for p, v in zip(percentiles, values, strict=True)}
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def probability_directional_above(
        self, threshold: float | None = None, month: int | None = None
    ) -> float:
        """Share of scenarios whose directional accuracy exceeds `threshold` in `month`."""
        bar = self.mc.directional_threshold if threshold is None else threshold
        rows = self._month(month)
        return float((rows["dir_acc"] > bar).mean())

    def probability_beats_baseline(self, month: int | None = None) -> float:
        """Share of scenarios where the model's MAE is below the baseline's in `month`."""
        rows = self._month(month)
        return float((rows["mae"] < rows["mae_lookup"]).mean())

    def _month(self, month: int | None) -> pd.DataFrame:
        target = self.mc.n_months if month is None else month
        rows = self.runs[self.runs["month"] == target]
        if rows.empty:
            raise ConfigError(f"no results for month {target}")
        return rows


def run_monte_carlo(
    history: pd.DataFrame,
    config: Config,
    spec: FeatureSpec,
    mc: MonteCarloConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MonteCarloResult:
    """Replay the retraining policy on `mc.n_scenarios` simulated futures.

    `history` is a modelling frame (`features.build_frame` output). Scenario s uses the random
    stream seeded by (mc.seed, s), so any single scenario can be reproduced on its own.
    """
    mc = mc or MonteCarloConfig()
    generator = ScenarioGenerator(history, config, mc)
    tail_dates = sorted(history["date"].unique())[-mc.window_days :]
    tail = history[history["date"].isin(tail_dates)]
    last_day = pd.Timestamp(max(history["date"])).date()
    runs: list[dict[str, float]] = []
    for scenario in range(mc.n_scenarios):
        rng = np.random.default_rng([mc.seed, scenario])
        simulated, month_dates = _simulate_horizon(generator, last_day, mc, rng)
        pool = pd.concat([tail, build_frame(simulated, spec)], ignore_index=True)
        runs += _replay(pool, month_dates, scenario, config, spec, mc)
        if progress:
            progress(scenario + 1, mc.n_scenarios)
    return MonteCarloResult(pd.DataFrame(runs), mc)


def _replay(
    pool: pd.DataFrame,
    month_dates: list[list[pd.Timestamp]],
    scenario: int,
    config: Config,
    spec: FeatureSpec,
    mc: MonteCarloConfig,
) -> list[dict[str, float]]:
    """After each month: refit on the rolling window, score on the next month."""
    all_dates = sorted(pool["date"].unique())
    rows = []
    for month in range(mc.n_months):
        cutoff = month_dates[month][-1]
        window = [d for d in all_dates if d <= cutoff][-mc.window_days :]
        train = pool[pool["date"].isin(window)]
        test = pool[pool["date"].isin(month_dates[month + 1])]
        rows.append(
            {"scenario": scenario, "month": month + 1, **_score(train, test, config, spec, mc)}
        )
    return rows


def _score(
    train: pd.DataFrame, test: pd.DataFrame, config: Config, spec: FeatureSpec, mc: MonteCarloConfig
) -> dict[str, float]:
    lookup = LookupForecaster(
        config.lookup_window_days,
        None,
        config.lookup_stat,
        config.lookup_treatment,
        config.postprocess,
    ).fit(train)
    knn = KnnForecaster(spec, mc.k, config.postprocess).fit(train)
    baseline, predicted = lookup.predict(test), knn.predict(test)
    actual = test["count"].to_numpy()
    return {
        "mae": mae(actual, predicted),
        "mae_lookup": mae(actual, baseline),
        "dir_acc": directional_accuracy(actual, predicted, baseline),
    }
