"""Invented data so the pipeline can run end to end with nothing connected.

The generator produces a plausible shape, not a realistic one: a bimodal hourly curve, a
weekday effect, a day-level multiplier (which is what makes "volume so far" informative about
the rest of the day) and negative-binomial noise. All parameters are invented.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .schedule import next_operating_days


def _default_levels() -> dict[str, float]:
    return {"auto": 45.0, "consumer": 28.0}


def _default_tickets() -> dict[str, float]:
    return {"auto": 500.0, "consumer": 120.0}


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs of the synthetic generator."""

    daily_level: Mapping[str, float] = field(default_factory=_default_levels)
    weekday_multipliers: tuple[float, ...] = (1.10, 1.05, 1.00, 0.95, 0.85)
    day_sigma: float = 0.30
    dispersion: float = 12.0
    growth_per_month: float = 0.0
    covariate_effect: float = 0.0
    event_period_days: int = 20


def hourly_shape(hours: range) -> np.ndarray:
    """Share of the day's volume that falls in each hour (a morning and an afternoon peak)."""
    h = np.arange(hours.start, hours.stop, dtype=float)
    curve = 0.6 + np.exp(-(((h - 10.5) / 1.8) ** 2)) + 1.1 * np.exp(-(((h - 15.0) / 2.0) ** 2))
    return curve / curve.sum()


def make_synthetic_hourly(
    n_days: int = 90,
    start: date = date(2026, 1, 5),
    seed: int = 7,
    config: Config | None = None,
    spec: SyntheticSpec | None = None,
    with_covariate: bool = False,
) -> pd.DataFrame:
    """Hourly counts for every bucket over `n_days` operating days beginning on or after `start`.

    Returns the contract columns (bucket, date, hour, count) and, with `with_covariate`, a
    `days_since_event` column that cycles through `event_period_days`.
    """
    cfg, params = config or Config(), spec or SyntheticSpec()
    rng = np.random.default_rng(seed)
    days = next_operating_days(start - timedelta(days=1), n_days, cfg.operating_weekdays)
    shape = hourly_shape(cfg.hours)
    rows = []
    for index, day in enumerate(days):
        since_event = index % params.event_period_days
        for bucket, level in sorted(params.daily_level.items()):
            mu_day = _daily_mean(level, day, index, since_event, cfg, params, rng)
            counts = _draw_counts(rng, mu_day * shape, params.dispersion)
            rows.append(
                _day_rows(bucket, day, cfg, counts, since_event if with_covariate else None)
            )
    return pd.concat(rows, ignore_index=True)


def _daily_mean(
    level: float,
    day: date,
    index: int,
    since_event: int,
    cfg: Config,
    params: SyntheticSpec,
    rng: np.random.Generator,
) -> float:
    weekday = cfg.operating_weekdays.index(day.weekday()) % len(params.weekday_multipliers)
    growth = (1.0 + params.growth_per_month) ** (index / 21.0)
    noise = rng.lognormal(-0.5 * params.day_sigma**2, params.day_sigma)
    event = 1.0 + params.covariate_effect * np.exp(-since_event / 5.0)
    return level * params.weekday_multipliers[weekday] * growth * noise * event


def _draw_counts(rng: np.random.Generator, mean: np.ndarray, dispersion: float) -> np.ndarray:
    mean = np.maximum(mean, 1e-3)
    return rng.negative_binomial(dispersion, dispersion / (dispersion + mean)).astype(float)


def _day_rows(
    bucket: str, day: date, cfg: Config, counts: np.ndarray, since_event: int | None
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"bucket": bucket, "date": pd.Timestamp(day), "hour": list(cfg.hours), "count": counts}
    )
    if since_event is not None:
        frame["days_since_event"] = float(since_event)
    return frame


def make_synthetic_amounts(
    hourly: pd.DataFrame,
    seed: int = 11,
    median_amount: Mapping[str, float] | None = None,
    sigma: float = 0.8,
) -> pd.DataFrame:
    """One row per counted event with a log-normal amount (heavy right tail, like real tickets)."""
    rng = np.random.default_rng(seed)
    medians = dict(median_amount or _default_tickets())
    repeats = hourly["count"].round().astype(int).to_numpy()
    events = hourly.loc[hourly.index.repeat(repeats), ["bucket", "date", "hour"]].reset_index(
        drop=True
    )
    centre = np.log(events["bucket"].map(lambda b: medians.get(b, 100.0)).to_numpy(float))
    events["amount"] = np.round(np.exp(rng.normal(centre, sigma)), 2)
    return events
