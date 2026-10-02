"""Intraday pace: expected versus actual per hour, as a triad of numbers.

For every hour of the operating day the output is:

    expected   what the model says this hour should produce
    actual     what it produced (known once the hour has closed)
    delta      actual - expected

The triad is the product. It is also the explanation: a manager can see the reasoning without
reading a model. Better models only make `expected` more accurate; the shape of the answer
never changes. This module is pure logic with no framework, I/O or model imports.

Status of an hour:
    above        closed, actual > expected
    below        closed, actual < expected
    ok           closed, actual == expected
    in_progress  the hour that is running now
    future       an hour that has not started
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

ABOVE, BELOW, OK = "above", "below", "ok"
IN_PROGRESS, FUTURE = "in_progress", "future"
ABOVE_BAND, WITHIN_BAND, BELOW_BAND = "above_band", "within_band", "below_band"
_OPEN_STATUSES = (IN_PROGRESS, FUTURE)


@dataclass(frozen=True)
class Band:
    """One hour of the pace table."""

    hour: int
    expected: float
    actual: float | None
    delta: float | None
    status: str
    cumulative: float | None
    lower: float | None = None
    upper: float | None = None
    position: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_delta(delta: float) -> str:
    """above / below / ok from the sign of delta."""
    if delta > 0:
        return ABOVE
    return BELOW if delta < 0 else OK


def band_position(actual: float, lower: float, upper: float) -> str:
    """Where a closed hour landed relative to the model's quantile band."""
    if actual < lower:
        return BELOW_BAND
    return ABOVE_BAND if actual > upper else WITHIN_BAND


def build_bands(
    expected: Sequence[float],
    observed: Mapping[int, float],
    current_hour: int,
    hours: Sequence[int],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    digits: int | None = None,
) -> list[Band]:
    """Pair each hour's expectation with what was observed.

    `observed` maps hour to the value seen so far; only hours before `current_hour` are
    treated as closed. An hour with no row in `observed` counts as zero once it has closed,
    EXCEPT when `observed` is completely empty: with no data at all the day has not visibly
    started, so even the current hour is reported as `future` rather than `in_progress`, and
    closed hours are scored against zero.

    `digits` rounds delta and the running total (use 2 for money, None for counts).
    """
    bands: list[Band] = []
    running = 0.0
    for i, hour in enumerate(hours):
        low = None if lower is None else float(lower[i])
        high = None if upper is None else float(upper[i])
        if not observed and hour >= current_hour:
            bands.append(Band(hour, float(expected[i]), None, None, FUTURE, None, low, high))
        elif hour < current_hour:
            actual = float(observed.get(hour, 0.0))
            running += actual
            bands.append(_closed(hour, float(expected[i]), actual, running, low, high, digits))
        elif hour == current_hour:
            bands.append(Band(hour, float(expected[i]), None, None, IN_PROGRESS, None, low, high))
        else:
            bands.append(Band(hour, float(expected[i]), None, None, FUTURE, None, low, high))
    return bands


def _closed(
    hour: int,
    expected: float,
    actual: float,
    running: float,
    low: float | None,
    high: float | None,
    digits: int | None,
) -> Band:
    delta = actual - expected
    position = None if low is None or high is None else band_position(actual, low, high)
    if digits is not None:
        delta, running = round(delta, digits), round(running, digits)
    return Band(hour, expected, actual, delta, classify_delta(delta), running, low, high, position)


def aggregate_bands(per_bucket: Sequence[Sequence[Band]], digits: int | None = None) -> list[Band]:
    """Sum the bands of several buckets hour by hour.

    Quantile bounds are dropped: the sum of per-bucket bounds is not a bound on the sum.
    The status of a not-yet-closed hour comes from the first bucket; a closed hour is
    re-classified from the aggregate delta.
    """
    if len(per_bucket) == 1:
        return list(per_bucket[0])
    merged: list[Band] = []
    for hour_bands in zip(*per_bucket, strict=True):
        expected = sum(b.expected for b in hour_bands)
        closed = [b for b in hour_bands if b.actual is not None]
        if not closed:
            merged.append(
                Band(hour_bands[0].hour, expected, None, None, hour_bands[0].status, None)
            )
            continue
        actual = sum(b.actual for b in closed)
        cumulative = sum(b.cumulative for b in hour_bands if b.cumulative is not None)
        delta = actual - expected
        if digits is not None:
            expected, actual = round(expected, digits), round(actual, digits)
            delta, cumulative = round(delta, digits), round(cumulative, digits)
        merged.append(
            Band(hour_bands[0].hour, expected, actual, delta, classify_delta(delta), cumulative)
        )
    return merged


@dataclass(frozen=True)
class PaceTotals:
    """Day-level numbers derived from a list of bands."""

    cumulative_actual: float
    expected_total: float
    projected_close: float


def summarize(bands: Sequence[Band], digits: int | None = None) -> PaceTotals:
    """Actual so far, expected for the whole day, and the projected close.

    projected close = actual so far + expected for every hour that has not closed yet. It is a
    point estimate built from per-hour expectations; it carries no interval.
    """
    actual = sum(b.actual for b in bands if b.actual is not None)
    expected = sum(b.expected for b in bands)
    remaining = sum(b.expected for b in bands if b.status in _OPEN_STATUSES)
    values = (actual, expected, actual + remaining)
    if digits is not None:
        values = tuple(round(v, digits) for v in values)
    return PaceTotals(*values)
