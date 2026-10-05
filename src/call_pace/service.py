"""The live pace service: today's counts in, a pace table out.

Responsibilities, in the order they run for one request:

  1. resolve the requested bucket (or "all") against what was trained;
  2. answer outside operating hours without touching models or data;
  3. load the fitted models each bucket is routed to (a failure here is an error);
  4. read today's counts through a loader (a failure here DEGRADES the response);
  5. build expected/actual/delta bands per bucket and aggregate them.

Failure policy, chosen for a wall display that must not go blank: a bad request is always
re-raised, a missing model is an error, but any failure while reading today's data returns a
well-formed payload with empty bands and a message in `errors`, so the consumer can keep
rendering and show that data is unavailable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from .artifacts import ArtifactStore, TtlCache
from .baseline import LookupForecaster
from .config import Config
from .contract import AmountLoader, HourlyLoader, validate_amounts, validate_hourly
from .errors import ConfigError, InvalidRequestError
from .features import FeatureSpec, serving_frame
from .knn import KnnForecaster
from .pace import Band, aggregate_bands, build_bands, summarize
from .quantile import QuantileForecaster
from .schedule import weekday_position
from .value import TicketLookup, expected_value, observed_value_by_hour

log = logging.getLogger(__name__)

ALL = "all"
VALUE_MODEL = "ticket_x_expected_count"
CovariateProvider = Callable[[str, date], Mapping[str, float]]


@dataclass(frozen=True)
class _Models:
    lookup: LookupForecaster | None = None
    knn: KnnForecaster | None = None
    quantile: QuantileForecaster | None = None
    ticket: TicketLookup | None = None


@dataclass(frozen=True)
class _Forecast:
    expected: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


def _num(value: float | None) -> float | int | None:
    """Whole numbers as ints so counts read as counts in JSON."""
    if value is None:
        return None
    return int(value) if float(value).is_integer() else round(float(value), 3)


class PaceService:
    """Builds the intraday pace payload for one bucket or for all of them."""

    def __init__(
        self,
        config: Config,
        loader: HourlyLoader,
        store: ArtifactStore,
        amount_loader: AmountLoader | None = None,
        covariate_provider: CovariateProvider | None = None,
        clock: Callable[[], datetime] = datetime.now,
        data_ttl_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.loader = loader
        self.store = store
        self.amount_loader = amount_loader
        self.covariate_provider = covariate_provider
        self._clock = clock
        self._data_ttl = data_ttl_seconds
        self._cache = TtlCache(data_ttl_seconds, monotonic)

    def pace(self, bucket: str = ALL) -> dict[str, Any]:
        """The pace payload: {"meta": ..., "data": ..., "errors": [...]}."""
        now = self._clock()
        buckets = self._resolve(bucket)
        spec = self.store.spec()
        self._check_consistent(spec)
        if not self._in_operation(now, spec):
            return self._empty(now, bucket, buckets, in_operation=False)
        self._require_covariates(spec)
        models = self._load_models(buckets)
        try:
            observed, values = self._observe(now.date(), models)
        except InvalidRequestError:
            raise
        except Exception:
            log.error("could not read today's data (bucket=%s)", bucket, exc_info=True)
            return self._empty(now, bucket, buckets, in_operation=True, degraded=True)
        views = [self._view(b, models, spec, observed, values, now) for b in buckets]
        return self._payload(now, bucket, buckets, views, models)

    # --- request resolution ---------------------------------------------------------------

    def _resolve(self, bucket: str) -> list[str]:
        known = self.store.known_buckets()
        if bucket == ALL:
            return known
        if bucket not in known:
            raise InvalidRequestError(f"unknown bucket {bucket!r}; known buckets: {known}")
        return [bucket]

    @staticmethod
    def _in_operation(now: datetime, spec: FeatureSpec) -> bool:
        open_hour = spec.start_hour <= now.hour <= spec.end_hour
        return open_hour and weekday_position(now.date(), spec.weekdays) is not None

    def _check_consistent(self, spec: FeatureSpec) -> None:
        """The serving configuration must describe the same day the models were trained on."""
        mine = (
            self.config.start_hour,
            self.config.end_hour,
            tuple(self.config.operating_weekdays),
            self.config.first_window_hours,
            tuple(self.config.covariates),
        )
        trained = (
            spec.start_hour,
            spec.end_hour,
            spec.weekdays,
            spec.first_window_hours,
            spec.covariates,
        )
        if mine != trained:
            raise ConfigError(
                "the serving configuration (hours, weekdays, first-window hours, covariates) "
                "differs from the one the models were trained with"
            )

    def _require_covariates(self, spec: FeatureSpec) -> None:
        if spec.covariates and self.covariate_provider is None:
            raise ConfigError(
                f"the models need covariates {list(spec.covariates)}: pass a provider"
            )

    def _load_models(self, buckets: Sequence[str]) -> _Models:
        kinds = {self.config.model_for(b) for b in buckets}
        loaded = {kind: self.store.load(kind) for kind in kinds}
        ticket = None
        if self.amount_loader is not None and self.store.exists("ticket"):
            candidate = self.store.load("ticket")
            ticket = candidate if candidate.covers(buckets) else None
        return _Models(loaded.get("lookup"), loaded.get("knn"), loaded.get("quantile"), ticket)

    # --- reading today --------------------------------------------------------------------

    def _cached(self, key: Any, load: Callable[[], Any]) -> Any:
        return load() if self._data_ttl <= 0 else self._cache.get_or_load(key, load)

    def _observe(
        self, day: date, models: _Models
    ) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
        raw = self._cached(("hourly", day), lambda: self.loader.load_hourly(day, day))
        clean = validate_hourly(raw, self.config, covariates=())
        observed: dict[str, dict[int, float]] = {}
        for bucket, hour, count in zip(clean["bucket"], clean["hour"], clean["count"], strict=True):
            observed.setdefault(str(bucket), {})[int(hour)] = float(count)
        values: dict[str, dict[int, float]] = {}
        if models.ticket is not None and self.amount_loader is not None:
            loader = self.amount_loader
            amounts = self._cached(("amounts", day), lambda: loader.load_amounts(day, day))
            values = observed_value_by_hour(validate_amounts(amounts, self.config), day)
        return observed, values

    # --- building one bucket --------------------------------------------------------------

    def _view(
        self,
        bucket: str,
        models: _Models,
        spec: FeatureSpec,
        observed: Mapping[str, Mapping[int, float]],
        values: Mapping[str, Mapping[int, float]],
        now: datetime,
    ) -> tuple[list[Band], list[Band] | None]:
        hours = list(spec.hours)
        seen = observed.get(bucket, {})
        forecast = self._forecast(bucket, models, spec, seen, now)
        bands = build_bands(
            forecast.expected, seen, now.hour, hours, forecast.lower, forecast.upper
        )
        if models.ticket is None:
            return bands, None
        tickets = [models.ticket.ticket_for(bucket, h) for h in hours]
        money = expected_value(forecast.expected, tickets)
        return bands, build_bands(money, values.get(bucket, {}), now.hour, hours, digits=2)

    def _forecast(
        self,
        bucket: str,
        models: _Models,
        spec: FeatureSpec,
        seen: Mapping[int, float],
        now: datetime,
    ) -> _Forecast:
        kind = self.config.model_for(bucket)
        if kind == "lookup":
            return _Forecast(models.lookup.expected_for_day(bucket, list(spec.hours)))
        covariates = self.covariate_provider(bucket, now.date()) if spec.covariates else None
        frame = serving_frame(spec, bucket, now.date(), seen, now.hour, covariates)
        if kind == "knn":
            return _Forecast(models.knn.predict(frame))
        bands = models.quantile.predict_bands(frame)
        return _Forecast(
            bands["center"].to_numpy(), bands["lower"].to_numpy(), bands["upper"].to_numpy()
        )

    # --- payloads -------------------------------------------------------------------------

    def _meta(self, now: datetime, bucket: str, buckets: Sequence[str], in_operation: bool) -> dict:
        kinds = {b: self.config.model_for(b) for b in buckets}
        label = (
            next(iter(kinds.values()))
            if len(set(kinds.values())) == 1
            else ("hybrid(" + ", ".join(f"{b}={k}" for b, k in kinds.items()) + ")")
        )
        return {
            "generated_at": now.isoformat(),
            "in_operation": in_operation,
            "bucket": bucket,
            "buckets": list(buckets),
            "model": label,
        }

    def _empty(
        self,
        now: datetime,
        bucket: str,
        buckets: Sequence[str],
        in_operation: bool,
        degraded: bool = False,
    ) -> dict[str, Any]:
        errors = [{"msg": "failed to read today's data"}] if degraded else []
        data = {"current_hour": now.hour, "cumulative_actual": 0, "bands": []}
        return {
            "meta": self._meta(now, bucket, buckets, in_operation),
            "data": data,
            "errors": errors,
        }

    def _payload(
        self,
        now: datetime,
        bucket: str,
        buckets: Sequence[str],
        views: list[tuple[list[Band], list[Band] | None]],
        models: _Models,
    ) -> dict[str, Any]:
        bands = aggregate_bands([count for count, _ in views])
        totals = summarize(bands)
        data: dict[str, Any] = {
            "current_hour": now.hour,
            "cumulative_actual": _num(totals.cumulative_actual),
            "expected_total": _num(totals.expected_total),
            "projected_close": _num(totals.projected_close),
            "bands": [_count_band(b) for b in bands],
        }
        meta = self._meta(now, bucket, buckets, True)
        if models.ticket is not None:
            meta["value_model"] = VALUE_MODEL
            self._add_value(data, [money for _, money in views if money is not None])
        return {"meta": meta, "data": data, "errors": []}

    @staticmethod
    def _add_value(data: dict[str, Any], money: list[list[Band]]) -> None:
        bands = aggregate_bands(money, digits=2)
        totals = summarize(bands, digits=2)
        data["value_cumulative_actual"] = totals.cumulative_actual
        data["value_expected_total"] = totals.expected_total
        data["value_projected_close"] = totals.projected_close
        for row, band in zip(data["bands"], bands, strict=True):
            row.update(
                {
                    "expected_value": band.expected,
                    "actual_value": band.actual,
                    "delta_value": band.delta,
                    "status_value": band.status,
                    "cumulative_value": band.cumulative,
                }
            )


def _count_band(band: Band) -> dict[str, Any]:
    row = band.to_dict()
    for key in ("expected", "actual", "delta", "cumulative", "lower", "upper"):
        row[key] = _num(row[key])
    return row
