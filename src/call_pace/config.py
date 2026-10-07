"""Central configuration.

Every number here is a default chosen for a small synthetic example. None of them
is calibrated on real data: window lengths, cut-offs, fold sizes and k grids must be tuned on
your own history (see docs/evaluation.md).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import ConfigError

MODEL_KINDS = ("lookup", "knn", "quantile")
POSTPROCESS_MODES = ("ceil", "round", "none")
LOOKUP_STATS = ("median", "mean")
LOOKUP_TREATMENTS = ("none", "winsor", "clip3sd", "log1p")

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    """Modelling, calendar, evaluation and serving parameters."""

    # Calendar: the operating window of one day and the days the site is open (Monday = 0).
    start_hour: int = 8
    end_hour: int = 19
    operating_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    first_window_hours: int = 2

    # Optional exogenous columns carried by the hourly frame (constant within a bucket-day).
    covariates: tuple[str, ...] = ()

    # Serving post-processing applied to every model output and to every offline metric.
    postprocess: str = "ceil"

    # Phase 1: lookup baseline.
    lookup_window_days: int | None = 60
    lookup_stat: str = "median"
    lookup_treatment: str = "none"
    maturity_cutoff_days: Mapping[str, int] = field(default_factory=dict)

    # Phase 2: KNN.
    k_grid: tuple[int, ...] = (5, 7, 10, 15, 20)
    default_k: int = 10

    # Phase 3: LightGBM quantile bands.
    quantiles: tuple[float, float] = (0.25, 0.75)
    lgbm_params: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))

    # Walk-forward protocol.
    n_folds: int = 4
    fold_size_days: int = 10
    min_train_days: int = 30

    # Serving.
    default_model: str = "knn"
    routing: Mapping[str, str] = field(default_factory=dict)
    artifacts_dir: str = "artifacts"
    artifact_ttl_seconds: int = 86400

    def __post_init__(self) -> None:
        self._check_calendar()
        self._check_choices()
        self._check_evaluation()

    def _check_calendar(self) -> None:
        if not 0 <= self.start_hour <= self.end_hour <= 23:
            raise ConfigError("hours must satisfy 0 <= start_hour <= end_hour <= 23")
        days = tuple(self.operating_weekdays)
        if not days or any(d not in range(7) for d in days) or len(set(days)) != len(days):
            raise ConfigError("operating_weekdays must be unique integers in 0..6")
        if tuple(sorted(days)) != days:
            raise ConfigError("operating_weekdays must be sorted ascending")
        if not 1 <= self.first_window_hours < len(self.hours):
            raise ConfigError("first_window_hours must be between 1 and the number of hours - 1")

    def _check_choices(self) -> None:
        if self.postprocess not in POSTPROCESS_MODES:
            raise ConfigError(f"postprocess must be one of {POSTPROCESS_MODES}")
        if self.lookup_stat not in LOOKUP_STATS:
            raise ConfigError(f"lookup_stat must be one of {LOOKUP_STATS}")
        if self.lookup_treatment not in LOOKUP_TREATMENTS:
            raise ConfigError(f"lookup_treatment must be one of {LOOKUP_TREATMENTS}")
        for kind in (self.default_model, *self.routing.values()):
            if kind not in MODEL_KINDS:
                raise ConfigError(f"model kind {kind!r} must be one of {MODEL_KINDS}")

    def _check_evaluation(self) -> None:
        low, high = self.quantiles
        if not 0.0 < low < high < 1.0:
            raise ConfigError("quantiles must satisfy 0 < lower < upper < 1")
        if self.n_folds < 2 or self.fold_size_days < 1 or self.min_train_days < 1:
            raise ConfigError("need n_folds >= 2, fold_size_days >= 1 and min_train_days >= 1")
        if not self.k_grid or any(k < 1 for k in self.k_grid) or self.default_k < 1:
            raise ConfigError("k values must be positive")
        if self.lookup_window_days is not None and self.lookup_window_days < 1:
            raise ConfigError("lookup_window_days must be positive or None")

    @property
    def hours(self) -> range:
        return range(self.start_hour, self.end_hour + 1)

    @property
    def weekday_period(self) -> int:
        return len(self.operating_weekdays)

    @property
    def nominal_coverage(self) -> float:
        """Coverage the quantile band is supposed to have if it were calibrated."""
        return self.quantiles[1] - self.quantiles[0]

    def model_for(self, bucket: str) -> str:
        """Model kind used to serve `bucket` (per-bucket routing, then the default)."""
        return self.routing.get(bucket, self.default_model)

    def cutoff_for(self, bucket: str) -> int:
        """Most recent days of `bucket` that are ignored when fitting (immature data)."""
        return int(self.maturity_cutoff_days.get(bucket, 0))

    def replace(self, **changes: Any) -> Config:
        return replace(self, **changes)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        """Build a Config from `CALL_PACE_*` variables; unset variables keep their default."""
        env = os.environ if environ is None else environ
        changes: dict[str, Any] = {}
        _put_int(env, "CALL_PACE_START_HOUR", "start_hour", changes)
        _put_int(env, "CALL_PACE_END_HOUR", "end_hour", changes)
        _put_int(env, "CALL_PACE_LOOKUP_WINDOW_DAYS", "lookup_window_days", changes)
        _put_int(env, "CALL_PACE_ARTIFACT_TTL_SECONDS", "artifact_ttl_seconds", changes)
        _put_str(env, "CALL_PACE_ARTIFACTS_DIR", "artifacts_dir", changes)
        _put_str(env, "CALL_PACE_DEFAULT_MODEL", "default_model", changes)
        _put_pairs(env, "CALL_PACE_MATURITY_CUTOFF_DAYS", "maturity_cutoff_days", int, changes)
        _put_pairs(env, "CALL_PACE_ROUTING", "routing", str, changes)
        return cls(**changes)


def parse_pairs(text: str) -> dict[str, str]:
    """Parse `a=1,b=2` into {'a': '1', 'b': '2'}. Blank input gives an empty dict."""
    pairs: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ConfigError(f"expected key=value, got {item!r}")
        pairs[key.strip()] = value.strip()
    return pairs


def _put_str(env: Mapping[str, str], name: str, field_name: str, out: dict[str, Any]) -> None:
    if env.get(name):
        out[field_name] = env[name]


def _put_int(env: Mapping[str, str], name: str, field_name: str, out: dict[str, Any]) -> None:
    if env.get(name):
        try:
            out[field_name] = int(env[name])
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc


def _put_pairs(
    env: Mapping[str, str], name: str, field_name: str, cast: type, out: dict[str, Any]
) -> None:
    if env.get(name):
        try:
            out[field_name] = {k: cast(v) for k, v in parse_pairs(env[name]).items()}
        except ValueError as exc:
            raise ConfigError(f"{name} has an invalid value: {exc}") from exc
