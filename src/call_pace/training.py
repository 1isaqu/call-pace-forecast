"""Fit every model and write the artifacts a service needs.

Training uses all the data it is given: the walk-forward folds in `validation` exist to decide
and to estimate error, and the model that is deployed is then refitted on everything.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import pandas as pd

from .artifacts import ArtifactStore
from .baseline import LookupForecaster
from .config import MODEL_KINDS, Config
from .errors import ConfigError, InsufficientDataError
from .features import FeatureSpec, build_frame
from .knn import KnnForecaster, knn_cross_validate
from .quantile import QuantileForecaster
from .validation import make_folds, split_frame
from .value import TicketLookup

log = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """What a training run did, also written into the manifest."""

    kinds: list[str]
    buckets: list[str]
    rows: int
    days: int
    first_day: str
    last_day: str
    k: int | None = None
    k_selected_by_cv: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def select_k(frame: pd.DataFrame, spec: FeatureSpec, config: Config) -> int | None:
    """Best k by cross-validation on every walk-forward fold except the last.

    Returns None when the history is too short for the configured protocol.
    """
    try:
        folds = make_folds(
            frame["date"].unique(), config.n_folds, config.fold_size_days, config.min_train_days
        )
    except InsufficientDataError:
        return None
    splits = [split_frame(frame, fold) for fold in folds[:-1]]
    return knn_cross_validate(splits, spec, config.k_grid, config.postprocess).best_k


def train_models(
    hourly: pd.DataFrame,
    config: Config,
    store: ArtifactStore,
    amounts: pd.DataFrame | None = None,
    kinds: Sequence[str] = ("lookup", "knn", "quantile"),
    k: int | None = None,
    fill_missing_days: bool = False,
) -> TrainingReport:
    """Fit the requested model kinds on validated hourly data and save them into `store`.

    `hourly` must satisfy the contract (`contract.validate_hourly`). When `amounts` is given a
    ticket lookup is fitted as well. `k` overrides cross-validated selection of the KNN k.
    """
    unknown = set(kinds) - set(MODEL_KINDS)
    if unknown:
        raise ConfigError(f"unknown model kinds {sorted(unknown)}; choose from {MODEL_KINDS}")
    spec = FeatureSpec.from_data(hourly, config)
    frame = build_frame(hourly, spec, fill_missing_days=fill_missing_days)
    report = _report(frame, spec, list(kinds))
    if "lookup" in kinds:
        store.save("lookup", _lookup(config).fit(frame))
    if "knn" in kinds:
        _train_knn(frame, spec, config, store, k, report)
    if "quantile" in kinds:
        model = QuantileForecaster(spec, config.quantiles, config.lgbm_params, config.postprocess)
        store.save("quantile", model.fit(frame))
    if amounts is not None:
        store.save("ticket", _ticket(config).fit(amounts))
        report.kinds.append("ticket")
    store.write_manifest(spec, report.to_dict())
    return report


def _lookup(config: Config) -> LookupForecaster:
    return LookupForecaster(
        config.lookup_window_days,
        config.maturity_cutoff_days,
        config.lookup_stat,
        config.lookup_treatment,
        config.postprocess,
    )


def _ticket(config: Config) -> TicketLookup:
    return TicketLookup(
        window_days=config.lookup_window_days or 60,
        maturity_cutoff_days=config.maturity_cutoff_days,
        weekdays=config.operating_weekdays,
    )


def _train_knn(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    config: Config,
    store: ArtifactStore,
    k: int | None,
    report: TrainingReport,
) -> None:
    chosen = k
    if chosen is None:
        chosen = select_k(frame, spec, config)
        report.k_selected_by_cv = chosen is not None
        if chosen is None:
            report.notes.append("history too short for k selection; used the configured default")
    report.k = chosen or config.default_k
    store.save("knn", KnnForecaster(spec, report.k, config.postprocess).fit(frame))


def _report(frame: pd.DataFrame, spec: FeatureSpec, kinds: list[str]) -> TrainingReport:
    dates = frame["date"]
    return TrainingReport(
        kinds=kinds,
        buckets=list(spec.buckets),
        rows=len(frame),
        days=int(dates.nunique()),
        first_day=pd.Timestamp(dates.min()).date().isoformat(),
        last_day=pd.Timestamp(dates.max()).date().isoformat(),
    )
