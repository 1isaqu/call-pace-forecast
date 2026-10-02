"""Ablation studies: does a feature, an estimator or a covariate earn its place?

Business intuition about what drives volume is often wrong, and every feature added to a
distance-based model makes the neighbours less similar. These tools test claims with the same
walk-forward discipline used everywhere else:

  * `feature_subset_search`: try every subset of features, rank on the early folds, confirm
    once on the final fold (never use the final fold to rank).
  * `compare_feature_sets`: MAE of named feature sets, fold by fold.
  * `compare_lookup_variants`: estimator x outlier treatment x key for the baseline lookup.
  * `covariate_signal_report`: after controlling for hour and bucket, is there any relation
    left between an exogenous covariate and the residual?
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .baseline import LookupForecaster, keyed_lookup_predict
from .config import LOOKUP_STATS, LOOKUP_TREATMENTS, Config
from .errors import ConfigError, InsufficientDataError
from .features import FeatureSpec, feature_matrix
from .metrics import apply_postprocess, mae
from .validation import make_folds, split_frame


@dataclass
class _ScaledFold:
    """One walk-forward split with every feature already standardised on the training rows."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train: pd.DataFrame
    test: pd.DataFrame


def _scaled_folds(frame: pd.DataFrame, spec: FeatureSpec, config: Config) -> list[_ScaledFold]:
    folds = make_folds(
        frame["date"].unique(), config.n_folds, config.fold_size_days, config.min_train_days
    )
    prepared = []
    for fold in folds:
        train, test = split_frame(frame, fold)
        scaler = StandardScaler().fit(feature_matrix(train, spec))
        prepared.append(
            _ScaledFold(
                scaler.transform(feature_matrix(train, spec)),
                scaler.transform(feature_matrix(test, spec)),
                train["count"].to_numpy(float),
                test["count"].to_numpy(float),
                train,
                test,
            )
        )
    return prepared


def _knn_mae(fold: _ScaledFold, columns: Sequence[int], k: int, postprocess: str) -> float:
    """Standardisation is per column, so slicing pre-scaled columns equals scaling the subset."""
    model = KNeighborsRegressor(n_neighbors=min(k, len(fold.y_train)), metric="euclidean")
    model.fit(fold.x_train[:, columns], fold.y_train)
    predicted = apply_postprocess(model.predict(fold.x_test[:, columns]), postprocess)
    return mae(fold.y_test, predicted)


def compare_feature_sets(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    config: Config,
    feature_sets: Mapping[str, Sequence[str]],
    k: int | None = None,
) -> pd.DataFrame:
    """KNN MAE per walk-forward fold for each named feature set, plus the lookup baseline."""
    index = {name: i for i, name in enumerate(spec.names)}
    for label, names in feature_sets.items():
        unknown = [n for n in names if n not in index]
        if unknown or not names:
            raise ConfigError(f"feature set {label!r} must be a non-empty subset of {spec.names}")
    neighbours = k or config.default_k
    folds = _scaled_folds(frame, spec, config)
    rows = {"lookup": [_lookup_mae(config, fold) for fold in folds]}
    for label, names in feature_sets.items():
        columns = [index[n] for n in names]
        rows[label] = [_knn_mae(fold, columns, neighbours, config.postprocess) for fold in folds]
    table = pd.DataFrame(rows).T
    table.columns = [f"fold_{i + 1}" for i in range(len(folds))]
    table["mean"] = table.mean(axis=1)
    return table.reset_index(names="feature_set")


def _lookup_mae(config: Config, fold: _ScaledFold) -> float:
    model = LookupForecaster(
        config.lookup_window_days,
        config.maturity_cutoff_days,
        config.lookup_stat,
        config.lookup_treatment,
        config.postprocess,
    ).fit(fold.train)
    return mae(fold.y_test, model.predict(fold.test))


@dataclass
class AblationResult:
    """Ranked configurations and the single confirmation on the final fold."""

    table: pd.DataFrame
    selected: pd.Series
    holdout_mae_selected: float
    holdout_mae_all_features: float
    holdout_mae_lookup: float

    def summary(self) -> str:
        chosen = self.selected
        return (
            f"selected: {chosen['features']} "
            f"(k={int(chosen['k'])}, cv MAE {chosen['cv_mae']:.3f})\n"
            f"holdout MAE: selected {self.holdout_mae_selected:.3f}, "
            f"all features {self.holdout_mae_all_features:.3f}, "
            f"lookup baseline {self.holdout_mae_lookup:.3f}"
        )


def feature_subset_search(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    config: Config,
    k_grid: Sequence[int] | None = None,
    required: Sequence[str] = ("hour",),
) -> AblationResult:
    """Exhaustive search over feature subsets and k.

    Ranking: lowest mean MAE on folds 1..n-1, then fewer features, then smaller k. The final
    fold confirms the winner once. `required` features are in every subset (the hour is normally
    required because the whole point is a curve that varies by hour).
    """
    names = list(spec.names)
    missing = [r for r in required if r not in names]
    if missing:
        raise ConfigError(f"required features {missing} are not in {names}")
    folds = _scaled_folds(frame, spec, config)
    early, final = folds[:-1], folds[-1]
    optional = [n for n in names if n not in required]
    rows = []
    for size in range(len(optional) + 1):
        for extra in itertools.combinations(optional, size):
            chosen = [*required, *extra]
            columns = [names.index(n) for n in chosen]
            for k in sorted(set(k_grid or config.k_grid)):
                cv = float(np.mean([_knn_mae(f, columns, k, config.postprocess) for f in early]))
                rows.append(
                    {"features": ", ".join(chosen), "n_features": len(chosen), "k": k, "cv_mae": cv}
                )
    table = pd.DataFrame(rows)
    table = table.assign(_rank=table["cv_mae"].round(4)).sort_values(["_rank", "n_features", "k"])
    table = table.drop(columns="_rank").reset_index(drop=True)
    return _confirm(table, names, final, config)


def _confirm(
    table: pd.DataFrame, names: list[str], final: _ScaledFold, config: Config
) -> AblationResult:
    best = table.iloc[0]
    columns = [names.index(n.strip()) for n in best["features"].split(",")]
    everything = list(range(len(names)))
    return AblationResult(
        table=table,
        selected=best,
        holdout_mae_selected=_knn_mae(final, columns, int(best["k"]), config.postprocess),
        holdout_mae_all_features=_knn_mae(final, everything, int(best["k"]), config.postprocess),
        holdout_mae_lookup=_lookup_mae(config, final),
    )


def compare_lookup_variants(
    frame: pd.DataFrame,
    config: Config,
    estimators: Sequence[str] = LOOKUP_STATS,
    treatments: Sequence[str] = LOOKUP_TREATMENTS,
    keys: Sequence[Sequence[str]] = (("hour",), ("hour", "bucket")),
) -> pd.DataFrame:
    """MAE per fold of lookup variants: key x central statistic x outlier treatment.

    `("hour",)` pools every bucket; `("hour", "bucket")` separates them. The hour is always a
    partition key. Rows are sorted by mean MAE across folds.
    """
    folds = make_folds(
        frame["date"].unique(), config.n_folds, config.fold_size_days, config.min_train_days
    )
    splits = [split_frame(frame, f) for f in folds]
    rows = []
    for key, stat, treatment in itertools.product(keys, estimators, treatments):
        errors = [
            mae(
                test["count"],
                keyed_lookup_predict(train, test, key, stat, treatment, config.postprocess),
            )
            for train, test in splits
        ]
        row = {"key": "+".join(key), "stat": stat, "treatment": treatment}
        row.update({f"fold_{i + 1}": e for i, e in enumerate(errors)})
        row["mean"] = float(np.mean(errors))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


@dataclass(frozen=True)
class SignalReport:
    """Relation between a covariate and what the (bucket, hour, weekday) median cannot explain."""

    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    kruskal_h: float
    kruskal_p: float

    def conclusion(self, alpha: float = 0.05) -> str:
        weak = max(self.pearson_p, self.spearman_p, self.kruskal_p) >= alpha
        verdict = "no detectable signal" if weak else "some signal"
        return (
            f"{verdict} (alpha={alpha}): pearson p={self.pearson_p:.3f}, "
            f"kruskal p={self.kruskal_p:.3f}"
        )


def covariate_signal_report(frame: pd.DataFrame, covariate: str, groups: int = 3) -> SignalReport:
    """Test whether `covariate` explains anything beyond hour, bucket and weekday.

    Residual = count minus the (bucket, hour, weekday) median over the whole frame, which is
    descriptive, not predictive. The weekday is controlled because a covariate that cycles
    (days since the last event) is easily confounded with the weekly pattern.

    The unit of analysis is the bucket-day: the covariate is constant within a day and the
    hours of one day move together, so testing hourly rows would overstate the sample size.
    Tests: Pearson and Spearman correlation between the covariate and the day's mean residual,
    and a Kruskal-Wallis test of that residual across quantile groups of the covariate. Low
    p-values mean there is a relation worth investigating; high p-values make the covariate a
    candidate for removal. This is a screening tool, not proof of causality.
    """
    if covariate not in frame.columns:
        raise ConfigError(f"column {covariate!r} not found in the frame")
    days = _daily_residuals(frame, covariate)
    if days[covariate].nunique() < 3 or days["residual"].nunique() < 2:
        raise InsufficientDataError("the covariate or the residual is (almost) constant")
    pearson = stats.pearsonr(days["residual"], days[covariate])
    spearman = stats.spearmanr(days["residual"], days[covariate])
    bins = pd.qcut(days[covariate], q=groups, duplicates="drop")
    samples = [g["residual"].to_numpy() for _, g in days.groupby(bins, observed=True)]
    if len(samples) < 2:
        raise InsufficientDataError("the covariate has too few distinct values to form groups")
    h_statistic, h_p_value = _kruskal(samples)
    return SignalReport(
        len(days),
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
        h_statistic,
        h_p_value,
    )


def _daily_residuals(frame: pd.DataFrame, covariate: str) -> pd.DataFrame:
    controls = ["bucket", "hour", *(["weekday_pos"] if "weekday_pos" in frame.columns else [])]
    data = frame[[*controls, "date", "count", covariate]].dropna().copy()
    data["residual"] = data["count"] - data.groupby(controls)["count"].transform("median")
    return data.groupby(["bucket", "date"], as_index=False).agg(
        residual=("residual", "mean"), **{covariate: (covariate, "first")}
    )


def _kruskal(samples: list[np.ndarray]) -> tuple[float, float]:
    """Kruskal-Wallis H and p; identical samples mean no difference at all."""
    try:
        result = stats.kruskal(*samples)
    except ValueError:
        return 0.0, 1.0
    return float(result.statistic), float(result.pvalue)
