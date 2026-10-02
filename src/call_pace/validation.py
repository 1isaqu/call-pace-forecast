"""Walk-forward validation and the rules for promoting a model.

Time series must never be shuffled: a model trained on Thursday must not be tested on
Tuesday. The protocol is an expanding window. The last `n_folds * fold_size` days are cut into
consecutive test blocks; each block is predicted by a model fitted only on the days before it.

  fold 1: train [0, A)      test [A, A+s)
  fold 2: train [0, A+s)    test [A+s, A+2s)
  ...
  fold n: train [0, ...)    test the final s days      <- untouched holdout

Hyperparameters (k) are selected on folds 1..n-1 only. The last fold is looked at once, to
confirm the choice, and is never used to pick anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .baseline import LookupForecaster
from .config import Config
from .errors import ConfigError, InsufficientDataError
from .features import FeatureSpec
from .knn import KnnCV, KnnForecaster, knn_cross_validate
from .metrics import directional_accuracy, interval_coverage, mae, mean_interval_width
from .quantile import QuantileForecaster

CHALLENGERS = ("knn", "quantile")


@dataclass(frozen=True)
class Fold:
    """One walk-forward split, described by the dates on each side."""

    index: int
    train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


def make_folds(
    dates: Sequence[pd.Timestamp], n_folds: int, fold_size: int, min_train_days: int
) -> list[Fold]:
    """Cut the last n_folds * fold_size dates into test blocks with expanding training sets."""
    ordered = sorted(pd.Timestamp(d) for d in set(dates))
    needed = n_folds * fold_size + min_train_days
    if len(ordered) < needed:
        raise InsufficientDataError(
            f"{n_folds} folds of {fold_size} days plus {min_train_days} training days need "
            f"{needed} operating days, found {len(ordered)}"
        )
    first_test = len(ordered) - n_folds * fold_size
    folds = []
    for i in range(n_folds):
        start = first_test + i * fold_size
        folds.append(Fold(i + 1, tuple(ordered[:start]), tuple(ordered[start : start + fold_size])))
    return folds


def split_frame(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and test rows of `frame` for one fold."""
    train = frame[frame["date"].isin(fold.train_dates)].reset_index(drop=True)
    test = frame[frame["date"].isin(fold.test_dates)].reset_index(drop=True)
    return train, test


@dataclass
class WalkForwardResult:
    """Everything a walk-forward run produced."""

    folds: list[Fold]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    knn_cv: KnnCV | None
    config: Config

    @property
    def best_k(self) -> int | None:
        return None if self.knn_cv is None else self.knn_cv.best_k

    def holdout(self) -> pd.Series:
        """Metrics of the final fold, the one that was never used for selection."""
        return self.fold_metrics.iloc[-1]

    def mean(self) -> pd.Series:
        """Mean of each metric across folds."""
        return self.fold_metrics.drop(columns=["fold"]).mean()

    def by(self, column: str, holdout_only: bool = True) -> pd.DataFrame:
        """MAE per model grouped by `column` (for example hour or bucket)."""
        data = self.predictions
        if holdout_only:
            data = data[data["fold"] == data["fold"].max()]
        models = [m for m in ("lookup", *CHALLENGERS) if m in data.columns]
        rows = []
        for key, group in data.groupby(column):
            rows.append({column: key, "n": len(group)})
            rows[-1].update({f"mae_{m}": mae(group["actual"], group[m]) for m in models})
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        parts = ["Fold metrics", markdown_table(self.fold_metrics)]
        if self.knn_cv is not None:
            parts += [
                f"\nK selection on folds 1..{len(self.folds) - 1}",
                markdown_table(self.knn_cv.table),
            ]
            parts.append(f"\nSelected k: {self.knn_cv.best_k}")
        return "\n".join(parts)


def run_walk_forward(
    frame: pd.DataFrame,
    config: Config,
    spec: FeatureSpec,
    challengers: Sequence[str] = CHALLENGERS,
    k_grid: Sequence[int] | None = None,
) -> WalkForwardResult:
    """Evaluate the lookup baseline and the chosen challengers on every walk-forward fold."""
    unknown = set(challengers) - set(CHALLENGERS)
    if unknown:
        raise ConfigError(f"unknown challengers {sorted(unknown)}; choose from {CHALLENGERS}")
    folds = make_folds(
        frame["date"].unique(), config.n_folds, config.fold_size_days, config.min_train_days
    )
    splits = [split_frame(frame, fold) for fold in folds]
    preds = [_lookup_block(config, train, test) for train, test in splits]
    cv = None
    if "knn" in challengers:
        cv = knn_cross_validate(splits[:-1], spec, k_grid or config.k_grid, config.postprocess)
        _add_knn(preds, splits, cv, spec, config)
    if "quantile" in challengers:
        _add_quantile(preds, splits, spec, config)
    return _assemble(folds, preds, cv, config)


def _lookup_block(config: Config, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    model = LookupForecaster(
        config.lookup_window_days,
        config.maturity_cutoff_days,
        config.lookup_stat,
        config.lookup_treatment,
        config.postprocess,
    ).fit(train)
    block = test[["bucket", "date", "hour"]].copy()
    block["actual"] = test["count"].to_numpy()
    block["lookup"] = model.predict(test)
    return block


def _add_knn(
    preds: list[pd.DataFrame],
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    cv: KnnCV,
    spec: FeatureSpec,
    config: Config,
) -> None:
    """Early folds reuse the cross-validation predictions; the holdout fold is fitted fresh."""
    for block, cached in zip(preds[:-1], cv.predictions[cv.best_k], strict=True):
        block["knn"] = cached
    train, test = splits[-1]
    final = KnnForecaster(spec, cv.best_k, config.postprocess).fit(train)
    preds[-1]["knn"] = final.predict(test)


def _add_quantile(
    preds: list[pd.DataFrame],
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    spec: FeatureSpec,
    config: Config,
) -> None:
    for block, (train, test) in zip(preds, splits, strict=True):
        model = QuantileForecaster(
            spec, config.quantiles, config.lgbm_params, config.postprocess
        ).fit(train)
        bands = model.predict_bands(test)
        block["quantile"] = bands["center"].to_numpy()
        block["lower"] = bands["lower"].to_numpy()
        block["upper"] = bands["upper"].to_numpy()


def _assemble(
    folds: list[Fold], preds: list[pd.DataFrame], cv: KnnCV | None, config: Config
) -> WalkForwardResult:
    for fold, block in zip(folds, preds, strict=True):
        block.insert(0, "fold", fold.index)
    predictions = pd.concat(preds, ignore_index=True)
    rows = [_fold_row(fold.index, block) for fold, block in zip(folds, preds, strict=True)]
    return WalkForwardResult(folds, predictions, pd.DataFrame(rows), cv, config)


def _fold_row(index: int, block: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {"fold": index, "n": len(block)}
    row["mae_lookup"] = mae(block["actual"], block["lookup"])
    for model in CHALLENGERS:
        if model in block.columns:
            row[f"mae_{model}"] = mae(block["actual"], block[model])
            row[f"dir_{model}"] = directional_accuracy(
                block["actual"], block[model], block["lookup"]
            )
    if "lower" in block.columns:
        row["coverage"] = interval_coverage(block["actual"], block["lower"], block["upper"])
        row["width"] = mean_interval_width(block["lower"], block["upper"])
    return row


@dataclass(frozen=True)
class PromotionCriteria:
    """When a challenger may replace the baseline. Defaults are invented: calibrate them.

    Thresholds relative to the baseline scale across series of different volume; an absolute
    MAE cap does not, so it is optional and off by default.
    """

    min_improvement_pct: float = 5.0
    min_directional_accuracy: float = 0.55
    fold_tolerance_abs: float = 0.05
    fold_tolerance_rel: float = 0.05
    max_mae: float | None = None


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    challenger: str
    checks: tuple[Check, ...]

    @property
    def promote(self) -> bool:
        return all(check.passed for check in self.checks)

    def explain(self) -> str:
        verdict = "PROMOTE" if self.promote else "KEEP BASELINE"
        lines = [f"{verdict}: {self.challenger}"]
        lines += [f"  [{'ok' if c.passed else 'fail'}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)


def decide_promotion(
    result: WalkForwardResult,
    challenger: str = "knn",
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    """Apply the promotion rules to a walk-forward result.

    A single fold is allowed to be slightly worse than the baseline when the gap is inside the
    noise: a fold only counts as a regression if it is worse by more than the absolute
    tolerance AND by more than the relative tolerance. Small folds have a small MAE, so a
    purely relative rule would fail on a harmless absolute difference.
    """
    rules = criteria or PromotionCriteria()
    metrics = result.fold_metrics
    if f"mae_{challenger}" not in metrics.columns:
        raise ConfigError(f"{challenger!r} was not evaluated in this result")
    holdout = result.holdout()
    checks = (
        _improvement_check(holdout, challenger, rules),
        _direction_check(holdout, challenger, rules),
        _fold_regression_check(metrics, challenger, rules),
        _cap_check(holdout, challenger, rules),
    )
    return PromotionDecision(challenger, checks)


def _improvement_check(holdout: pd.Series, model: str, rules: PromotionCriteria) -> Check:
    base, challenger = holdout["mae_lookup"], holdout[f"mae_{model}"]
    gain = 0.0 if base == 0 else (base - challenger) / base * 100.0
    detail = f"holdout MAE {challenger:.3f} vs baseline {base:.3f} ({gain:+.1f}%)"
    return Check(
        "improvement",
        gain >= rules.min_improvement_pct,
        f"{detail}, need >= {rules.min_improvement_pct}%",
    )


def _direction_check(holdout: pd.Series, model: str, rules: PromotionCriteria) -> Check:
    value = holdout[f"dir_{model}"]
    passed = bool(np.isfinite(value)) and value >= rules.min_directional_accuracy
    shown = "undefined" if not np.isfinite(value) else f"{value:.1%}"
    return Check(
        "directional_accuracy", passed, f"{shown}, need >= {rules.min_directional_accuracy:.0%}"
    )


def _fold_regression_check(metrics: pd.DataFrame, model: str, rules: PromotionCriteria) -> Check:
    gap = metrics[f"mae_{model}"] - metrics["mae_lookup"]
    relative = gap / metrics["mae_lookup"].where(metrics["mae_lookup"] > 0)
    bad = metrics.loc[
        (gap > rules.fold_tolerance_abs) & (relative.fillna(np.inf) > rules.fold_tolerance_rel),
        "fold",
    ]
    detail = (
        "no fold clearly worse than the baseline"
        if bad.empty
        else f"worse on folds {bad.astype(int).tolist()}"
    )
    return Check("no_fold_regression", bad.empty, detail)


def _cap_check(holdout: pd.Series, model: str, rules: PromotionCriteria) -> Check:
    if rules.max_mae is None:
        return Check("max_mae", True, "no absolute cap configured")
    value = holdout[f"mae_{model}"]
    return Check("max_mae", value <= rules.max_mae, f"holdout MAE {value:.3f}, cap {rules.max_mae}")


@dataclass(frozen=True)
class PairedTest:
    mean_difference: float
    t_statistic: float
    p_value: float


def paired_error_test(errors_a: Sequence[float], errors_b: Sequence[float]) -> PairedTest:
    """Paired t-test on per-observation absolute errors of two models on the same rows.

    A negative `mean_difference` means model A has the smaller error. Hourly errors are
    autocorrelated within a day, so treat the p-value as a rough guide, not a verdict.
    """
    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    if a.shape != b.shape or a.size < 2:
        raise InsufficientDataError("a paired test needs two equal-length samples of size >= 2")
    if np.allclose(a, b):
        return PairedTest(0.0, 0.0, 1.0)
    result = stats.ttest_rel(a, b)
    return PairedTest(float(np.mean(a - b)), float(result.statistic), float(result.pvalue))


def compare_models(result: WalkForwardResult, model_a: str, model_b: str) -> PairedTest:
    """Paired test of absolute errors of two evaluated models over all folds."""
    data = result.predictions
    for name in (model_a, model_b):
        if name not in data.columns:
            raise ConfigError(f"{name!r} was not evaluated in this result")
    return paired_error_test(
        (data["actual"] - data[model_a]).abs(), (data["actual"] - data[model_b]).abs()
    )


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra dependency)."""

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
