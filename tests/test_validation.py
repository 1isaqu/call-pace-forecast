import numpy as np
import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.errors import ConfigError, InsufficientDataError
from call_pace.metrics import mae
from call_pace.validation import (
    PromotionCriteria,
    WalkForwardResult,
    compare_models,
    decide_promotion,
    make_folds,
    markdown_table,
    paired_error_test,
    run_walk_forward,
)

DAYS = pd.bdate_range("2026-01-05", periods=40)


def test_folds_are_consecutive_expanding_and_end_at_the_last_day():
    folds = make_folds(DAYS, n_folds=3, fold_size=5, min_train_days=10)
    assert [f.index for f in folds] == [1, 2, 3]
    assert folds[-1].test_dates[-1] == DAYS[-1]
    assert all(len(f.test_dates) == 5 for f in folds)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_dates[-1] < later.test_dates[0]
        assert (
            later.train_dates[-5:] == earlier.test_dates
        )  # the previous test block is now history
        assert len(later.train_dates) == len(earlier.train_dates) + 5


def test_no_fold_trains_on_the_future():
    for fold in make_folds(DAYS, 4, 4, 10):
        assert max(fold.train_dates) < min(fold.test_dates)
        assert not set(fold.train_dates) & set(fold.test_dates)


def test_unsorted_or_duplicated_dates_are_normalised():
    shuffled = list(DAYS[::-1]) + list(DAYS[:3])
    assert make_folds(shuffled, 3, 5, 10) == make_folds(DAYS, 3, 5, 10)


def test_too_little_history_says_how_much_is_needed():
    with pytest.raises(InsufficientDataError, match="need 25 operating days, found 20"):
        make_folds(DAYS[:20], n_folds=3, fold_size=5, min_train_days=10)


@pytest.fixture(scope="module")
def result(frame, spec, config):
    return run_walk_forward(frame, config, spec)


def test_result_covers_every_test_row_once_with_all_models(result, config):
    preds = result.predictions
    assert len(preds) == config.n_folds * config.fold_size_days * 2 * 12  # days x buckets x hours
    assert {"actual", "lookup", "knn", "quantile", "lower", "upper"} <= set(preds.columns)
    assert not preds.duplicated(["fold", "bucket", "date", "hour"]).any()
    assert preds[["lookup", "knn", "quantile"]].notna().all().all()
    assert (preds["lower"] <= preds["upper"]).all()


def test_each_folds_test_days_follow_its_training_days(result):
    for fold in result.folds:
        days = pd.to_datetime(
            result.predictions.loc[result.predictions["fold"] == fold.index, "date"]
        )
        assert set(days.unique()) == set(fold.test_dates)
        assert days.min() > max(fold.train_dates)


def test_k_is_selected_without_looking_at_the_holdout_fold(result, config):
    cv_columns = {c for c in result.knn_cv.table.columns if c.startswith("fold_")}
    assert cv_columns == {f"fold_{i}" for i in range(1, config.n_folds)}
    assert result.best_k in config.k_grid


def test_fold_metrics_match_a_recomputation_from_the_predictions(result):
    preds = result.predictions
    for _, row in result.fold_metrics.iterrows():
        block = preds[preds["fold"] == row["fold"]]
        assert row["mae_lookup"] == pytest.approx(mae(block["actual"], block["lookup"]))
        assert row["mae_knn"] == pytest.approx(mae(block["actual"], block["knn"]))
        assert 0.0 <= row["coverage"] <= 1.0 and row["width"] >= 0.0
    assert result.holdout()["fold"] == result.fold_metrics["fold"].max()


def test_breakdown_by_hour_and_bucket(result):
    by_hour = result.by("hour")
    assert len(by_hour) == 12 and {"mae_lookup", "mae_knn", "mae_quantile"} <= set(by_hour.columns)
    by_bucket = result.by("bucket", holdout_only=False)
    assert set(by_bucket["bucket"]) == {"auto", "consumer"}
    assert by_bucket["n"].sum() == len(result.predictions)


def test_a_run_is_reproducible(frame, spec, config, result):
    again = run_walk_forward(frame, config, spec)
    pd.testing.assert_frame_equal(again.predictions, result.predictions)


def test_only_the_requested_challengers_are_evaluated(frame, spec, config):
    lookup_only = run_walk_forward(frame, config, spec, challengers=())
    assert "knn" not in lookup_only.predictions.columns
    assert lookup_only.knn_cv is None and lookup_only.best_k is None
    with pytest.raises(ConfigError):
        run_walk_forward(frame, config, spec, challengers=("arima",))


def test_markdown_report_lists_folds_and_selected_k(result):
    text = result.to_markdown()
    assert "Fold metrics" in text and "Selected k:" in text and "| fold |" in text


# --- promotion rules, driven by hand-made fold metrics -----------------------------------


def crafted(rows):
    metrics = pd.DataFrame(rows, columns=["fold", "mae_lookup", "mae_knn", "dir_knn"])
    return WalkForwardResult([], pd.DataFrame(), metrics, None, Config())


GOOD = [(1, 1.0, 0.9, 0.7), (2, 1.0, 0.9, 0.7), (3, 2.0, 1.5, 0.7)]


def test_a_clearly_better_model_is_promoted():
    decision = decide_promotion(crafted(GOOD))
    assert decision.promote and all(c.passed for c in decision.checks)
    assert decision.explain().startswith("PROMOTE: knn")


def test_improvement_must_reach_the_configured_percentage():
    rows = [(1, 1.0, 1.0, 0.7), (2, 1.0, 1.0, 0.7), (3, 2.0, 1.95, 0.7)]
    decision = decide_promotion(crafted(rows))
    assert not decision.promote
    assert [c.name for c in decision.checks if not c.passed] == ["improvement"]
    assert decide_promotion(
        crafted(rows), criteria=PromotionCriteria(min_improvement_pct=2.0)
    ).promote


def test_direction_must_be_right_often_enough_and_defined():
    low = [(1, 1.0, 0.9, 0.7), (2, 1.0, 0.9, 0.7), (3, 2.0, 1.5, 0.4)]
    assert [c.name for c in decide_promotion(crafted(low)).checks if not c.passed] == [
        "directional_accuracy"
    ]
    undefined = [(1, 1.0, 0.9, 0.7), (2, 1.0, 0.9, 0.7), (3, 2.0, 1.5, np.nan)]
    failed = [c for c in decide_promotion(crafted(undefined)).checks if not c.passed]
    assert failed[0].name == "directional_accuracy" and "undefined" in failed[0].detail


def test_a_slightly_worse_fold_is_noise_but_a_clearly_worse_one_is_not():
    # Worse by 0.04 absolute: inside the absolute tolerance, so tolerated.
    tolerated = [(1, 1.00, 1.04, 0.7), (2, 1.0, 0.8, 0.7), (3, 2.0, 1.5, 0.7)]
    assert decide_promotion(crafted(tolerated)).promote
    # Worse by 0.5 absolute and 50 percent relative: a real regression.
    clear = [(1, 1.0, 1.5, 0.7), (2, 1.0, 0.8, 0.7), (3, 2.0, 1.5, 0.7)]
    decision = decide_promotion(crafted(clear))
    failed = [c for c in decision.checks if not c.passed]
    assert failed[0].name == "no_fold_regression" and "[1]" in failed[0].detail


def test_both_tolerances_must_be_exceeded_to_count_as_a_regression():
    # Large relative gap on a tiny absolute scale: 0.02 -> 0.04 is +100 percent but only 0.02.
    tiny = [(1, 0.02, 0.04, 0.7), (2, 1.0, 0.8, 0.7), (3, 2.0, 1.5, 0.7)]
    assert decide_promotion(crafted(tiny)).promote
    # Large absolute gap on a huge scale: +0.1 on 10.0 is only 1 percent.
    huge = [(1, 10.0, 10.1, 0.7), (2, 1.0, 0.8, 0.7), (3, 2.0, 1.5, 0.7)]
    assert decide_promotion(crafted(huge)).promote


def test_a_perfect_baseline_cannot_be_beaten():
    rows = [(1, 0.0, 0.0, 0.7), (2, 0.0, 0.0, 0.7), (3, 0.0, 0.0, 0.7)]
    assert not decide_promotion(crafted(rows)).promote


def test_absolute_cap_is_optional_and_enforced_when_set():
    assert decide_promotion(crafted(GOOD)).checks[-1].detail == "no absolute cap configured"
    capped = decide_promotion(crafted(GOOD), criteria=PromotionCriteria(max_mae=1.0))
    assert not capped.promote and capped.checks[-1].name == "max_mae"


def test_promotion_needs_the_challenger_to_have_been_evaluated():
    with pytest.raises(ConfigError):
        decide_promotion(crafted(GOOD), challenger="quantile")


# --- paired comparison ---------------------------------------------------------------------


def test_paired_test_detects_a_consistently_smaller_error():
    rng = np.random.default_rng(0)
    worse = rng.uniform(1, 2, 200)
    better = worse - rng.uniform(0.2, 0.4, 200)
    outcome = paired_error_test(better, worse)
    assert outcome.mean_difference < 0 and outcome.t_statistic < 0 and outcome.p_value < 0.001


def test_paired_test_on_identical_samples_is_a_clean_no_difference():
    same = [1.0, 2.0, 3.0]
    outcome = paired_error_test(same, same)
    assert (outcome.mean_difference, outcome.p_value) == (0.0, 1.0)


def test_paired_test_rejects_bad_shapes():
    with pytest.raises(InsufficientDataError):
        paired_error_test([1.0], [1.0])
    with pytest.raises(InsufficientDataError):
        paired_error_test([1.0, 2.0], [1.0])


def test_compare_models_uses_absolute_errors_over_all_folds(result):
    outcome = compare_models(result, "knn", "lookup")
    assert np.isfinite(outcome.p_value)
    with pytest.raises(ConfigError):
        compare_models(result, "knn", "missing")


def test_markdown_table_renders_floats_nans_and_ints():
    df = pd.DataFrame({"fold": [1, 2], "mae": [0.12345, np.nan]})
    assert markdown_table(df, digits=2).splitlines() == [
        "| fold | mae |",
        "|---|---|",
        "| 1 | 0.12 |",
        "| 2 |  |",
    ]
