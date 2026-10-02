import numpy as np
import pandas as pd
import pytest

from call_pace.ablation import (
    compare_feature_sets,
    compare_lookup_variants,
    covariate_signal_report,
    feature_subset_search,
)
from call_pace.config import Config
from call_pace.contract import validate_hourly
from call_pace.errors import ConfigError, InsufficientDataError
from call_pace.features import FeatureSpec, build_frame
from call_pace.synthetic import SyntheticSpec, make_synthetic_hourly

CFG = Config(n_folds=3, fold_size_days=4, min_train_days=15, k_grid=(3, 5), default_k=3)


def level_frame(levels=(1, 5, 9), repeat=12, bucket="a", bump_last=0.0):
    """Days whose hours all share one level: volume so far reveals the rest of the day."""
    days = pd.bdate_range("2026-01-05", periods=len(levels) * repeat)
    rows = []
    for index, day in enumerate(days):
        level = levels[index % len(levels)] + (bump_last if index >= len(days) - 4 else 0.0)
        rows += [(bucket, day, h, float(level)) for h in range(8, 20)]
    data = pd.DataFrame(rows, columns=["bucket", "date", "hour", "count"])
    spec = FeatureSpec.from_config(CFG, [bucket])
    return build_frame(data, spec), spec


def test_the_informative_feature_set_beats_the_uninformative_one():
    frame, spec = level_frame()
    table = compare_feature_sets(
        frame,
        spec,
        CFG,
        {"hour only": ["hour"], "hour + volume so far": ["hour", "cumulative_lag"]},
    )
    means = table.set_index("feature_set")["mean"]
    assert set(table["feature_set"]) == {"lookup", "hour only", "hour + volume so far"}
    assert means["hour + volume so far"] < means["hour only"]
    assert {"fold_1", "fold_2", "fold_3"} <= set(table.columns)


def test_feature_sets_must_name_real_features():
    frame, spec = level_frame()
    with pytest.raises(ConfigError, match="oops"):
        compare_feature_sets(frame, spec, CFG, {"oops": ["hour", "weather"]})
    with pytest.raises(ConfigError):
        compare_feature_sets(frame, spec, CFG, {"empty": []})


def test_subset_search_ranks_on_early_folds_and_finds_the_informative_feature():
    frame, spec = level_frame()
    result = feature_subset_search(frame, spec, CFG)
    assert "cumulative_lag" in result.selected["features"]
    assert len(result.table) == 2 ** (len(spec.names) - 1) * 2  # subsets with hour x two k values
    assert result.table["features"].str.startswith("hour").all()
    ranks = result.table["cv_mae"].round(4)
    assert ranks.is_monotonic_increasing
    assert result.holdout_mae_selected <= result.holdout_mae_lookup
    assert "selected:" in result.summary() and "holdout MAE" in result.summary()


def test_the_final_fold_cannot_change_the_ranking():
    frame, spec = level_frame()
    baseline = feature_subset_search(frame, spec, CFG)
    distorted, _ = level_frame(bump_last=500.0)  # only the last four days change
    changed = feature_subset_search(distorted, spec, CFG)
    pd.testing.assert_frame_equal(baseline.table, changed.table)
    assert baseline.holdout_mae_selected != changed.holdout_mae_selected


def test_ties_prefer_fewer_features_then_a_smaller_k():
    frame, spec = level_frame(levels=(4,), repeat=36)  # constant series: every subset is perfect
    result = feature_subset_search(frame, spec, CFG)
    assert result.selected["features"] == "hour"
    assert result.selected["k"] == 3
    assert result.table["cv_mae"].eq(0.0).all()


def test_required_features_must_exist():
    frame, spec = level_frame()
    with pytest.raises(ConfigError):
        feature_subset_search(frame, spec, CFG, required=("hour", "weather"))


def test_keyed_lookup_beats_pooled_lookup_when_buckets_differ():
    low, spec_low = level_frame(levels=(1,), repeat=36, bucket="a")
    high, _ = level_frame(levels=(9,), repeat=36, bucket="b")
    both = pd.concat([low, high], ignore_index=True)
    table = compare_lookup_variants(both, CFG, treatments=("none",))
    best = table.iloc[0]
    assert best["key"] == "hour+bucket" and best["mean"] == 0.0
    pooled = table[table["key"] == "hour"].iloc[0]
    assert pooled["mean"] > 0.0
    assert table["mean"].is_monotonic_increasing
    assert len(table) == 2 * 2  # two keys x two statistics x one treatment


def test_lookup_variants_cover_the_whole_grid_by_default():
    frame, _ = level_frame()
    table = compare_lookup_variants(frame, CFG)
    assert len(table) == 2 * 2 * 4  # keys x statistics x treatments
    assert {"fold_1", "fold_2", "fold_3", "mean"} <= set(table.columns)


def covariate_frame(effect, seed):
    config = Config(
        covariates=("days_since_event",), n_folds=3, fold_size_days=4, min_train_days=15
    )
    raw = make_synthetic_hourly(
        n_days=120,
        seed=seed,
        config=config,
        spec=SyntheticSpec(covariate_effect=effect),
        with_covariate=True,
    )
    hourly = validate_hourly(raw, config)
    return build_frame(hourly, FeatureSpec.from_data(hourly, config))


def test_a_covariate_that_really_matters_shows_up_in_the_signal_report():
    report = covariate_signal_report(covariate_frame(effect=2.0, seed=1), "days_since_event")
    assert report.n == 240  # 120 days x 2 buckets: the unit is the bucket-day, not the hour
    assert report.pearson_p < 0.001 and report.kruskal_p < 0.001
    assert report.pearson_r < 0  # more days since the event means lower volume
    assert report.conclusion().startswith("some signal")


def test_a_covariate_with_no_effect_is_reported_as_no_signal():
    report = covariate_signal_report(covariate_frame(effect=0.0, seed=1), "days_since_event")
    assert report.conclusion().startswith("no detectable signal")


def test_signal_report_rejects_missing_or_constant_covariates():
    frame = covariate_frame(effect=0.0, seed=1)
    with pytest.raises(ConfigError):
        covariate_signal_report(frame, "weather")
    flat = frame.assign(days_since_event=3.0)
    with pytest.raises(InsufficientDataError):
        covariate_signal_report(flat, "days_since_event")
    assert np.isfinite(covariate_signal_report(frame, "days_since_event").kruskal_h)
