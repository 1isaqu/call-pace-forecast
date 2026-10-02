from datetime import date

import numpy as np
import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.errors import ConfigError, ContractError
from call_pace.features import (
    FeatureSpec,
    bucket_values,
    build_frame,
    complete_grid,
    feature_matrix,
    serving_frame,
)

# A four-hour day keeps every expected number checkable by hand.
CFG = Config(start_hour=8, end_hour=11, first_window_hours=2)


def hourly(*items, covariate=None):
    df = pd.DataFrame(items, columns=["bucket", "date", "hour", "count"])
    df["date"] = pd.to_datetime(df["date"])
    if covariate is not None:
        df["days_since_event"] = covariate
    return df


def spec_for(*buckets, config=CFG):
    return FeatureSpec.from_config(config, buckets)


def test_cumulative_lag_total_and_first_window_by_hand():
    data = hourly(("a", "2026-03-02", 8, 1), ("a", "2026-03-02", 9, 2), ("a", "2026-03-02", 11, 4))
    frame = build_frame(data, spec_for("a"))
    assert frame["hour"].tolist() == [8, 9, 10, 11]
    assert frame["count"].tolist() == [1, 2, 0, 4]  # hour 10 was missing and became zero
    assert frame["cumulative"].tolist() == [1, 3, 3, 7]
    assert frame["cumulative_lag"].tolist() == [0, 1, 3, 3]
    assert frame["day_total"].tolist() == [7, 7, 7, 7]
    assert frame["share"].round(4).tolist() == [0.1429, 0.4286, 0.4286, 1.0]
    # The first-window total (hours 8 and 9) only becomes visible from hour 10 on.
    assert frame["first_window_total"].tolist() == [0, 0, 3, 3]


def test_days_do_not_leak_into_each_other():
    data = hourly(
        ("a", "2026-03-02", 8, 5),
        ("a", "2026-03-02", 9, 5),
        ("a", "2026-03-03", 8, 1),
        ("a", "2026-03-03", 9, 1),
    )
    frame = build_frame(data, spec_for("a"))
    second = frame[frame["date"] == "2026-03-03"]
    assert second["cumulative_lag"].tolist() == [0, 1, 2, 2]
    assert second["first_window_total"].tolist() == [0, 0, 2, 2]


def test_zero_volume_day_has_undefined_share_not_a_division_error():
    data = hourly(("a", "2026-03-02", 8, 0))
    frame = build_frame(data, spec_for("a"))
    assert frame["share"].isna().all()
    assert (frame["cumulative_lag"] == 0).all()


def test_weekday_features_are_cyclic_over_the_schedule():
    monday = build_frame(hourly(("a", "2026-03-02", 8, 1)), spec_for("a"))
    friday = build_frame(hourly(("a", "2026-03-06", 8, 1)), spec_for("a"))
    assert monday["weekday_pos"].iloc[0] == 0 and friday["weekday_pos"].iloc[0] == 4
    assert monday["dow_sin"].iloc[0] == pytest.approx(0.0)
    assert monday["dow_cos"].iloc[0] == pytest.approx(1.0)
    assert friday["dow_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 4 / 5))


def test_closed_weekday_in_unvalidated_data_is_rejected():
    saturday = hourly(("a", "2026-03-07", 8, 1))
    with pytest.raises(ContractError, match="operating weekdays"):
        build_frame(saturday, spec_for("a"))


def test_bucket_columns_depend_on_how_many_buckets_exist():
    assert spec_for("a").bucket_columns == ()
    assert spec_for("a", "b").bucket_columns == ("bucket_code",)
    three = spec_for("a", "b", "c")
    assert three.bucket_columns == ("bucket_is_a", "bucket_is_b", "bucket_is_c")
    assert bucket_values(spec_for("a"), "a") == []
    assert bucket_values(spec_for("a", "b"), "b") == [1.0]
    assert bucket_values(three, "c") == [0.0, 0.0, 1.0]


def test_feature_names_have_a_stable_documented_order():
    spec = FeatureSpec.from_config(Config(covariates=("days_since_event",)), ["a", "b"])
    assert spec.names == (
        "hour",
        "dow_sin",
        "dow_cos",
        "cumulative_lag",
        "bucket_code",
        "first_window_total",
        "days_since_event",
    )


def test_unknown_bucket_is_rejected_everywhere():
    data = hourly(("z", "2026-03-02", 8, 1))
    with pytest.raises(ContractError, match="unknown buckets"):
        build_frame(data, spec_for("a", "b"))
    with pytest.raises(ContractError, match="unknown bucket"):
        bucket_values(spec_for("a", "b"), "z")


def test_fill_missing_days_adds_silent_operating_days_only():
    data = hourly(("a", "2026-03-02", 8, 1), ("a", "2026-03-06", 8, 1))  # Monday and Friday
    spec = spec_for("a")
    assert complete_grid(data, spec)["date"].nunique() == 2
    filled = complete_grid(data, spec, fill_missing_days=True)
    assert filled["date"].nunique() == 5  # Mon..Fri
    silent = filled[filled["date"] == "2026-03-04"]
    assert (silent["count"] == 0).all() and len(silent) == 4


def test_fill_missing_days_refuses_covariates():
    config = Config(start_hour=8, end_hour=11, covariates=("days_since_event",))
    data = hourly(("a", "2026-03-02", 8, 1), covariate=[2.0])
    spec = FeatureSpec.from_config(config, ["a"])
    with pytest.raises(ConfigError):
        build_frame(data, spec, fill_missing_days=True)


def test_covariates_are_carried_to_every_hour_of_the_day():
    config = Config(start_hour=8, end_hour=11, covariates=("days_since_event",))
    data = hourly(("a", "2026-03-02", 8, 1), covariate=[7.0])
    frame = build_frame(data, FeatureSpec.from_config(config, ["a"]))
    assert frame["days_since_event"].tolist() == [7.0] * 4


def test_feature_matrix_follows_requested_column_order(frame, spec):
    matrix = feature_matrix(frame, spec)
    assert matrix.shape == (len(frame), len(spec.names))
    swapped = feature_matrix(frame, ["cumulative_lag", "hour"])
    assert np.array_equal(swapped[:, 0], frame["cumulative_lag"].to_numpy(float))
    assert np.array_equal(swapped[:, 1], frame["hour"].to_numpy(float))


def test_spec_roundtrips_through_a_dict_and_rejects_garbage():
    spec = spec_for("a", "b")
    assert FeatureSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ConfigError):
        FeatureSpec.from_dict({"buckets": ["a"]})


def test_serving_features_match_training_features_for_the_same_moment(frame, spec):
    """Train/serve parity: replaying a past day hour by hour gives the training features."""
    one_day = frame[(frame["bucket"] == "auto")].iloc[:12]
    day = one_day["date"].iloc[0].date()
    observed = dict(zip(one_day["hour"], one_day["count"], strict=True))
    for hour in spec.hours:
        served = serving_frame(spec, "auto", day, observed, current_hour=hour)
        row_served = served[served["hour"] == hour].iloc[0]
        row_train = one_day[one_day["hour"] == hour].iloc[0]
        for name in spec.names:
            assert row_served[name] == pytest.approx(row_train[name]), (hour, name)


def test_serving_future_hours_use_the_cumulative_reached_so_far():
    spec = spec_for("a")
    day = date(2026, 3, 2)
    served = serving_frame(spec, "a", day, {8: 3, 9: 5, 10: 100}, current_hour=10)
    # Hour 10 is running, so its own partial count (100) is not visible to anyone.
    assert served["cumulative_lag"].tolist() == [0, 3, 8, 8]
    assert served["first_window_total"].tolist() == [0, 0, 8, 8]


def test_serving_with_no_observations_starts_from_zero():
    served = serving_frame(spec_for("a"), "a", date(2026, 3, 2), {}, current_hour=9)
    assert (served["cumulative_lag"] == 0).all()


def test_serving_rejects_closed_days_and_missing_covariates():
    spec = spec_for("a")
    with pytest.raises(ContractError, match="operating weekday"):
        serving_frame(spec, "a", date(2026, 3, 7), {}, current_hour=9)
    config = Config(start_hour=8, end_hour=11, covariates=("days_since_event",))
    cov_spec = FeatureSpec.from_config(config, ["a"])
    with pytest.raises(ContractError, match="covariate"):
        serving_frame(cov_spec, "a", date(2026, 3, 2), {}, current_hour=9)
    ok = serving_frame(cov_spec, "a", date(2026, 3, 2), {}, 9, {"days_since_event": 4})
    assert ok["days_since_event"].tolist() == [4.0] * 4
