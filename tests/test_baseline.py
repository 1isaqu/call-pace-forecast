import json

import numpy as np
import pandas as pd
import pytest

from call_pace.baseline import LookupForecaster, central_estimate, keyed_lookup_predict
from call_pace.errors import ArtifactError, ConfigError, InsufficientDataError


def make_frame(bucket, per_day):
    """per_day maps an ISO date to {hour: count}. Only the listed hours exist."""
    rows = [
        (bucket, pd.Timestamp(day), hour, float(count))
        for day, hours in per_day.items()
        for hour, count in hours.items()
    ]
    return pd.DataFrame(rows, columns=["bucket", "date", "hour", "count"])


DAYS = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"]


def five_days(bucket="a", hour8=(1, 2, 3, 4, 100), hour9=(0, 0, 0, 0, 0)):
    return make_frame(bucket, {d: {8: a, 9: b} for d, a, b in zip(DAYS, hour8, hour9, strict=True)})


def test_median_ignores_an_outlier_that_would_drag_the_mean():
    values = [1, 2, 3, 4, 100]
    assert central_estimate(values, "median") == 3
    assert central_estimate(values, "mean") == pytest.approx(22.0)


def test_outlier_treatments_pull_the_mean_back():
    values = list(range(1, 20)) + [1000]
    plain = central_estimate(values, "mean", "none")
    assert central_estimate(values, "mean", "winsor") < plain
    assert central_estimate(values, "mean", "clip3sd") < plain
    assert central_estimate(values, "mean", "log1p") < plain


def test_log1p_of_a_constant_is_that_constant():
    assert central_estimate([4, 4, 4], "mean", "log1p") == pytest.approx(4.0)


def test_central_estimate_validates_input():
    with pytest.raises(InsufficientDataError):
        central_estimate([])
    with pytest.raises(ConfigError):
        central_estimate([1], stat="mode")
    with pytest.raises(ConfigError):
        central_estimate([1], treatment="trim")


def test_lookup_uses_the_median_per_bucket_and_hour():
    model = LookupForecaster(window_days=None).fit(five_days())
    assert model.table["a"] == {8: 3.0, 9: 0.0}
    assert model.expected_for_day("a", [8, 9]).tolist() == [3.0, 0.0]


def test_postprocess_rounds_up_as_the_dashboard_does():
    frame = five_days(hour8=(1, 1, 2, 2, 2), hour9=(1, 1, 1, 1, 1))
    frame.loc[frame["hour"] == 8, "count"] = [1.2, 1.2, 1.2, 1.2, 1.2]
    assert LookupForecaster(window_days=None, postprocess="ceil").fit(frame).expected_for_day(
        "a", [8]
    ).tolist() == [2.0]
    assert LookupForecaster(window_days=None, postprocess="none").fit(frame).expected_for_day(
        "a", [8]
    ).tolist() == [pytest.approx(1.2)]


def test_trailing_window_keeps_only_the_most_recent_days():
    model = LookupForecaster(window_days=2).fit(five_days(hour8=(50, 50, 50, 1, 3)))
    assert model.table["a"][8] == 2.0  # median of the last two days (1 and 3)
    assert model.trained["a"] == {"from": "2026-03-05", "to": "2026-03-06", "days": 2}


def test_maturity_cutoff_drops_recent_days_for_that_bucket_only():
    both = pd.concat(
        [five_days("a", hour8=(2, 2, 2, 2, 90)), five_days("b", hour8=(2, 2, 2, 2, 90))]
    )
    model = LookupForecaster(window_days=None, maturity_cutoff_days={"a": 1}).fit(both)
    assert model.table["a"][8] == 2.0
    assert model.table["b"][8] == 2.0  # median shrugs the outlier off anyway
    assert model.trained["a"]["to"] == "2026-03-05"
    assert model.trained["b"]["to"] == "2026-03-06"


def test_cutoff_that_removes_everything_is_an_error():
    with pytest.raises(InsufficientDataError, match="bucket 'a'"):
        LookupForecaster(maturity_cutoff_days={"a": 5}).fit(five_days())


def test_unseen_bucket_falls_back_to_the_pooled_hour_then_zero():
    model = LookupForecaster(window_days=None).fit(
        pd.concat([five_days("a", hour8=(2, 2, 2, 2, 2)), five_days("b", hour8=(6, 6, 6, 6, 6))])
    )
    assert model.predict_raw_one("new", 8) == 4.0  # pooled median of all rows at hour 8
    assert model.predict_raw_one("a", 3) == 0.0  # an hour never seen anywhere


def test_refitting_does_not_accumulate_old_buckets():
    model = LookupForecaster(window_days=None).fit(five_days("a"))
    model.fit(five_days("b"))
    assert set(model.table) == {"b"}


def test_informative_lookup_needs_enough_non_zero_hours():
    frame = make_frame("a", {d: {8: 3, 9: 0, 10: 0} for d in DAYS})
    model = LookupForecaster(window_days=None).fit(frame)
    assert model.nonzero_hours("a") == 1
    assert not model.is_informative("a", min_nonzero_hours=2)
    assert model.is_informative("a", min_nonzero_hours=1)
    assert model.nonzero_hours("missing") == 0


def test_predict_uses_the_frame_bucket_and_hour_columns():
    model = LookupForecaster(window_days=None).fit(
        pd.concat([five_days("a", hour8=(1,) * 5), five_days("b", hour8=(9,) * 5)])
    )
    probe = pd.DataFrame({"bucket": ["b", "a"], "hour": [8, 8]})
    assert model.predict(probe).tolist() == [9.0, 1.0]


def test_json_roundtrip_preserves_predictions(tmp_path):
    model = LookupForecaster(window_days=3, maturity_cutoff_days={"a": 1}, stat="mean").fit(
        five_days()
    )
    path = model.save(tmp_path / "nested" / "lookup.json")
    loaded = LookupForecaster.load(path)
    probe = pd.DataFrame({"bucket": ["a", "a", "zzz"], "hour": [8, 9, 8]})
    assert loaded.predict(probe).tolist() == model.predict(probe).tolist()
    assert (loaded.window_days, loaded.stat) == (3, "mean")
    assert loaded.maturity_cutoff_days == {"a": 1}


def test_loading_a_wrong_or_missing_file_raises_artifact_error(tmp_path):
    with pytest.raises(ArtifactError):
        LookupForecaster.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "knn"}), encoding="utf-8")
    with pytest.raises(ArtifactError):
        LookupForecaster.load(bad)
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError):
        LookupForecaster.load(bad)


def test_keyed_lookup_separates_buckets_only_when_keyed_by_bucket():
    train = pd.concat([five_days("a", hour8=(1,) * 5), five_days("b", hour8=(9,) * 5)])
    test = pd.DataFrame({"bucket": ["a", "b"], "hour": [8, 8]})
    pooled = keyed_lookup_predict(train, test, ["hour"])
    keyed = keyed_lookup_predict(train, test, ["hour", "bucket"])
    assert pooled.tolist() == [5.0, 5.0]
    assert keyed.tolist() == [1.0, 9.0]


def test_keyed_lookup_falls_back_to_hour_for_a_cell_it_never_saw():
    train = five_days("a", hour8=(4,) * 5)
    test = pd.DataFrame({"bucket": ["new"], "hour": [8]})
    assert keyed_lookup_predict(train, test, ["hour", "bucket"]).tolist() == [4.0]
    assert np.isfinite(
        keyed_lookup_predict(
            train, pd.DataFrame({"bucket": ["a"], "hour": [15]}), ["hour", "bucket"]
        )
    ).all()
