import numpy as np
import pandas as pd
import pytest

from call_pace.baseline import LookupForecaster
from call_pace.config import Config
from call_pace.errors import ArtifactError, ConfigError, InsufficientDataError
from call_pace.features import FeatureSpec, build_frame
from call_pace.knn import KnnForecaster, knn_cross_validate
from call_pace.metrics import mae


def level_days(levels, repeat):
    """Days where every hour of a day has the same count, so volume so far reveals the day."""
    rows = []
    start = pd.Timestamp("2026-01-05")
    day = 0
    for _ in range(repeat):
        for level in levels:
            while (start + pd.Timedelta(days=day)).weekday() >= 5:
                day += 1
            date = start + pd.Timedelta(days=day)
            rows += [("a", date, h, float(level)) for h in range(8, 20)]
            day += 1
    return pd.DataFrame(rows, columns=["bucket", "date", "hour", "count"])


def test_prediction_is_a_non_negative_whole_number_per_row(frame, spec):
    model = KnnForecaster(spec, k=5).fit(frame)
    pred = model.predict(frame)
    assert pred.shape == (len(frame),)
    assert (pred >= 0).all() and np.array_equal(pred, np.ceil(pred))


def test_knn_reacts_to_the_day_so_far_where_a_lookup_cannot():
    cfg = Config()
    data = level_days([1, 5, 9], repeat=12)
    spec = FeatureSpec.from_config(cfg, ["a"])
    frame = build_frame(data, spec)
    dates = sorted(frame["date"].unique())
    train = frame[frame["date"].isin(dates[:30])]
    test = frame[frame["date"].isin(dates[30:]) & (frame["hour"] >= 10)]
    knn = KnnForecaster(spec, k=3, features=("hour", "cumulative_lag")).fit(train)
    lookup = LookupForecaster(window_days=None).fit(train)
    assert mae(test["count"], knn.predict(test)) == 0.0
    assert mae(test["count"], lookup.predict(test)) > 1.0


def test_k_is_capped_at_the_number_of_training_rows(spec):
    tiny = build_frame(
        pd.DataFrame(
            {"bucket": ["auto"], "date": [pd.Timestamp("2026-03-02")], "hour": [8], "count": [2.0]}
        ),
        spec,
    ).iloc[:3]
    model = KnnForecaster(spec, k=50).fit(tiny)
    assert model.pipeline.steps[-1][1].n_neighbors == 3


def test_feature_subset_is_validated_and_used(frame, spec):
    with pytest.raises(ConfigError):
        KnnForecaster(spec, features=("hour", "not_a_feature"))
    with pytest.raises(ConfigError):
        KnnForecaster(spec, features=())
    model = KnnForecaster(spec, features=("hour",)).fit(frame)
    assert model.pipeline.steps[0][1].n_features_in_ == 1


def test_unfitted_and_empty_fits_fail_loudly(frame, spec, tmp_path):
    model = KnnForecaster(spec)
    with pytest.raises(ArtifactError):
        model.predict(frame)
    with pytest.raises(ArtifactError):
        model.save(tmp_path / "k.joblib")
    with pytest.raises(InsufficientDataError):
        model.fit(frame.iloc[:0])


def test_save_and_load_give_identical_predictions(frame, spec, tmp_path):
    model = KnnForecaster(spec, k=7, postprocess="round").fit(frame)
    path = model.save(tmp_path / "artifacts" / "knn.joblib")
    loaded = KnnForecaster.load(path)
    assert (loaded.k, loaded.postprocess, loaded.spec) == (7, "round", spec)
    assert np.array_equal(loaded.predict(frame), model.predict(frame))


def test_loading_the_wrong_kind_of_file_is_an_artifact_error(tmp_path):
    with pytest.raises(ArtifactError):
        KnnForecaster.load(tmp_path / "missing.joblib")
    import joblib

    joblib.dump({"model": "something-else"}, tmp_path / "other.joblib")
    with pytest.raises(ArtifactError):
        KnnForecaster.load(tmp_path / "other.joblib")
    joblib.dump({"model": "knn", "format_version": 999}, tmp_path / "old.joblib")
    with pytest.raises(ArtifactError, match="version"):
        KnnForecaster.load(tmp_path / "old.joblib")


def test_cross_validation_reports_every_k_on_every_split_and_picks_the_lowest(frame, spec):
    dates = sorted(frame["date"].unique())
    splits = [
        (frame[frame["date"].isin(dates[:n])], frame[frame["date"].isin(dates[n : n + 5])])
        for n in (30, 40, 50)
    ]
    cv = knn_cross_validate(splits, spec, k_grid=(3, 5, 9))
    assert list(cv.table["k"]) == [3, 5, 9]
    assert {"fold_1", "fold_2", "fold_3", "mean_mae"} <= set(cv.table.columns)
    best_row = cv.table.loc[cv.table["mean_mae"].idxmin()]
    assert cv.best_k == int(best_row["k"])
    assert len(cv.predictions[3]) == 3 and cv.predictions[3][0].shape[0] == len(splits[0][1])


def test_ties_between_k_values_go_to_the_smaller_k(spec):
    constant = pd.DataFrame(
        {
            "bucket": "auto",
            "date": pd.Timestamp("2026-03-02"),
            "hour": list(range(8, 20)),
            "count": 3.0,
        }
    )
    frame = build_frame(constant, spec)
    cv = knn_cross_validate([(frame, frame)], spec, k_grid=(9, 2, 5))
    assert cv.table["mean_mae"].eq(0.0).all()
    assert cv.best_k == 2
