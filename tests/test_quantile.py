import numpy as np
import pandas as pd
import pytest

from call_pace.config import DEFAULT_LGBM_PARAMS
from call_pace.errors import ArtifactError, ConfigError, InsufficientDataError
from call_pace.features import build_frame
from call_pace.metrics import interval_coverage, mean_interval_width
from call_pace.quantile import QuantileForecaster

FAST = {**DEFAULT_LGBM_PARAMS, "n_estimators": 40, "min_child_samples": 5}


def test_bands_are_ordered_non_negative_whole_numbers(frame, spec):
    model = QuantileForecaster(spec, params=FAST).fit(frame)
    bands = model.predict_bands(frame)
    assert list(bands.columns) == ["center", "lower", "upper"]
    assert (bands["lower"] <= bands["upper"]).all()
    assert (bands.to_numpy() >= 0).all()
    assert np.array_equal(bands.to_numpy(), np.ceil(bands.to_numpy()))
    assert np.array_equal(model.predict(frame), bands["center"].to_numpy())


def test_a_wider_quantile_pair_gives_a_wider_band(frame, spec):
    narrow = QuantileForecaster(spec, (0.45, 0.55), FAST, postprocess="none").fit(frame)
    wide = QuantileForecaster(spec, (0.05, 0.95), FAST, postprocess="none").fit(frame)
    b_narrow, b_wide = narrow.predict_bands(frame), wide.predict_bands(frame)
    assert mean_interval_width(b_wide["lower"], b_wide["upper"]) > mean_interval_width(
        b_narrow["lower"], b_narrow["upper"]
    )
    assert interval_coverage(frame["count"], b_wide["lower"], b_wide["upper"]) > interval_coverage(
        frame["count"], b_narrow["lower"], b_narrow["upper"]
    )


def test_crossed_quantile_models_are_reordered(frame, spec):
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(len(x), self.value)

    model = QuantileForecaster(spec, params=FAST).fit(frame)
    model.lower, model.upper, model.center = Constant(9.0), Constant(2.0), Constant(5.0)
    bands = model.predict_bands(frame.iloc[:4])
    assert bands["lower"].tolist() == [2.0] * 4
    assert bands["upper"].tolist() == [9.0] * 4
    assert bands["center"].tolist() == [5.0] * 4


def test_invalid_inputs_are_rejected(frame, spec):
    with pytest.raises(ConfigError):
        QuantileForecaster(spec, quantiles=(0.9, 0.1))
    with pytest.raises(InsufficientDataError):
        QuantileForecaster(spec, params=FAST).fit(frame.iloc[:0])
    negative = frame.copy()
    negative.loc[negative.index[0], "count"] = -1.0
    with pytest.raises(ConfigError, match="non-negative"):
        QuantileForecaster(spec, params=FAST).fit(negative)


def test_unfitted_model_cannot_predict_or_be_saved(frame, spec, tmp_path):
    model = QuantileForecaster(spec, params=FAST)
    with pytest.raises(ArtifactError):
        model.predict_bands(frame)
    with pytest.raises(ArtifactError):
        model.save(tmp_path / "q.joblib")


def test_save_and_load_give_identical_bands(frame, spec, tmp_path):
    model = QuantileForecaster(spec, (0.2, 0.8), FAST).fit(frame)
    loaded = QuantileForecaster.load(model.save(tmp_path / "q.joblib"))
    assert loaded.quantiles == (0.2, 0.8)
    pd.testing.assert_frame_equal(loaded.predict_bands(frame), model.predict_bands(frame))


def test_a_series_with_no_events_cannot_be_fitted(spec):
    zeros = pd.DataFrame(
        {
            "bucket": "auto",
            "date": pd.Timestamp("2026-03-02"),
            "hour": list(range(8, 20)),
            "count": 0.0,
        }
    )
    with pytest.raises(InsufficientDataError, match="no events"):
        QuantileForecaster(spec, params=FAST).fit(build_frame(zeros, spec))
