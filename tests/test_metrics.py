import numpy as np
import pytest

from call_pace.errors import ConfigError
from call_pace.metrics import (
    apply_postprocess,
    directional_accuracy,
    interval_coverage,
    mae,
    mean_error,
    mean_interval_width,
    wape,
)


def test_postprocess_modes_and_floor_at_zero():
    raw = np.array([-0.4, 0.2, 1.5, 2.0])
    assert apply_postprocess(raw, "ceil").tolist() == [0, 1, 2, 2]
    assert apply_postprocess(raw, "round").tolist() == [0, 0, 2, 2]
    assert apply_postprocess(raw, "none").tolist() == [0, 0.2, 1.5, 2.0]
    with pytest.raises(ConfigError):
        apply_postprocess(raw, "floor")


def test_ceil_and_round_can_change_which_model_looks_better():
    actual = np.array([1, 1, 1, 1])
    raw = np.array([0.6, 0.6, 0.6, 0.6])
    assert mae(actual, apply_postprocess(raw, "ceil")) == 0.0
    assert mae(actual, apply_postprocess(raw, "round")) == 0.0
    raw = np.array([0.4, 0.4, 0.4, 0.4])
    assert mae(actual, apply_postprocess(raw, "ceil")) == 0.0
    assert mae(actual, apply_postprocess(raw, "round")) == 1.0


def test_mae_bias_and_wape_by_hand():
    actual, predicted = np.array([2, 4, 0]), np.array([3, 2, 1])
    assert mae(actual, predicted) == pytest.approx(4 / 3)
    assert mean_error(actual, predicted) == pytest.approx(0.0)
    assert wape(actual, predicted) == pytest.approx(4 / 6)
    assert np.isnan(mae([], [])) and np.isnan(mean_error([], []))
    assert np.isnan(wape([0, 0], [1, 1]))


def test_directional_accuracy_ignores_ties_with_the_baseline():
    actual = np.array([5, 5, 1, 3])
    baseline = np.array([3, 3, 3, 3])
    model = np.array([4, 2, 2, 9])
    # Hour 4 is a tie (actual == baseline) and drops out: signs match in 2 of the 3 others.
    assert directional_accuracy(actual, model, baseline) == pytest.approx(2 / 3)
    assert np.isnan(directional_accuracy([3, 3], [4, 4], [3, 3]))


def test_a_model_equal_to_the_baseline_never_has_the_right_direction():
    actual, baseline = np.array([5, 1]), np.array([3, 3])
    assert directional_accuracy(actual, baseline, baseline) == 0.0


def test_interval_coverage_includes_the_ends_and_width_is_mean_gap():
    actual = np.array([1, 2, 5])
    lower, upper = np.array([1, 3, 4]), np.array([2, 4, 4])
    assert interval_coverage(actual, lower, upper) == pytest.approx(1 / 3)
    assert mean_interval_width(lower, upper) == pytest.approx(2 / 3)
    assert np.isnan(interval_coverage([], [], []))
    assert np.isnan(mean_interval_width([], []))
