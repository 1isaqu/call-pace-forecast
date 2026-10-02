"""Error metrics and the post-processing that makes them match what a user sees.

Offline metrics are computed AFTER the same rounding the serving layer applies. A model that
looks better on raw floats can look worse once its output is shown as a whole number, so the
two must never be evaluated under different conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import POSTPROCESS_MODES
from .errors import ConfigError


def apply_postprocess(raw: np.ndarray | pd.Series | float, mode: str) -> np.ndarray:
    """Round a raw prediction to what is shown, never below zero."""
    if mode not in POSTPROCESS_MODES:
        raise ConfigError(f"postprocess must be one of {POSTPROCESS_MODES}")
    values = np.asarray(raw, dtype=float)
    if mode == "ceil":
        values = np.ceil(values)
    elif mode == "round":
        values = np.round(values)
    return np.clip(values, 0.0, None)


def mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Mean absolute error. NaN for empty input."""
    actual_arr, predicted_arr = np.asarray(actual, float), np.asarray(predicted, float)
    if actual_arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def mean_error(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Signed bias: positive means over-prediction."""
    actual_arr, predicted_arr = np.asarray(actual, float), np.asarray(predicted, float)
    if actual_arr.size == 0:
        return float("nan")
    return float(np.mean(predicted_arr - actual_arr))


def wape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: total absolute error over total actual volume."""
    actual_arr, predicted_arr = np.asarray(actual, float), np.asarray(predicted, float)
    total = float(np.sum(np.abs(actual_arr)))
    if total == 0.0:
        return float("nan")
    return float(np.sum(np.abs(actual_arr - predicted_arr)) / total)


def directional_accuracy(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    baseline: np.ndarray | pd.Series,
) -> float:
    """Share of hours where the model moves away from the baseline the same way reality does.

    sign(predicted - baseline) must equal sign(actual - baseline). Hours where actual equals
    the baseline carry no direction and are excluded. NaN when every hour is excluded.
    """
    actual_arr = np.asarray(actual, float)
    moved = actual_arr - np.asarray(baseline, float)
    mask = moved != 0
    if not mask.any():
        return float("nan")
    model_move = np.asarray(predicted, float) - np.asarray(baseline, float)
    return float(np.mean(np.sign(model_move[mask]) == np.sign(moved[mask])))


def interval_coverage(
    actual: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> float:
    """Share of observations that fall inside [lower, upper], ends included."""
    actual_arr = np.asarray(actual, float)
    if actual_arr.size == 0:
        return float("nan")
    inside = (actual_arr >= np.asarray(lower, float)) & (actual_arr <= np.asarray(upper, float))
    return float(np.mean(inside))


def mean_interval_width(lower: np.ndarray | pd.Series, upper: np.ndarray | pd.Series) -> float:
    """Average width of the interval, in the unit of the target."""
    width = np.asarray(upper, float) - np.asarray(lower, float)
    return float(np.mean(width)) if width.size else float("nan")
