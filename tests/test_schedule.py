from datetime import date

import numpy as np
import pandas as pd

from call_pace.schedule import (
    cyclic_encoding,
    next_operating_days,
    operating_days,
    weekday_position,
    weekday_positions,
)

WEEKDAYS = (0, 1, 2, 3, 4)


def test_weekday_position_is_index_among_operating_days():
    assert weekday_position(date(2026, 3, 2), WEEKDAYS) == 0  # a Monday
    assert weekday_position(date(2026, 3, 6), WEEKDAYS) == 4  # a Friday
    assert weekday_position(date(2026, 3, 7), WEEKDAYS) is None  # a Saturday


def test_weekday_position_follows_a_custom_schedule():
    assert weekday_position(date(2026, 3, 7), (1, 3, 5)) == 2


def test_weekday_positions_marks_closed_days_as_nan():
    dates = pd.Series(pd.to_datetime(["2026-03-02", "2026-03-07", "2026-03-04"]))
    result = weekday_positions(dates, WEEKDAYS)
    assert result.iloc[0] == 0 and result.iloc[2] == 2
    assert np.isnan(result.iloc[1])


def test_cyclic_encoding_wraps_around():
    sin, cos = cyclic_encoding(np.array([0, 5]), 5)
    assert sin[0] == sin[1] or abs(sin[0] - sin[1]) < 1e-12
    assert abs(cos[0] - cos[1]) < 1e-12
    sin_mid, _ = cyclic_encoding(2.5, 5)
    assert abs(float(sin_mid)) < 1e-12


def test_operating_days_is_inclusive_and_skips_weekends():
    days = operating_days(date(2026, 3, 5), date(2026, 3, 10), WEEKDAYS)
    assert days == [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 9), date(2026, 3, 10)]


def test_next_operating_days_is_strictly_after():
    days = next_operating_days(date(2026, 3, 6), 3, WEEKDAYS)  # after a Friday
    assert days == [date(2026, 3, 9), date(2026, 3, 10), date(2026, 3, 11)]
