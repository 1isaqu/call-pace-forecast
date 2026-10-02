import logging

import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.contract import (
    AmountLoader,
    HourlyLoader,
    empty_hourly,
    validate_amounts,
    validate_hourly,
)
from call_pace.errors import ContractError


def rows(*items):
    return pd.DataFrame(items, columns=["bucket", "date", "hour", "count"])


CFG = Config()


def test_valid_rows_are_typed_sorted_and_reindexed():
    df = rows(("b", "2026-03-03", 9, 2), ("a", "2026-03-02", 8, 1), ("a", "2026-03-02", 10, 0))
    out = validate_hourly(df, CFG)
    assert list(out["bucket"]) == ["a", "a", "b"]
    assert list(out["hour"]) == [8, 10, 9]
    assert str(out["date"].dtype).startswith("datetime64")
    assert out["count"].dtype == "float64"
    assert list(out.index) == [0, 1, 2]


def test_missing_columns_are_named():
    with pytest.raises(ContractError, match="count"):
        validate_hourly(pd.DataFrame({"bucket": ["a"], "date": ["2026-03-02"], "hour": [8]}), CFG)


@pytest.mark.parametrize(
    "bad",
    [
        rows(("a", "2026-03-02", 8, -1)),
        rows(("a", "2026-03-02", 8, None)),
        rows(("a", "2026-03-02", 8, float("inf"))),
        rows(("a", "not-a-date", 8, 1)),
        rows(("a", None, 8, 1)),
        rows(("a", "2026-03-02", 8.5, 1)),
        rows(("a", "2026-03-02", "eight", 1)),
    ],
)
def test_bad_values_raise(bad):
    with pytest.raises(ContractError):
        validate_hourly(bad, CFG)


def test_duplicate_keys_raise_instead_of_being_summed():
    df = rows(("a", "2026-03-02", 8, 1), ("a", "2026-03-02", 8, 4))
    with pytest.raises(ContractError, match="duplicate"):
        validate_hourly(df, CFG)


def test_rows_outside_the_schedule_are_dropped_and_logged(caplog):
    df = rows(
        ("a", "2026-03-02", 7, 1),  # before opening
        ("a", "2026-03-02", 20, 1),  # after closing
        ("a", "2026-03-07", 10, 1),  # a Saturday
        ("a", "2026-03-02", 10, 3),  # kept
    )
    with caplog.at_level(logging.WARNING):
        out = validate_hourly(df, CFG)
    assert len(out) == 1 and out.loc[0, "count"] == 3
    assert "dropped 3 rows" in caplog.text


def test_timezone_aware_dates_become_naive_midnight():
    df = rows(("a", "2026-03-02T15:30:00+00:00", 9, 1))
    out = validate_hourly(df, CFG)
    assert out.loc[0, "date"] == pd.Timestamp("2026-03-02")


def test_covariates_must_exist_be_numeric_and_constant_per_day():
    cfg = Config(covariates=("days_since_event",))
    base = rows(("a", "2026-03-02", 8, 1), ("a", "2026-03-02", 9, 2))
    with pytest.raises(ContractError, match="days_since_event"):
        validate_hourly(base, cfg)
    good = base.assign(days_since_event=[3, 3])
    assert validate_hourly(good, cfg)["days_since_event"].tolist() == [3.0, 3.0]
    with pytest.raises(ContractError, match="constant"):
        validate_hourly(base.assign(days_since_event=[3, 4]), cfg)
    with pytest.raises(ContractError, match="numeric"):
        validate_hourly(base.assign(days_since_event=["x", "y"]), cfg)


def test_empty_input_is_valid():
    out = validate_hourly(empty_hourly(), CFG)
    assert out.empty and list(out.columns) == ["bucket", "date", "hour", "count"]


def test_amounts_allow_many_events_per_hour_but_not_negative_values():
    df = pd.DataFrame(
        {
            "bucket": ["a", "a"],
            "date": ["2026-03-02", "2026-03-02"],
            "hour": [9, 9],
            "amount": [10.0, 20.0],
        }
    )
    assert len(validate_amounts(df, CFG)) == 2
    with pytest.raises(ContractError):
        validate_amounts(df.assign(amount=[10.0, -1.0]), CFG)
    with pytest.raises(ContractError, match="amount"):
        validate_amounts(df.drop(columns="amount"), CFG)


def test_loader_protocols_are_structural():
    class Both:
        def load_hourly(self, start, end):
            return empty_hourly()

        def load_amounts(self, start, end):
            return pd.DataFrame()

    assert isinstance(Both(), HourlyLoader)
    assert isinstance(Both(), AmountLoader)
    assert not isinstance(object(), HourlyLoader)
