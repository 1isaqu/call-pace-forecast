from datetime import date

import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.contract import AmountLoader, HourlyLoader, validate_hourly
from call_pace.errors import ContractError, DataSourceError
from call_pace.loaders import CsvLoader, InMemoryLoader, filter_dates


def sample():
    return pd.DataFrame(
        {
            "bucket": ["a", "a", "a", "b"],
            "date": ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-03"],
            "hour": [9, 9, 9, 10],
            "count": [1, 2, 3, 4],
            "days_since_event": [5, 6, 7, 6],
        }
    )


def test_date_filter_includes_both_ends_and_ignores_time_of_day():
    df = pd.DataFrame({"date": ["2026-03-02 15:30", "2026-03-03", "2026-03-04"], "x": [1, 2, 3]})
    out = filter_dates(df, date(2026, 3, 2), date(2026, 3, 3))
    assert out["x"].tolist() == [1, 2]
    assert list(out.index) == [0, 1]
    assert filter_dates(df, date(2027, 1, 1), date(2027, 1, 2)).empty


def test_in_memory_loader_filters_hourly_and_amounts():
    amounts = pd.DataFrame({"bucket": ["a"], "date": ["2026-03-03"], "hour": [9], "amount": [5.0]})
    loader = InMemoryLoader(sample(), amounts)
    assert loader.load_hourly(date(2026, 3, 3), date(2026, 3, 3))["count"].tolist() == [2, 4]
    assert loader.load_amounts(date(2026, 3, 3), date(2026, 3, 3))["amount"].tolist() == [5.0]
    assert isinstance(loader, HourlyLoader) and isinstance(loader, AmountLoader)


def test_in_memory_loader_without_amounts_says_so():
    with pytest.raises(ContractError, match="amounts"):
        InMemoryLoader(sample()).load_amounts(date(2026, 3, 3), date(2026, 3, 3))


def test_csv_loader_reads_a_user_supplied_file_and_keeps_extra_columns(tmp_path):
    path = tmp_path / "hourly.csv"
    sample().to_csv(path, index=False)
    out = CsvLoader(path).load_hourly(date(2026, 3, 3), date(2026, 3, 4))
    assert out["count"].tolist() == [2, 3, 4]
    assert "days_since_event" in out.columns
    cfg = Config(covariates=("days_since_event",))
    assert len(validate_hourly(out, cfg)) == 3  # and it satisfies the contract


def test_csv_loader_supports_other_separators_and_an_amounts_file(tmp_path):
    hourly, amounts = tmp_path / "h.csv", tmp_path / "a.csv"
    sample().to_csv(hourly, index=False, sep=";")
    pd.DataFrame({"bucket": ["a"], "date": ["2026-03-02"], "hour": [9], "amount": [1.5]}).to_csv(
        amounts, index=False, sep=";"
    )
    loader = CsvLoader(hourly, amounts, sep=";")
    assert len(loader.load_hourly(date(2026, 3, 2), date(2026, 3, 4))) == 4
    assert loader.load_amounts(date(2026, 3, 2), date(2026, 3, 2))["amount"].tolist() == [1.5]


def test_csv_loader_errors_are_data_source_errors(tmp_path):
    with pytest.raises(DataSourceError, match="cannot read"):
        CsvLoader(tmp_path / "missing.csv").load_hourly(date(2026, 3, 2), date(2026, 3, 2))
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataSourceError):
        CsvLoader(empty).load_hourly(date(2026, 3, 2), date(2026, 3, 2))
    with pytest.raises(ContractError, match="amounts"):
        CsvLoader(empty).load_amounts(date(2026, 3, 2), date(2026, 3, 2))
