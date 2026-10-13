"""The public call-center data: cleaning, hourly counts, hash checks. No network, no data."""

import hashlib
import io
import sys

import pandas as pd
import pytest

from call_pace import validate_hourly
from call_pace.anonymous_bank import (
    ARCHIVE_URL,
    DOCUMENTATION_URL,
    MONTH_SHA256,
    OPERATING_WEEKDAYS,
    anonymous_bank_config,
    check_hash,
    clean_calls,
    extract_archive,
    fetch,
    hourly_counts,
    read_calls,
)
from call_pace.errors import ContractError, DataSourceError

COLUMNS = [
    "vru+line", "call_id", "customer_id", "priority", "type", "date", "vru_entry", "vru_exit",
    "vru_time", "q_start", "q_exit", "q_time", "outcome", "ser_start", "ser_exit", "ser_time",
    "server",
]  # fmt: skip


def row(call_id, kind, date, entry, outcome="AGENT"):
    """One invented record in the layout of the files."""
    return [
        "AA0101", str(call_id), "0", "0", kind, date, entry, entry, "8", "0:00:00", "0:00:00", "0",
        outcome, "0:00:00", "0:00:00", "0", "NO_SERVER",
    ]  # fmt: skip


# 1999-01-03 was a Sunday, 1999-01-04 a Monday and 1999-01-01 a Friday.
RECORDS = [
    row(1, "PS", "990103", "7:05:00"),
    row(2, "PS", "990103", "7:30:10"),
    row(3, "PS", "990103", "9:10:00", "HANG"),
    row(4, "TT ", "990103", "10:15:00"),  # trailing space in the type
    row(5, "NW", "990104", "11:00:00"),
    row(6, "PS", "990103", "8:00:00", "PHANTOM"),  # to be ignored
    row(7, "AA", "990103", "8:00:00"),  # a type the documentation does not describe
    ["AA0101", "8", "0", "0", "990103", "8:30:00", "8:30:05", "0:00:00", "0:00:00"],  # no type cell
    row(9, "PS", "990101", "9:00:00"),  # Friday: kept by cleaning, outside the schedule
    row(10, "PS", "990103", "2:00:00"),  # night: kept by cleaning, outside the schedule
]


def raw_frame() -> pd.DataFrame:
    text = "\n".join("\t".join(r) for r in [COLUMNS, *RECORDS])
    frame = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    return frame.rename(columns={"vru+line": "vru_line"})


def test_clean_calls_counts_every_drop():
    calls, quality = clean_calls(raw_frame())
    assert quality.raw_rows == 10
    assert quality.malformed_rows == 1
    assert quality.type_trailing_space == 1
    assert quality.phantom_rows == 1
    assert quality.undocumented_type_rows == 1 and quality.undocumented_types == ("AA",)
    assert quality.calls_kept == len(calls) == 7
    assert quality.calendar_days == 3
    assert list(calls.columns) == ["type", "date", "hour"]
    assert "TT" in set(calls["type"]) and "TT " not in set(calls["type"])


def test_clean_calls_needs_the_documented_columns():
    with pytest.raises(ContractError, match="vru_entry"):
        clean_calls(raw_frame().drop(columns="vru_entry"))


def test_hourly_counts_follow_the_data_contract():
    calls, _ = clean_calls(raw_frame())
    counts = hourly_counts(calls)
    assert list(counts.columns) == ["bucket", "date", "hour", "count"]
    sunday = counts[counts["date"] == "1999-01-03"].set_index(["bucket", "hour"])["count"]
    assert sunday[("PS", 7)] == 2 and sunday[("PS", 9)] == 1 and sunday[("TT", 10)] == 1
    assert counts["count"].sum() == 7


def test_the_calendar_keeps_sunday_to_thursday_and_the_staffed_hours():
    config = anonymous_bank_config()
    assert config.operating_weekdays == OPERATING_WEEKDAYS == (0, 1, 2, 3, 6)
    assert pd.Timestamp("1999-01-03").weekday() == 6  # a Sunday counts as an operating day
    calls, _ = clean_calls(raw_frame())
    hourly = validate_hourly(hourly_counts(calls), config)
    assert hourly["count"].sum() == 5  # the Friday call and the 2h call are outside the schedule
    assert set(hourly["bucket"]) == {"PS", "TT", "NW"}
    assert hourly["hour"].min() >= 7 and hourly["hour"].max() <= 23


def test_the_protocol_can_be_overridden():
    config = anonymous_bank_config(n_folds=3)
    assert config.n_folds == 3 and config.fold_size_days == 20 and config.start_hour == 7


def test_read_calls_reads_the_twelve_files(tmp_path):
    text = "\n".join("\t".join(r) for r in [COLUMNS, *RECORDS]) + "\n"
    for name in MONTH_SHA256:
        (tmp_path / name).write_text(text, encoding="utf-8")
    calls, quality = read_calls(tmp_path)
    assert quality.raw_rows == 120 and quality.calls_kept == 84 and len(calls) == 84


def test_read_calls_names_the_missing_files(tmp_path):
    (tmp_path / "January1999.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DataSourceError, match="run fetch first"):
        read_calls(tmp_path)


def test_check_hash_accepts_a_match_and_rejects_a_difference():
    data = b"call center"
    check_hash(data, hashlib.sha256(data).hexdigest(), "sample")
    with pytest.raises(DataSourceError, match="SHA-256"):
        check_hash(data + b"!", hashlib.sha256(data).hexdigest(), "sample")


def test_the_pinned_sources_are_https_hashes_of_the_right_shape():
    assert ARCHIVE_URL.startswith("https://") and DOCUMENTATION_URL.startswith("https://")
    assert len(MONTH_SHA256) == 12
    for name, digest in MONTH_SHA256.items():
        assert name.endswith("1999.txt") and len(digest) == 64 and int(digest, 16) >= 0


class Response:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.data


def test_fetch_refuses_a_download_with_the_wrong_hash(tmp_path):
    def urlopen(url, timeout):
        return Response(b"not the archive")

    with pytest.raises(DataSourceError, match="AnonymousBank.rar"):
        fetch(tmp_path, urlopen=urlopen)
    assert not (tmp_path / "AnonymousBank.rar").exists()


def test_fetch_turns_a_network_failure_into_a_data_source_error(tmp_path):
    def urlopen(url, timeout):
        raise OSError("no route")

    with pytest.raises(DataSourceError, match="could not download"):
        fetch(tmp_path, urlopen=urlopen)


def test_extract_archive_says_which_extra_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "unrar", None)  # as if the extra were not installed
    with pytest.raises(DataSourceError, match="realdata"):
        extract_archive(tmp_path / "AnonymousBank.rar", tmp_path)
