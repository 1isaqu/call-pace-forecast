"""SQL Server adapter: query construction and plumbing, tested with fakes. No database involved."""

import sys
import types
from datetime import date, datetime

import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.contract import validate_hourly
from call_pace.errors import ConfigError, DataSourceError
from call_pace.loaders.sqlserver import (
    SqlOptions,
    SqlServerLoader,
    agreement_amounts_query,
    connect_from_env,
    connection_string,
    database_names,
    excluded_agents_clause,
    hourly_agreements_query,
    hourly_events_query,
    hourly_query,
    loader_from_env,
)

DAY = date(2026, 3, 4)
ARGS = (DAY, DAY, 8, 19)


def every_query(options):
    return [
        hourly_agreements_query(options, *ARGS),
        agreement_amounts_query(options, *ARGS),
        hourly_events_query(options, *ARGS),
    ]


def test_excluded_agent_patterns_are_bound_trimmed_and_upper_cased():
    clause, values = excluded_agents_clause(("system%", " Test% "))
    assert values == ["SYSTEM%", "SYSTEM%", "TEST%", "TEST%"]
    assert clause.count("NOT LIKE ?") == 4
    assert "COALESCE(u.login, '')" in clause and "COALESCE(u.display_name, '')" in clause
    assert "SYSTEM" not in clause.upper().replace("COALESCE", "")


def test_no_patterns_means_no_agent_filter():
    assert excluded_agents_clause(()) == ("", [])


@pytest.mark.parametrize("metric", ["agreements", "outreach", "answers", "cpc"])
def test_placeholders_match_parameters_for_every_query(metric):
    options = SqlOptions(metric=metric)
    for query in every_query(options):
        assert query.sql.count("?") == len(query.params)


def test_values_are_bound_never_pasted_into_the_sql_text():
    options = SqlOptions(excluded_agent_patterns=("robot%",), generated_status_ids=(41, 42))
    for query in every_query(options):
        for value in query.params:
            if isinstance(value, str | datetime):
                assert str(value) not in query.sql
        assert "ROBOT" not in query.sql.upper()
        assert "41" not in query.sql and "42" not in query.sql


def test_the_date_window_is_half_open_and_covers_the_whole_end_day():
    query = hourly_agreements_query(SqlOptions(), date(2026, 3, 2), date(2026, 3, 4), 8, 19)
    assert query.params[0] == datetime(2026, 3, 2, 0, 0)
    assert query.params[1] == datetime(2026, 3, 5, 0, 0)
    assert "a.issued_at >= ? AND a.issued_at < ?" in query.sql


def test_parameters_follow_the_order_of_the_placeholders():
    options = SqlOptions(generated_status_ids=(7, 8, 9), first_installment_no=3)
    query = hourly_agreements_query(options, *ARGS)
    assert query.params[2:5] == (7, 8, 9)
    assert query.params[5:8] == (3, 8, 19)
    assert query.sql.count("IN (?, ?, ?)") == 1


def test_every_table_has_the_nolock_hint_and_it_can_be_turned_off():
    for query in every_query(SqlOptions(metric="cpc")):
        tables = query.sql.count(" AS a") + query.sql.count(" AS u") + query.sql.count(" AS e")
        tables += query.sql.count("contact_results AS r")
        assert query.sql.count("WITH (NOLOCK)") == tables
    for query in every_query(SqlOptions(use_nolock=False)):
        assert "NOLOCK" not in query.sql


@pytest.mark.parametrize("builder", [agreement_amounts_query, hourly_agreements_query])
def test_agreement_queries_filter_before_ranking_and_keep_one_row_per_installment(builder):
    query = builder(SqlOptions(), *ARGS)
    inner, _, outer = query.sql.partition("\n) AS r\n")
    assert "PARTITION BY a.agreement_id, a.portfolio_id, a.installment_no" in inner
    assert "ORDER BY a.issued_at DESC" in inner
    for business_filter in ("status_id IN", "installment_no = ?", "NOT LIKE", "BETWEEN"):
        assert business_filter in inner and business_filter not in outer
    assert outer.strip().startswith("WHERE r.rn = 1")


def test_agreement_counts_count_ranked_rows_so_rewrites_cannot_inflate_them():
    sql = hourly_agreements_query(SqlOptions(), *ARGS).sql
    assert "COUNT(*) AS [count]" in sql and "DISTINCT" not in sql
    assert "GROUP BY CAST(r.issued_at AS DATE), DATEPART(HOUR, r.issued_at)" in sql


def test_counts_and_amounts_rank_the_same_rows():
    count = hourly_agreements_query(SqlOptions(), *ARGS)
    amounts = agreement_amounts_query(SqlOptions(), *ARGS)

    def ranked_source(sql):
        return sql[sql.index("ROW_NUMBER()") : sql.index("WHERE r.rn = 1")]

    assert count.params == amounts.params
    assert ranked_source(count.sql) == ranked_source(amounts.sql)


def test_event_metrics_differ_only_in_the_result_flags():
    outreach = hourly_events_query(SqlOptions(metric="outreach"), *ARGS).sql
    answers = hourly_events_query(SqlOptions(metric="answers"), *ARGS).sql
    cpc = hourly_events_query(SqlOptions(metric="cpc"), *ARGS).sql
    assert "contact_results" not in outreach
    assert "r.answered = 1" in answers and "right_party" not in answers
    assert "r.answered = 1" in cpc and "r.right_party = 1" in cpc


def test_hourly_query_picks_the_builder_from_the_metric():
    assert "agreements AS a" in hourly_query(SqlOptions(), *ARGS).sql
    assert "contact_events AS e" in hourly_query(SqlOptions(metric="outreach"), *ARGS).sql


def test_invalid_options_are_rejected():
    with pytest.raises(ConfigError):
        SqlOptions(metric="revenue")
    with pytest.raises(ConfigError):
        SqlOptions(generated_status_ids=())


# --- loader with a fake driver -----------------------------------------------------------


class FakeCursor:
    def __init__(self, log, rows, fail):
        self.log, self.rows, self.fail = log, rows, fail

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("driver exploded with secret host details")
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, log, rows, fail=False):
        self.log, self.rows, self.fail, self.closed = log, rows, fail, False

    def cursor(self):
        return FakeCursor(self.log, self.rows, self.fail)

    def close(self):
        self.closed = True


def make_loader(rows_by_bucket, log, connections, fail=()):
    def connect(bucket):
        connection = FakeConnection(log, rows_by_bucket[bucket], fail=bucket in fail)
        connections.append(connection)
        return connection

    return SqlServerLoader(connect, list(rows_by_bucket))


def test_loader_returns_contract_columns_for_every_bucket_and_closes_connections():
    log, connections = [], []
    loader = make_loader(
        {"auto": [(date(2026, 3, 4), 9, 3), (date(2026, 3, 4), 10, 1)], "consumer": []},
        log,
        connections,
    )
    frame = loader.load_hourly(DAY, DAY)
    assert list(frame.columns) == ["bucket", "date", "hour", "count"]
    assert frame["bucket"].tolist() == ["auto", "auto"]
    assert len(log) == 2 and all(c.closed for c in connections)
    clean = validate_hourly(frame, Config())
    assert clean["count"].tolist() == [3.0, 1.0]


def test_loader_amounts_have_one_row_per_event():
    log, connections = [], []
    loader = make_loader({"auto": [(DAY, 9, 12.5), (DAY, 9, 7.0)]}, log, connections)
    frame = loader.load_amounts(DAY, DAY)
    assert list(frame.columns) == ["bucket", "date", "hour", "amount"]
    assert frame["amount"].sum() == 19.5
    assert "ROW_NUMBER()" in log[0][0]


def test_a_failing_query_becomes_a_data_source_error_without_leaking_driver_text():
    log, connections = [], []
    loader = make_loader({"auto": [], "consumer": []}, log, connections, fail=("consumer",))
    with pytest.raises(DataSourceError) as caught:
        loader.load_hourly(DAY, DAY)
    assert "consumer" in str(caught.value) and "secret host" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert all(c.closed for c in connections)  # even the one that failed


def test_connection_errors_that_are_already_data_source_errors_pass_through():
    def connect(bucket):
        raise DataSourceError("cannot reach the server")

    with pytest.raises(DataSourceError, match="cannot reach"):
        SqlServerLoader(connect, ["auto"]).load_hourly(DAY, DAY)


def test_a_loader_needs_at_least_one_bucket():
    with pytest.raises(ConfigError):
        SqlServerLoader(lambda b: None, [])


# --- connection settings ------------------------------------------------------------------

ENV = {
    "DB_HOST": "db.example.invalid",
    "DB_USER": "readonly_user",
    "DB_PASSWORD": "change-me",
    "DB_NAMES": "auto=agreements_auto,consumer=agreements_consumer",
}


def test_connection_string_uses_defaults_and_declares_read_only_intent():
    text = connection_string(ENV, "agreements_auto")
    assert text.startswith("DRIVER={ODBC Driver 17 for SQL Server};SERVER=db.example.invalid,1433;")
    assert "DATABASE=agreements_auto;" in text and text.endswith("ApplicationIntent=ReadOnly")
    custom = connection_string({**ENV, "DB_DRIVER": "Driver X", "DB_PORT": "1500"}, "d")
    assert "DRIVER={Driver X}" in custom and "SERVER=db.example.invalid,1500" in custom


def test_special_characters_in_credentials_are_quoted():
    text = connection_string({**ENV, "DB_PASSWORD": "a;b}c"}, "d")
    assert "PWD={a;b}}c};" in text


def test_missing_credentials_are_named_not_defaulted():
    with pytest.raises(ConfigError, match="DB_PASSWORD"):
        connection_string({"DB_HOST": "h", "DB_USER": "u"}, "d")


def test_database_names_map_buckets_to_databases():
    assert database_names(ENV) == {"auto": "agreements_auto", "consumer": "agreements_consumer"}
    with pytest.raises(ConfigError):
        database_names({})


def install_fake_driver(monkeypatch):
    calls = []
    fake = types.ModuleType("pyodbc")
    fake.connect = lambda text, **kwargs: calls.append((text, kwargs)) or "connection"
    monkeypatch.setitem(sys.modules, "pyodbc", fake)
    return calls


def test_connect_from_env_builds_a_read_only_connection_per_bucket(monkeypatch):
    calls = install_fake_driver(monkeypatch)
    connect = connect_from_env(ENV)
    assert connect("consumer") == "connection"
    text, kwargs = calls[0]
    assert "DATABASE=agreements_consumer" in text
    assert kwargs == {"readonly": True, "timeout": 10}
    with pytest.raises(ConfigError, match="bucket"):
        connect("unknown")


def test_missing_driver_package_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyodbc", None)
    with pytest.raises(DataSourceError, match="sqlserver"):
        connect_from_env(ENV)("auto")


def test_loader_from_env_takes_its_buckets_from_db_names():
    loader = loader_from_env(ENV, Config(), SqlOptions(metric="outreach"))
    assert loader.buckets == ("auto", "consumer")
    assert loader.options.metric == "outreach"


def test_loader_output_survives_the_contract_round_trip():
    log, connections = [], []
    loader = make_loader({"auto": [(date(2026, 3, 4), 9, 3)]}, log, connections)
    frame = loader.load_hourly(DAY, DAY)
    assert isinstance(frame, pd.DataFrame)
    assert validate_hourly(frame, Config()).loc[0, "date"] == pd.Timestamp("2026-03-04")
