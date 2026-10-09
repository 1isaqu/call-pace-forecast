"""Read-only SQL Server (T-SQL) loader for the generic schema in docs/data-contract.md.

NOT exercised against a live database in this repository. What is tested without one: query
construction (text and parameters), result-to-DataFrame conversion with a fake cursor, error
wrapping and connection-string assembly. Run it against your own server in a test environment
before trusting it, and map your tables to the generic names (or edit the builders below).

Conventions:
  * Every value is passed as a `?` parameter. Nothing from configuration or callers is ever
    pasted into the SQL text.
  * Every table carries `WITH (NOLOCK)` by default: analytical reads should not block writers.
    The price is that uncommitted rows can be seen; disable it with `use_nolock=False`.
  * Business filters (status set, first installment, hours, excluded agents) sit inside the
    derived table, BEFORE the `ROW_NUMBER()` that keeps one row per installment. A filter applied
    after the ranking would let a re-written row with a non-qualifying status win the rank and
    hide the valid one.
  * Excluded-agent patterns are LIKE patterns matched case-insensitively against both the
    agent login and display name (`%` and `_` are wildcards).
  * One database per bucket: the connection factory is called with the bucket name.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from ..config import Config, parse_pairs
from ..contract import empty_hourly
from ..errors import ConfigError, DataSourceError

METRICS = ("agreements", "outreach", "answers", "cpc")
DEFAULT_DRIVER = "ODBC Driver 17 for SQL Server"


@dataclass(frozen=True)
class SqlOptions:
    """What to count and which rows are excluded. Status ids here are invented defaults."""

    metric: str = "agreements"
    generated_status_ids: tuple[int, ...] = (10, 20, 30, 60, 70)
    first_installment_no: int = 0
    excluded_agent_patterns: tuple[str, ...] = ("system%", "test%")
    use_nolock: bool = True

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ConfigError(f"metric must be one of {METRICS}")
        if not self.generated_status_ids:
            raise ConfigError("generated_status_ids must not be empty")


@dataclass(frozen=True)
class Query:
    """SQL text with `?` placeholders and the parameters to bind, in order."""

    sql: str
    params: tuple[Any, ...]


def _hint(options: SqlOptions) -> str:
    return " WITH (NOLOCK)" if options.use_nolock else ""


def _placeholders(count: int) -> str:
    return ", ".join("?" * count)


def excluded_agents_clause(patterns: Sequence[str], alias: str = "u") -> tuple[str, list[str]]:
    """AND-ed NOT LIKE predicates over login and display name, plus their bind values.

    COALESCE keeps rows whose name is NULL: `NULL NOT LIKE 'x%'` is unknown, which would
    silently drop them.
    """
    clauses: list[str] = []
    values: list[str] = []
    for pattern in patterns:
        for column in ("login", "display_name"):
            clauses.append(f"AND UPPER(LTRIM(RTRIM(COALESCE({alias}.{column}, '')))) NOT LIKE ?")
            values.append(pattern.strip().upper())
    return "\n      ".join(clauses), values


def _day_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    """Half-open [start 00:00, day after end 00:00)."""
    return datetime.combine(start, time.min), datetime.combine(end + timedelta(days=1), time.min)


def _latest_first_installments(
    options: SqlOptions, start: date, end: date, start_hour: int, end_hour: int, columns: str
) -> tuple[str, tuple[Any, ...]]:
    """FROM clause and parameters: one row per (agreement_id, portfolio_id, installment_no).

    The same installment can be re-written, so rows are ranked by `issued_at`, newest first, and
    only the newest is kept. Every business filter is inside the derived table, before the
    ranking. Rows with the same `issued_at` tie, and the server picks one of them arbitrarily.
    `columns` lists what the outer query reads from the derived table `r`.
    """
    low, high = _day_bounds(start, end)
    agents, agent_values = excluded_agents_clause(options.excluded_agent_patterns)
    hint = _hint(options)
    sql = f"""
FROM (
    SELECT {columns},
           ROW_NUMBER() OVER (
               PARTITION BY a.agreement_id, a.portfolio_id, a.installment_no
               ORDER BY a.issued_at DESC
           ) AS rn
    FROM agreements AS a{hint}
    JOIN agents AS u{hint} ON u.agent_id = a.agent_id
    WHERE a.issued_at >= ? AND a.issued_at < ?
      AND a.status_id IN ({_placeholders(len(options.generated_status_ids))})
      AND a.installment_no = ?
      AND DATEPART(HOUR, a.issued_at) BETWEEN ? AND ?
      {agents}
) AS r
WHERE r.rn = 1""".strip()
    params = (
        low,
        high,
        *options.generated_status_ids,
        options.first_installment_no,
        start_hour,
        end_hour,
        *agent_values,
    )
    return sql, params


def hourly_agreements_query(
    options: SqlOptions, start: date, end: date, start_hour: int, end_hour: int
) -> Query:
    """Agreements generated per day and hour, counted on first installments.

    One row is kept per (agreement_id, portfolio_id, installment_no) before counting, so a
    re-written installment is counted once, in the hour of its newest row.
    """
    source, params = _latest_first_installments(
        options, start, end, start_hour, end_hour, "a.issued_at"
    )
    sql = f"""
SELECT CAST(r.issued_at AS DATE) AS [date],
       DATEPART(HOUR, r.issued_at) AS [hour],
       COUNT(*) AS [count]
{source}
GROUP BY CAST(r.issued_at AS DATE), DATEPART(HOUR, r.issued_at)
ORDER BY [date], [hour]
""".strip()
    return Query(sql, params)


def agreement_amounts_query(
    options: SqlOptions, start: date, end: date, start_hour: int, end_hour: int
) -> Query:
    """One row per generated agreement with its first-installment amount.

    Re-written installments are collapsed before any amount is read: duplicates would otherwise
    be counted twice in a SUM. Every business filter is inside the derived table.
    """
    source, params = _latest_first_installments(
        options, start, end, start_hour, end_hour, "a.issued_at, a.amount"
    )
    sql = f"""
SELECT CAST(r.issued_at AS DATE) AS [date],
       DATEPART(HOUR, r.issued_at) AS [hour],
       r.amount AS [amount]
{source}
ORDER BY [date], [hour]
""".strip()
    return Query(sql, params)


def hourly_events_query(
    options: SqlOptions, start: date, end: date, start_hour: int, end_hour: int
) -> Query:
    """Outreach attempts per day and hour.

    metric "outreach" counts the attempts inside the operating hours whose agent exists in
    `agents` and is not excluded. "answers" and "cpc" apply the same filters: "answers" keeps
    the attempts whose result is flagged answered, "cpc" those flagged answered AND right-party
    (right-party contact).
    """
    low, high = _day_bounds(start, end)
    agents, agent_values = excluded_agents_clause(options.excluded_agent_patterns)
    hint = _hint(options)
    join, flags = "", ""
    if options.metric in ("answers", "cpc"):
        join = f"\nJOIN contact_results AS r{hint} ON r.result_id = e.result_id"
        flags = "\n  AND r.answered = 1" + (
            "\n  AND r.right_party = 1" if options.metric == "cpc" else ""
        )
    sql = f"""
SELECT CAST(e.occurred_at AS DATE) AS [date],
       DATEPART(HOUR, e.occurred_at) AS [hour],
       COUNT(*) AS [count]
FROM contact_events AS e{hint}
JOIN agents AS u{hint} ON u.agent_id = e.agent_id{join}
WHERE e.occurred_at >= ? AND e.occurred_at < ?
  AND DATEPART(HOUR, e.occurred_at) BETWEEN ? AND ?{flags}
      {agents}
GROUP BY CAST(e.occurred_at AS DATE), DATEPART(HOUR, e.occurred_at)
ORDER BY [date], [hour]
""".strip()
    return Query(sql, (low, high, start_hour, end_hour, *agent_values))


def hourly_query(
    options: SqlOptions, start: date, end: date, start_hour: int, end_hour: int
) -> Query:
    """The hourly-count query for `options.metric`."""
    builder = hourly_agreements_query if options.metric == "agreements" else hourly_events_query
    return builder(options, start, end, start_hour, end_hour)


class SqlServerLoader:
    """Hourly counts and event amounts straight from SQL Server, one database per bucket."""

    def __init__(
        self,
        connect: Callable[[str], Any],
        buckets: Sequence[str],
        options: SqlOptions | None = None,
        config: Config | None = None,
    ) -> None:
        if not buckets:
            raise ConfigError("at least one bucket is required")
        self._connect = connect
        self.buckets = tuple(buckets)
        self.options = options or SqlOptions()
        self._config = config or Config()

    def load_hourly(self, start: date, end: date) -> pd.DataFrame:
        frames = []
        for bucket in self.buckets:
            query = hourly_query(
                self.options, start, end, self._config.start_hour, self._config.end_hour
            )
            rows = self._run(bucket, query)
            frames.append(_to_frame(bucket, rows, ["date", "hour", "count"]))
        return pd.concat(frames, ignore_index=True) if frames else empty_hourly()

    def load_amounts(self, start: date, end: date) -> pd.DataFrame:
        frames = []
        for bucket in self.buckets:
            query = agreement_amounts_query(
                self.options, start, end, self._config.start_hour, self._config.end_hour
            )
            frames.append(_to_frame(bucket, self._run(bucket, query), ["date", "hour", "amount"]))
        return pd.concat(frames, ignore_index=True)

    def _run(self, bucket: str, query: Query) -> list[tuple[Any, ...]]:
        try:
            connection = self._connect(bucket)
            try:
                cursor = connection.cursor()
                cursor.execute(query.sql, query.params)
                return [tuple(row) for row in cursor.fetchall()]
            finally:
                connection.close()
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"query failed for bucket {bucket!r} ({type(exc).__name__})"
            ) from exc


def _to_frame(bucket: str, rows: list[tuple[Any, ...]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=columns)
    frame.insert(0, "bucket", bucket)
    return frame


def _odbc_value(value: str) -> str:
    """Quote a connection-string value that contains characters ODBC treats specially."""
    if any(ch in value for ch in ";{}"):
        return "{" + value.replace("}", "}}") + "}"
    return value


def connection_string(environ: Mapping[str, str], database: str) -> str:
    """ODBC connection string from DB_* variables. Read-only intent is declared."""
    missing = [name for name in ("DB_HOST", "DB_USER", "DB_PASSWORD") if not environ.get(name)]
    if missing:
        raise ConfigError(f"missing environment variables: {missing}")
    driver = environ.get("DB_DRIVER") or DEFAULT_DRIVER
    port = environ.get("DB_PORT") or "1433"
    return (
        f"DRIVER={{{driver}}};SERVER={environ['DB_HOST']},{port};DATABASE={database};"
        f"UID={_odbc_value(environ['DB_USER'])};PWD={_odbc_value(environ['DB_PASSWORD'])};"
        "ApplicationIntent=ReadOnly"
    )


def database_names(environ: Mapping[str, str]) -> dict[str, str]:
    """Bucket to database name from `DB_NAMES=bucket=database,...`."""
    names = parse_pairs(environ.get("DB_NAMES", ""))
    if not names:
        raise ConfigError("DB_NAMES must map at least one bucket to a database")
    return names


def connect_from_env(environ: Mapping[str, str]) -> Callable[[str], Any]:
    """A connection factory configured only from environment variables.

    `pyodbc` is imported when the first connection is requested, so the rest of the library
    works without it installed.
    """
    names = database_names(environ)

    def connect(bucket: str) -> Any:
        if bucket not in names:
            raise ConfigError(f"no database configured for bucket {bucket!r}")
        try:
            driver = importlib.import_module("pyodbc")
        except ImportError as exc:
            raise DataSourceError('install the optional extra: pip install ".[sqlserver]"') from exc
        return driver.connect(connection_string(environ, names[bucket]), readonly=True, timeout=10)

    return connect


def loader_from_env(
    environ: Mapping[str, str], config: Config | None = None, options: SqlOptions | None = None
) -> SqlServerLoader:
    """A ready SqlServerLoader whose buckets are the keys of `DB_NAMES`."""
    return SqlServerLoader(
        connect_from_env(environ), tuple(database_names(environ)), options, config
    )
