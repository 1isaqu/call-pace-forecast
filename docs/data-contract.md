# Data contract

This repository ships no data. It expects you to connect your own through a small loader. This
page is the full specification of what a loader must return, what the library validates, and how
the optional SQL Server adapter maps onto a generic relational schema.

## 1. The hourly series (required)

One row per `(bucket, date, hour)` that had activity.

| Column | Type | Meaning |
|---|---|---|
| `bucket` | string | Identifier of the series: a product line, queue or team. Models know the set of buckets seen at training time. |
| `date` | date or timestamp | Calendar day. Time of day is discarded. Time zones are dropped (naive local date). |
| `hour` | integer | Local hour of day in which the events happened, inside the operating window (default 8 to 19 inclusive). |
| `count` | number >= 0 | How many events happened in that hour. |

Optional extra columns are exogenous **covariates** (for example `days_since_event`). They must be
numeric and constant within a `(bucket, date)`; list them in `Config.covariates` (CLI:
`--covariate NAME`) to make the models use them.

Example (invented values):

```csv
bucket,date,hour,count
auto,2026-03-02,8,4
auto,2026-03-02,9,7
auto,2026-03-02,11,6
consumer,2026-03-02,9,1
```

### What an "event" is

Anything you want to forecast the cumulative volume of during the day. Typical choices:

- agreements generated (first installment of each agreement),
- outreach attempts,
- answered calls,
- right-party contacts (CPC).

Any non-negative hourly quantity works, including a monetary amount; in that case use
`Config(postprocess="none")` because rounding up to a whole number makes no sense for money.

### Rules enforced by `contract.validate_hourly`

| Situation | Behaviour |
|---|---|
| Missing required column | `ContractError` naming the columns |
| `count` null, non-numeric, infinite or negative | `ContractError` |
| `date` null or not parseable; `hour` null or not an integer | `ContractError` |
| Two rows with the same `(bucket, date, hour)` | `ContractError` (rows are never silently summed) |
| Covariate not numeric, or not constant within a bucket-day | `ContractError` |
| Row outside the operating hours, or on a closed weekday | dropped, logged as a warning |

### Sparse rows and silent days

- A missing hour inside a bucket-day that has at least one row is **zero**. The modelling frame is
  expanded to every operating hour (`features.complete_grid`).
- A day with no rows at all is **not part of the history** by default: it is assumed to be a
  holiday or an outage. If your loader omits zero-activity days that were real working days, pass
  `--fill-missing-days` (`fill_missing_days=True`) to treat every operating weekday between a
  bucket's first and last date as present. This cannot be combined with covariates.
- Events are attributed to the hour in which they occurred. "Cumulative at hour h" is inclusive of
  h; the feature used for prediction (`cumulative_lag`) excludes h.

## 2. Event amounts (optional)

Needed only for the value forecast (expected count times a typical ticket). One row per event.

| Column | Type | Meaning |
|---|---|---|
| `bucket`, `date`, `hour` | as above | when and where the event happened |
| `amount` | number >= 0 | the amount of that event, for example the down payment of an agreement |

Several rows per hour are expected. Validation is `contract.validate_amounts`.

## 3. The loader protocol

```python
from datetime import date
import pandas as pd


class MyLoader:
    def load_hourly(self, start: date, end: date) -> pd.DataFrame:
        """Rows with columns bucket, date, hour, count (+ covariates), start and end INCLUSIVE."""

    def load_amounts(self, start: date, end: date) -> pd.DataFrame:  # optional
        """Rows with columns bucket, date, hour, amount, one per event."""
```

`contract.HourlyLoader` and `contract.AmountLoader` are `typing.Protocol` classes, so no
inheritance is needed. Conventions:

- Return unvalidated data; consumers call `validate_hourly` / `validate_amounts`.
- `start` and `end` are inclusive calendar dates. The service asks for `(today, today)`.
- Raise `errors.DataSourceError` when the source cannot be read. The service turns that into a
  degraded response (see architecture). Raise `errors.InvalidRequestError` only for a caller
  mistake, which is always propagated.

Ready-made loaders: `InMemoryLoader` (DataFrames), `CsvLoader` (files in the format above) and
`SqlServerLoader` (section 4).

## 4. SQL Server adapter and the generic schema

`loaders/sqlserver.py` builds read-only T-SQL against the neutral schema below. Map your own
tables to these names with views, or edit the query builders.

| Table | Meaning | Key columns | Read by this adapter |
|---|---|---|---|
| `agreements` | one row per installment of an agreement | `agreement_id`, `portfolio_id`, `installment_no` (0 = first installment), `amount`, `paid_amount`, `issued_at`, `due_at`, `paid_at`, `status_id`, `agent_id` | yes (`agreement_id`, `portfolio_id`, `installment_no`, `amount`, `issued_at`, `status_id`, `agent_id`) |
| `agents` | users | `agent_id`, `display_name`, `login` | yes (all three) |
| `contact_events` | one row per outreach attempt | `event_id`, `debt_owner_id`, `result_id`, `agent_id`, `occurred_at` | yes (`result_id`, `agent_id`, `occurred_at`) |
| `contact_results` | result catalogue | `result_id`, `label`, `answered` (bit), `right_party` (bit) | yes (`result_id`, `answered`, `right_party`) |
| `agreement_statuses` | status catalogue | `status_id`, `label` | no |
| `agreement_debts` | links agreements to debts | `agreement_id`, `portfolio_id`, `debt_id` | no |
| `debts` | debt or contract detail | `debt_id`, `portfolio_id`, `contract_no`, `balance`, `due_at`, `days_overdue` | no |
| `debt_attributes` | extra attributes, holds the portfolio label | `debt_id`, `portfolio_label` | no |

The last four tables belong to the shared schema used across related projects; this repository
does not query them.

### Metrics the adapter can count (`SqlOptions.metric`)

| metric | Counts |
|---|---|
| `agreements` | first installments whose status is in the generated set, one row kept per `(agreement_id, portfolio_id, installment_no)` |
| `outreach` | every row of `contact_events` |
| `answers` | attempts whose result has `answered = 1` |
| `cpc` | attempts whose result has `answered = 1 AND right_party = 1` (right-party contact) |

### Status ids

Statuses keep their meaning; the ids below are **invented defaults**. Set
`SqlOptions.generated_status_ids` to your own.

| id | label | in the "generated" set |
|---|---|---|
| 10 | ACTIVE | yes |
| 20 | BROKEN | yes |
| 30 | PAID | yes |
| 40 | PENDING_EXCEPTION | no |
| 50 | REJECTED | no |
| 60 | AUTO_BROKEN | yes |
| 70 | PAID_MANUAL | yes |

The generated set counts an agreement on the day it was generated even if it breaks afterwards:
breaking is a later outcome, not a reason the agreement was not generated that day.

### Standard rules, all configurable

- Read-only: only `SELECT`, and the connection is opened with `readonly=True` and
  `ApplicationIntent=ReadOnly`.
- First installment: `installment_no = 0` (`SqlOptions.first_installment_no`).
- Deduplication: the same installment can be re-written by a system error. The count and the
  amount queries both keep the latest row per `(agreement_id, portfolio_id, installment_no)` and
  apply every business filter inside the derived table, before `ROW_NUMBER()`. A re-written
  installment is therefore counted once, in the hour of its newest row. Rows with the same
  `issued_at` tie, and the server picks one of them arbitrarily.
- Excluded agents: `SqlOptions.excluded_agent_patterns` (default `system%`, `test%`), `LIKE`
  patterns matched case-insensitively against `login` and `display_name`, applied in SQL.
- `WITH (NOLOCK)` on every table by default (`use_nolock=False` to disable). It avoids blocking
  writers at the price of possibly reading uncommitted rows.
- One database per bucket: `DB_NAMES=auto=db_a,consumer=db_b` maps buckets to databases.

Environment variables read by the adapter: `DB_DRIVER`, `DB_HOST`, `DB_PORT`, `DB_USER`,
`DB_PASSWORD`, `DB_NAMES`. See `.env.example` (placeholders only).

### What was and was not tested

Tested without a database: query text and parameter order, that no value is pasted into SQL text,
that filters sit before the ranking, result-to-DataFrame conversion with a fake cursor, error
wrapping, connection-string assembly. **Not tested:** execution against a real SQL Server. Run it
against your own test database before relying on it.

## 5. Connecting your own source

Pick one:

1. **Export to CSV** in the format of section 1 and use `--source csv --csv your_file.csv`.
2. **Expose views** named as the generic tables and use `--source sqlserver`.
3. **Write a loader** (any database, API or file). Example:

```python
class WarehouseLoader:
    def __init__(self, engine):
        self.engine = engine

    def load_hourly(self, start, end):
        sql = """
            SELECT line AS bucket, CAST(created_at AS DATE) AS date,
                   EXTRACT(HOUR FROM created_at) AS hour, COUNT(*) AS count
            FROM my_events WHERE created_at >= %(s)s AND created_at < %(e)s + 1
            GROUP BY 1, 2, 3
        """
        return pd.read_sql(sql, self.engine, params={"s": start, "e": end})
```

Then pass it to `train_models(validate_hourly(loader.load_hourly(a, b), config), ...)` and to
`PaceService(config, loader, store)`.
