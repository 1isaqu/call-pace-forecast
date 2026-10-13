# call-pace-forecast

Intraday forecasting of cumulative outreach or agreement volume. For every hour of the operating
day it answers "is the operation on pace?" with three numbers: what was expected, what happened and
the difference. Expectations come from three phases scored with the same walk-forward protocol: a
lookup baseline, KNN regression on time-of-day curves, and LightGBM quantile bands. The protocol
never lets the future leak into the past, and a Monte Carlo stress test shows how a monthly
retraining policy behaves under volume growth.

It is a library, a command line and an optional HTTP endpoint. It contains no data and no fitted
models: you connect your own data through a small, documented contract.

## What it does

- The expected count for an hour is the median of what that (bucket, hour) produced recently. KNN
  and LightGBM have to beat it under scale-relative promotion rules. A fold counts against a
  challenger only when it is worse than the baseline by more than both an absolute and a relative
  tolerance, so any number of folds that are only slightly worse do not block promotion.
- The volume so far is the total before the hour being predicted, and the early-morning total is
  hidden until it has closed. Training and serving build features through the same code, and a test
  replays a past day hour by hour to check that they agree.
- Walk-forward evaluation with an expanding window. k is chosen on the early folds and the last fold
  is a holdout that is looked at once. Metrics are computed after the same rounding the display
  applies.
- The quantile model reports a band per hour, and the tool prints the nominal and the empirical
  coverage so you can see whether the band can be trusted.
- The experiments are included as code: feature-subset search, lookup variants, a covariate signal
  test, an end-of-day projection comparison (a formula against neighbour models), a finer-key test
  that weighs a gain against how thin the cells become, and a ticket-lookup search for an
  expected-value forecast.
- Bad requests raise and missing models raise, but a failing data source degrades to an empty,
  well-formed payload so a polling display keeps working.

## Architecture

```
 loader (CSV, SQL Server, yours)
     |
     v  validate_hourly                      training / evaluation (offline)
 hourly series --> FeatureSpec + build_frame --+--> run_walk_forward --> promotion decision
 bucket,date,hour,count                        +--> feature_subset_search, lookup variants, signal test
                                               +--> run_monte_carlo (rolling retraining under growth)
                                               +--> train_models --> artifacts/ + manifest.json
                                                                           |
 request --> PaceService.pace(bucket) <------------------------------------+   serving (online)
               - outside operating hours: empty payload, no model or data reads
               - load models routed to the bucket (lookup | knn | quantile)
               - read today's counts (cached 30 s), build features for the current hour
               - expected / actual / delta per hour, aggregate buckets, projected close
               - optional value: expected count x typical ticket
           --> {"meta": ..., "data": ..., "errors": [...]}
```

More detail in [docs/architecture.md](docs/architecture.md).

## The three phases

| Phase | Model | Uses today's volume | Output |
|---|---|---|---|
| 1 | `LookupForecaster`: median per (bucket, hour) over a trailing window | no | expected count |
| 2 | `KnnForecaster`: standardised KNN over hour, cyclic weekday, volume so far, early-window total, bucket | yes | expected count |
| 3 | `QuantileForecaster`: LightGBM, Poisson centre plus lower and upper quantile models | yes | expected count and a band |

The shape of the pace table stays the same across phases: a better model makes `expected` more
accurate, and the third phase adds a band around it.

## Quick start

Requires Python 3.11 or newer. Everything below runs on invented data generated on the spot.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

call-pace synth --out data/hourly.csv --amounts-out data/amounts.csv --days 120
call-pace train --csv data/hourly.csv --amounts-csv data/amounts.csv
call-pace validate --csv data/hourly.csv --per-bucket
call-pace montecarlo --csv data/hourly.csv --scenarios 100
call-pace pace --csv data/hourly.csv --amounts-csv data/amounts.csv --bucket all --now 2026-06-19T11:20:00
```

`synth` writes `data/` (git-ignored). `train` writes `artifacts/` (git-ignored). `pace` replays a
moment of the last synthetic day (2026-06-19 with these options): the `--now` option overrides the
clock and the loader reads that day's rows from the CSV.

Abridged output of the `validate` command above. The data is invented and the run is deterministic.
The examples in this README were run on Linux, the platform CI uses, with Python 3.13, numpy 2.5,
pandas 3.0, scikit-learn 1.9 and LightGBM 4.7. The KNN digits can differ on another platform or
library version:

```
Fold metrics
| fold | n | mae_lookup | mae_knn | dir_knn | mae_quantile | dir_quantile | coverage | width |
|---|---|---|---|---|---|---|---|---|
| 1 | 240 | 1.846 | 1.938 | 0.487 | 1.783 | 0.534 | 0.533 | 2.846 |
...
Selected k: 10

bucket auto
PROMOTE: quantile
  [ok] improvement: holdout MAE 2.633 vs baseline 2.783 (+5.4%), need >= 5.0%
  [ok] directional_accuracy: 62.9%, need >= 55%
  [ok] no_fold_regression: no fold clearly worse than the baseline
  [ok] max_mae: no absolute cap configured
quantile band: nominal coverage 50%, empirical 57.5% on the holdout fold, mean width 4.50

bucket consumer
KEEP BASELINE: quantile
  [fail] improvement: holdout MAE 1.083 vs baseline 0.908 (-19.3%), need >= 5.0%
```

The synthetic data has one high-volume bucket and one sparse bucket. In this run the quantile model
is promoted on the high-volume bucket and the KNN is not, and neither challenger beats the lookup on
the sparse one, which is why decisions are made per bucket.

The serving payload of `call-pace pace` (abridged, from the same run on the synthetic data with
seed 7; the `knn` figures can differ by platform as above):

```json
{
  "meta": {"in_operation": true, "bucket": "all", "buckets": ["auto", "consumer"],
           "model": "knn", "value_model": "ticket_x_expected_count"},
  "data": {
    "current_hour": 11, "cumulative_actual": 21, "expected_total": 75, "projected_close": 76,
    "value_cumulative_actual": 6473.25, "value_expected_total": 30501.42,
    "value_projected_close": 28541.11,
    "bands": [
      {"hour": 10, "expected": 8, "actual": 10, "delta": 2, "status": "above", "cumulative": 21,
       "expected_value": 3598.31, "actual_value": 3140.1, "status_value": "below"},
      {"hour": 11, "expected": 9, "actual": null, "delta": null, "status": "in_progress",
       "expected_value": 3720.14, "actual_value": null, "status_value": "in_progress"}
    ]
  },
  "errors": []
}
```

`montecarlo` prints, per simulated month, percentiles of the model's MAE, of the lookup's MAE and of
directional accuracy, then two probabilities. 100 scenarios took about 20 seconds on a 4-core Linux
machine and about 2 minutes 40 seconds on the slower development machine. On the synthetic data the
last lines were:

```
P(directional accuracy > 60% in month 6) = 8.0%
P(model beats the lookup in month 6) = 75.0%
```

The HTTP endpoint is the same payload: `pip install -e ".[api]"` then
`call-pace serve --csv data/hourly.csv --amounts-csv data/amounts.csv` serves `GET /pace/{bucket}`.

### As a library

```python
from call_pace import Config, FeatureSpec, build_frame, decide_promotion, run_walk_forward
from call_pace import validate_hourly
from call_pace.synthetic import make_synthetic_hourly

config = Config()
hourly = validate_hourly(make_synthetic_hourly(n_days=100), config)  # replace with your DataFrame
spec = FeatureSpec.from_data(hourly, config)
frame = build_frame(hourly, spec)

result = run_walk_forward(frame, config, spec)
print(result.fold_metrics.round(3))
print(decide_promotion(result, "knn").explain())
```

```python
from datetime import datetime
from call_pace import ArtifactStore, PaceService, train_models
from call_pace.loaders import InMemoryLoader

store = ArtifactStore("artifacts")
train_models(hourly, config, store)
last_day = hourly["date"].max()
service = PaceService(
    config,
    InMemoryLoader(hourly),
    store,
    clock=lambda: datetime(last_day.year, last_day.month, last_day.day, 11, 20),
)
payload = service.pace("all")
print(payload["data"]["projected_close"])
```

## Connect your own data

Everything you need is in [docs/data-contract.md](docs/data-contract.md).
[docs/data-sources.md](docs/data-sources.md) says where the data in this repository comes from
(only the seeded generator) and how to keep your own out of it. In short:

1. The core input is one row per `(bucket, date, hour)` with a non-negative `count`.
   Optional extras: event amounts for value forecasts, numeric covariates.
2. Provide it as a CSV (`--source csv`), through the SQL Server adapter (`--source sqlserver`, a
   generic schema documented in the contract), or by writing a loader with `load_hourly(start, end)`.
3. Run `call-pace validate` first. If the lookup baseline wins on your data, keep serving the lookup.

The SQL Server adapter builds parameterised, read-only T-SQL. **It has not been executed against a
live database in this repository.** What is tested without one: query text and parameters, that no
value is pasted into the SQL, that business filters precede the ranking, result conversion with a
fake cursor, error wrapping and connection-string assembly.

## Walk-forward protocol and promotion

The last `n_folds * fold_size_days` days are cut into consecutive test blocks; each block is
predicted by a model fitted only on earlier days. k is selected on the early folds, the final fold
is the holdout. A challenger is promoted only if it improves on the baseline by a configured
percentage, has acceptable directional accuracy and has no fold that is clearly worse (worse by both
an absolute and a relative tolerance). See [docs/evaluation.md](docs/evaluation.md).

## How to interpret the bands

- A band is a range of plausible hourly counts. With quantiles 0.25 and 0.75 the nominal coverage
  is 50 percent; `validate` prints the empirical coverage on the holdout fold. Far above nominal
  means the band is wider than needed, far below means the model is overconfident. The library
  does not recalibrate bands.
- For a bucket served by the quantile model, each closed hour carries `position`: `below_band`,
  `within_band` or `above_band`. At low volume a delta of -1 is noise; `within_band` says so,
  `below_band` is the signal worth a look.
- Bands exist per bucket only. The projected close is a point estimate without an interval, because
  summing per-hour bounds does not bound the sum.
- A band is served only for a bucket routed to the quantile model. Set
  `CALL_PACE_ROUTING=auto=quantile` in the environment (`export` in a POSIX shell,
  `$env:CALL_PACE_ROUTING = "auto=quantile"` in PowerShell, `set` in cmd), then run
  `call-pace pace --bucket auto` with the other options of the quick start.
  On that invented data hour 8 comes out as an expectation of 4, a band of 3 to 6, an actual of 2
  and the position `below_band`.

## Configuration

`Config` is a frozen dataclass. Every default was chosen for the synthetic example and has to be
calibrated on your own data.

| Field | Default | Meaning |
|---|---|---|
| `start_hour`, `end_hour` | 8, 19 | Operating window (inclusive) |
| `operating_weekdays` | Monday to Friday (0 to 4) | Days the site is open |
| `first_window_hours` | 2 | Hours summed into the early-window feature |
| `covariates` | none | Extra numeric columns constant within a bucket-day |
| `postprocess` | `ceil` | Rounding applied to every model output and metric (`ceil`, `round`, `none`) |
| `lookup_window_days` | 60 | Trailing window of the lookup and the ticket. The lookup keeps the last N days that have rows; the ticket keeps the last N operating weekdays, silent days included. With `None` the lookup uses all history and the ticket uses 60 days |
| `lookup_stat`, `lookup_treatment` | `median`, `none` | Central statistic and outlier treatment |
| `maturity_cutoff_days` | none | Per bucket: most recent days ignored when fitting |
| `k_grid`, `default_k` | 5, 7, 10, 15, 20; 10 | KNN neighbours searched and fallback |
| `quantiles` | 0.25, 0.75 | Band quantiles |
| `lgbm_params` | 300 trees, learning rate 0.05, 15 leaves | LightGBM settings |
| `n_folds`, `fold_size_days`, `min_train_days` | 4, 10, 30 | Walk-forward protocol |
| `default_model`, `routing` | `knn`, none | Model served per bucket |
| `artifacts_dir`, `artifact_ttl_seconds` | `artifacts`, 86400 | Where models live and how long they are cached |

Environment variables read by `Config.from_env()`: `CALL_PACE_START_HOUR`, `CALL_PACE_END_HOUR`,
`CALL_PACE_LOOKUP_WINDOW_DAYS`, `CALL_PACE_ARTIFACTS_DIR`, `CALL_PACE_ARTIFACT_TTL_SECONDS`,
`CALL_PACE_DEFAULT_MODEL`, `CALL_PACE_MATURITY_CUTOFF_DAYS` (`bucket=days,...`),
`CALL_PACE_ROUTING` (`bucket=model,...`). The SQL Server adapter reads `DB_DRIVER`, `DB_HOST`,
`DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAMES` (`bucket=database,...`). `.env.example` lists them
with placeholder values. The library does not load `.env` files itself, so set the variables in
the environment.

Commands:

| Command | What it does |
|---|---|
| `call-pace synth` | write an invented dataset (hourly counts, optional amounts, optional covariate) |
| `call-pace train` | fit lookup, KNN, quantile (and a ticket lookup with `--amounts-csv`) into `artifacts/` |
| `call-pace validate` | walk-forward table, k selection, promotion decisions, band coverage (`--per-bucket`) |
| `call-pace projection` | end-of-day total: share formula against neighbour models; `--extra-key weekday_pos` also tests whether a finer share-curve key earns its thinner cells |
| `call-pace ablate features\|lookup\|covariate` | ablation studies |
| `call-pace montecarlo` | retraining stress test under volume growth |
| `call-pace pace` | print today's expected / actual / delta table as JSON |
| `call-pace serve` | run the HTTP endpoint (needs the `api` extra) |

## Running the tests

```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```

The suite uses tiny inline fixtures and an in-memory synthetic generator; there are no data files.
It covers the data contract, feature engineering (including train/serve parity), the three models,
walk-forward folds and promotion rules, projection, ablation, Monte Carlo reproducibility, the
ticket lookup, the pace logic, the service failure policy, artifacts and caching, the CLI, the HTTP
app and SQL query construction. The whole suite (333 tests) took about 25 seconds on a 4-core Linux
machine and between two and three minutes on the slower development machine, depending on load.

## Project layout

```
src/call_pace/
  config.py  errors.py  schedule.py  contract.py  features.py  metrics.py
  baseline.py  knn.py  quantile.py  persistence.py        the three phases
  projection.py  value.py                                  end-of-day total, ticket lookup
  validation.py  ablation.py  montecarlo.py  synthetic.py  evaluation and experiments
  pace.py  service.py  api.py                              serving
  artifacts.py  training.py                                runtime model files, train_models
  loaders/  common.py  memory.py  csv_loader.py  sqlserver.py
  cli.py  __main__.py
tests/                                                     one test module per area
docs/  architecture.md  data-contract.md  data-sources.md  evaluation.md  decisions.md  glossary.md
```

## Key design decisions

The full list with reasons is in [docs/decisions.md](docs/decisions.md). The ones that shaped the code:

- The hour is an exact partition key in the lookup, not a distance dimension.
- Offline metrics use the serving rounding, so a ranking cannot flip between lab and display.
- A fold counts against a challenger only if it is worse than the baseline by both an absolute and
  a relative margin.
- The share formula is the reference for the day total, and the neighbour models are tested
  against it on held-out days. On the synthetic data the neighbour models win in the first hours.
- Routing is per bucket, and the review trigger for a sparse bucket is explicit.
- Value is expected count times a typical ticket, with an error instead of an invented ticket when
  there is no data.
- A data failure degrades the response; a bad request or a missing model does not.
- Data access uses bound parameters and applies every business filter before the "latest row" ranking.

## Limitations

- Nothing here is validated on real data. Every example output in the documentation comes from
  synthetic data, and the defaults were chosen for it. Window lengths, cut-offs, fold sizes, k grids,
  thresholds and Monte Carlo parameters must be calibrated on your own history.
- The SQL Server adapter has never been run against a live server (see above).
- Development ran on Python 3.14 with numpy 2.5, pandas 3.0, scikit-learn 1.9 and LightGBM 4.7. CI
  runs lint and the tests on Python 3.11, 3.12, 3.13 and 3.14 on Ubuntu with the newest releases
  pip resolves. The minimum versions in `pyproject.toml` have not been tested.
- The data is hourly, the schedule is by weekday and there is one site. Intra-hour effects,
  holidays and multiple time zones are not modelled. Days with no rows are treated as closed unless
  you ask otherwise.
- KNN and LightGBM here are tuned for hundreds of days of hourly counts. With little history or very
  sparse buckets the lookup baseline can be the right answer.
- Quantile bands are not calibrated, and there is no band for the projected close.
- Covariate models cannot be served from the command line. Training and validation accept
  `--covariate`, but today's covariate value has to come from a Python `covariate_provider`, so
  `pace` and `serve` refuse such models; use `PaceService` from the library.
- The Monte Carlo test depends on its assumptions (stable intraday shape, independent days, fixed
  dispersion) and is not a forecast of real error.
- Artifacts are pickles. Load only files your own training run wrote.
- The service and the artifact store each keep their own `TtlCache` with one lock, and there is no
  other concurrency control. The service is meant for a modest polling load.
- Whole-number rounding (`ceil`) is the default because the target is a count. For amounts use
  `postprocess="none"`.

## License and disclaimer

MIT, see [LICENSE](LICENSE). Built from problems I work on in debt collections and rewritten on synthetic data. It contains no
real data: all data in the tests, the documentation and the quick start is generated or invented.
