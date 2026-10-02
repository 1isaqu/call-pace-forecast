# Architecture

## The product in one table

For every hour of the operating day the system shows three numbers:

| expected | actual | delta |
|---|---|---|
| what the model says this hour should produce | what it produced (known when the hour closes) | actual minus expected |

The triad is also the explanation: a manager sees the reasoning without reading a model. Models
only make `expected` more accurate. The shape of the answer never changes, which is why the
forecasting method can be upgraded in phases without touching any consumer.

## Three phases

| Phase | Model | Reacts to today? | Role |
|---|---|---|---|
| 1 | `LookupForecaster`: median count per (bucket, hour) over a trailing window | no | Baseline. Nothing to tune except the window. Every other model has to beat it. |
| 2 | `KnnForecaster`: standardised KNN over hour, weekday (cyclic), volume so far, early-window total, bucket | yes | Finds past situations that look like the current one. |
| 3 | `QuantileForecaster`: LightGBM with a Poisson centre and lower/upper quantile models | yes | Same inputs, plus a band of plausible counts. |

A model is promoted only through the walk-forward protocol in [evaluation.md](evaluation.md).

## Data flow

```
                         TRAINING / EVALUATION (offline)
 loader --> validate_hourly --> FeatureSpec + build_frame --+--> run_walk_forward --> promotion decision
 (CSV, SQL, yours)   (contract)     (one code path)         +--> feature_subset_search, lookup variants
                                                            +--> run_monte_carlo (retraining stress test)
                                                            +--> train_models --> artifacts/ (+ manifest)

                              SERVING (online, per request)
 request --> PaceService.pace(bucket)
               1. resolve bucket against manifest          (unknown bucket: InvalidRequestError)
               2. outside operating hours?  -> empty payload, no models, no data
               3. load models routed to the bucket(s)       (missing model: ArtifactError)
               4. loader.load_hourly(today, today)          (failure: degraded payload)
               5. serving_frame -> model -> expected per hour
               6. build_bands -> aggregate_bands -> summarize
               7. optional value bands: expected x ticket
           --> {"meta", "data", "errors"}
```

## Modules

| Module | Responsibility |
|---|---|
| `config` | One frozen `Config`: calendar, post-processing, window and cut-offs, k grid, quantiles, fold sizes, routing. All defaults are invented. `from_env()` reads `CALL_PACE_*`. |
| `contract` | The data contract, `validate_hourly` / `validate_amounts`, the loader protocols. |
| `features` | `FeatureSpec` (persisted with each model), `build_frame` (training) and `serving_frame` (one day, as known now). Both paths use the same helpers. |
| `baseline`, `knn`, `quantile` | The three phases. Each has `fit`, `predict`, `save`, `load`. |
| `projection` | End-of-day total: share formula versus neighbour models. |
| `value` | Ticket lookup, `expected_value`, honest configuration search. |
| `metrics`, `validation` | Post-processing alignment, MAE, directional accuracy, coverage, WAPE, folds, walk-forward, promotion rules, paired test. |
| `ablation` | Feature subsets, lookup variants, covariate signal report. |
| `montecarlo`, `synthetic` | Retraining stress test; invented data generator. |
| `pace` | Pure band logic: statuses, aggregation, projected close. No I/O, no models. |
| `service`, `api` | Orchestration with a failure policy; optional FastAPI app. |
| `artifacts`, `persistence`, `training` | Runtime model files, manifest, TTL cache, `train_models`. |
| `loaders/*` | In-memory, CSV and SQL Server loaders. |
| `cli` | `call-pace` commands. |

## Design properties worth knowing

- **One feature path.** Training builds features from a whole history; serving builds them for
  one day at the current hour. Both use the same helpers, and a test replays a past day hour by
  hour and checks that the served features equal the training features.
- **Leakage guards.** `cumulative_lag` excludes the hour being predicted. `first_window_total` is
  zero until the first hours of the day have closed.
- **Models are routed per bucket.** `Config.routing` (env `CALL_PACE_ROUTING`) maps a bucket to
  `lookup`, `knn` or `quantile`; `default_model` covers the rest. A bucket whose lookup is zero in
  nearly every hour cannot flag a shortfall, so `LookupForecaster.is_informative` exists as the
  review trigger for choosing a model that reacts to the day so far.
- **Failure policy** (`service.py`). A bad request always propagates. A missing model is an error.
  Any failure while reading today's data returns a well-formed payload with empty bands and a
  message in `errors`, so a display that polls the endpoint does not go blank.
- **Caching.** Today's data is cached for `data_ttl_seconds` (default 30) to protect the source
  from polling bursts; loaded models are cached for `artifact_ttl_seconds` (default 24 hours).
  Both use one lock-protected `TtlCache`, so concurrent requests trigger one load.
- **Artifacts live at runtime only.** `train` writes `artifacts/` (gitignored): `lookup.json`,
  `knn.joblib`, `quantile.joblib`, `ticket.json`, `manifest.json`. Nothing fitted is committed.
  `joblib` is pickle based, so load only artifacts you produced yourself.
- **Future bands are conditioned, not rolled forward.** Expected counts for hours that have not
  started use the cumulative reached so far; they are not recomputed from earlier predictions.
- **No interval on the projected close.** The close is `actual so far + expected for the open
  hours`, a point estimate. Summing per-hour quantile bounds does not give a bound on the sum, so
  the library does not report one, and `aggregate_bands` drops bounds when it adds buckets.

## Extension points

- New data source: implement `load_hourly` (and optionally `load_amounts`).
- New model: provide `fit(frame)` and `predict(frame) -> array` (post-processed expected counts per
  row of the modelling frame); `run_walk_forward` is written against the three built-ins, but the
  baseline and metrics functions are independent of the model class.
- New covariate: add a numeric, bucket-day-constant column and list it in `Config.covariates`;
  use `ablate covariate` first to find out whether it carries any signal.
