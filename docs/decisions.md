# Key decisions

Each entry states the choice, the reason and, where it matters, what was rejected. The numbers
that appear in code are defaults chosen for the synthetic example, not results from any real
system.

## Modelling

### 1. Start from a lookup baseline and make everything else beat it

The expected count for an hour is the median of what that (bucket, hour) produced recently. There is
nothing to overfit and it is trivially explainable. Skipping it makes it impossible to say whether a
complicated model is worth its cost.

### 2. The hour is a partition key in the lookup

In the lookup the hour matches exactly. Putting it into a distance with several other features
dilutes it: the neighbours of a 10 o'clock row stop being 10 o'clock rows. The KNN does take the hour
as a feature (scaled like the others), but the end-of-day projection experiment compares both forms
(`NeighborProjector` modes `hour_partition` and `hour_feature`) so the effect can be measured.

### 3. The lookup uses the median

Daily volumes contain outlier days, and the median is barely moved by them. Alternatives
(winsorising, capping, log scale) are available through `central_estimate` and compared by
`compare_lookup_variants` instead of assumed.

### 4. Evaluate under the serving rounding

Offline metrics are computed after `ceil`-and-clip, the same transformation the display applies.
Comparing models on raw floats and then showing rounded numbers can reverse the ranking.

### 5. Features available at prediction time, nothing else

`cumulative_lag` is the total before the hour being predicted. `first_window_total` (sum of the first
hours of the day) is zero until those hours have closed. Violating either would make offline results
look better than anything achievable live.

### 6. Standardise inside the model, fit the scaler on training rows only

KNN distances are meaningless across features with different units. The scaler lives in the model's
pipeline so it cannot see test rows.

### 7. The share formula is the reference for the day total

Because `share = cumulative / total`, projecting the total by `cumulative / typical share` inverts
exactly the identity a neighbour model has to learn from past days. The library ships the formula
(`ProportionProjector`) as the reference and the neighbour models (`NeighborProjector`) as
challengers, and `call-pace projection` scores them per hour on held-out days. On the synthetic data
of the quick start the neighbour models are much better at 8 and 9 o'clock, where the
cumulative is small and dividing it by a small share amplifies its noise. From late morning on the
formula is better in most hours, and over the whole day the `hour_feature` neighbour model had the
lower MAE. Run the comparison on your own data before choosing.

### 8. Prefer a trailing window to all history

Volume regimes shift. A model fitted on the entire past can be worse than one fitted on the recent
past. `lookup_window_days` is a parameter (its default was chosen for the synthetic example) that
should be chosen by walk-forward on your own data.

### 9. Maturity cut-off for recent days

The newest days of a source can be incomplete because records arrive late. `maturity_cutoff_days`
drops the last N days of a bucket before fitting the lookup and the ticket. Without it the model
learns that recent days are quiet.

### 10. Route per bucket; a sparse bucket needs a model that reacts

If the median of a bucket is zero in almost every hour, its expected band is a flat zero that can
never signal a shortfall. `LookupForecaster.is_informative(bucket, min_nonzero_hours)` is the review
trigger. Per-bucket routing (`Config.routing`) lets a high-volume bucket use one model while a sparse
one uses another, and `validate --per-bucket` provides the evidence.

### 11. Value is count times a typical ticket

Amounts are heavy tailed; a learned amount model is dominated by a few large events. The decomposition
reuses whatever count model is deployed and adds a lookup of recent tickets (median or winsorised
mean, bucket or hour-by-bucket, trailing window, maturity cut-off, fallback to the bucket value for
sparse cells). With no events in the window it raises: an invented placeholder ticket would silently
become invented money on a dashboard. Configuration search uses the real event count so the ticket
error is isolated; its numbers are therefore a best case for the decomposition.

### 12. Test covariates before using them

A business-plausible covariate (days since the last bulk intake of accounts, say) adds a dimension to
every distance and may add nothing. `compare_feature_sets` and `covariate_signal_report` exist so that
the claim is tested; the signal report controls for weekday because cyclic covariates are easily
confounded with the weekly pattern. The same applies to a finer lookup key such as the weekday:
`compare_share_keys` accepts it only when the held-out gain is real and the cells it creates are not
mostly thin.

### 13. Quantile bands are uncalibrated by design

The library reports nominal versus empirical coverage instead of silently recalibrating. A band that
claims 50 percent and delivers 85 percent is mis-specified in a way a user must see.

## Evaluation

### 14. Walk-forward, expanding window, k chosen on early folds, last fold confirms

Shuffled cross-validation leaks the future into the past. Choosing k on the same fold that is reported
inflates the result. See [evaluation.md](evaluation.md).

### 15. A fold is a regression only if it is worse by both margins

The margins are an absolute and a relative tolerance, and any number of folds inside either one pass.
A purely relative rule fails on small-volume folds; a purely absolute one on large-volume ones.

### 16. Scale-relative thresholds

Promotion is expressed as improvement over the baseline, not as a fixed MAE, because the same code
serves series of very different volume.

### 17. Monte Carlo tests the retraining policy; it does not forecast error

It varies growth under stated assumptions to see how the retraining policy behaves, and says so.

## Serving

### 18. Degrade on data failure, fail on a bad request or a missing model

A wall display polling the endpoint must not turn into an error page because of a short database
outage. But a request for a bucket that does not exist is a bug in the caller, and a missing model
is a deployment problem; hiding either would be worse than failing.

### 19. The empty-day quirk is explicit

If the loader returns no rows at all, the current hour is reported as `future`, not `in_progress`:
nothing has been observed, so the day has not visibly started. A test in `tests/test_pace.py` pins
this behaviour.

### 20. Short-lived cache for today's data, long-lived for models

Polling displays can multiply load on the source; a 30 second cache absorbs it, at the price of data
that can be up to 30 seconds old. Models change once per training run: they are cached for a long TTL
but keyed on the file's modification time, so retraining is picked up without a restart. The service
and the artifact store each keep their own lock-protected `TtlCache`, and concurrent requests for the
same key trigger one load.

### 21. One feature path for training and serving

A forecast degrades without raising an error when serving computes a feature slightly differently
from training. The serving frame uses the same helpers as the training frame, with a test that replays
a past day hour by hour.

## Data access

### 22. Bound parameters everywhere

Nothing from configuration or callers is pasted into SQL text; a test checks it.

### 23. Business filters inside the ranking's derived table

The "keep the latest row per installment" step ranks rows. A filter applied after the ranking could
let a re-written row with a non-qualifying status win and hide the valid one, so every business filter
(status set, first installment, hours, excluded agents) is applied before `ROW_NUMBER()`.

### 24. `NOLOCK` is on by default and can be switched off

It keeps analytical reads from blocking writers and allows reading uncommitted rows. Pass
`use_nolock=False` to `SqlOptions` to turn it off.

### 25. Excluded agents are filtered in SQL, never afterwards

Post-filtering would require the row-level agent column in every result. `COALESCE` keeps rows whose
name is NULL, because `NULL NOT LIKE ...` is unknown and would otherwise drop them silently.

## Repository

### 26. The repository holds no data and no fitted artifacts

`.gitignore` excludes data files and model files; `train` writes into an ignored directory. The
quick start and many tests use data generated by `call_pace.synthetic`, and the other tests build
small invented frames inline. See [data-sources.md](data-sources.md).

### 27. Artifacts are pickles

`joblib` files can execute code when loaded. The store only loads from a directory you configure, and
the documentation says to load only what you trained yourself.
