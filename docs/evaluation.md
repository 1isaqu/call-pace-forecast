# Evaluation

How models are compared, how a promotion is decided, what the Monte Carlo test does and how to
read the quantile bands. All thresholds mentioned are defaults chosen for the synthetic example:
calibrate them on your own data.

## 1. Walk-forward validation

Time series must never be shuffled; a model trained on Thursday must not be tested on Tuesday.
`validation.make_folds` cuts the last `n_folds * fold_size_days` operating days into consecutive
test blocks. Each block is predicted by a model fitted only on the days before it (an expanding
window).

```
days:   [ ............ train ............ ][ test 1 ][ test 2 ][ test 3 ][ test 4 ]
fold 1: train [0, A)                          test [A, A+s)
fold 2: train [0, A+s)                                       test [A+s, A+2s)
fold 3: train [0, A+2s)                                                  test [A+2s, A+3s)
fold 4: train [0, A+3s)                                                              test final s days
```

Rules the code enforces and tests:

1. No fold trains on a date later than or equal to a date it is tested on.
2. k is selected on folds 1 to n-1 only (`knn_cross_validate`). The last fold never influences a
   choice: a test changes only the last fold's data and checks that the ranking in
   `feature_subset_search` is identical.
3. The last fold is the holdout. It confirms; it does not select.
4. The model that is deployed (`train_models`) is refitted on all data.
5. The run needs `n_folds * fold_size_days + min_train_days` operating days, otherwise
   `InsufficientDataError` says how many are needed.
6. Runs are deterministic (fixed seeds, deterministic LightGBM settings); a test compares two runs.

The lookup baseline is refitted inside every fold, using its own trailing window
(`lookup_window_days`) and maturity cut-offs.

## 2. Metrics

All metrics are computed after the serving post-processing (`ceil` by default, then clipped at
zero). A model that looks better on raw floats can look worse once its output is shown as a whole
number, so offline and online must use the same rounding.

| Metric | Definition | Notes |
|---|---|---|
| MAE | mean of `abs(actual - predicted)` per hour | Primary metric. It does not divide by the actual, so it stays defined for hours with zero events, unlike MAPE. |
| Directional accuracy | share of hours where `sign(model - baseline) == sign(actual - baseline)` | Hours where the actual equals the baseline carry no direction and are dropped. It answers: when the model moves away from the baseline, is it moving the way reality moves? |
| Coverage | share of actual values inside `[lower, upper]` | Compare to the nominal coverage, `upper quantile - lower quantile`. |
| Band width | mean of `upper - lower` | A band that is too wide is "calibrated" and useless. Read together with coverage. |
| WAPE | total absolute error over total actual | Used for daily totals of value. |
| Paired test | t-test on per-row absolute errors of two models on the same rows | Hourly errors are correlated within a day, so treat the p-value as a rough guide. |

## 3. Promotion rules

`decide_promotion(result, challenger, PromotionCriteria())` returns a decision with one line per
check. A challenger is promoted only if every check passes.

| Check | Rule | Default |
|---|---|---|
| improvement | holdout MAE at least this much below the baseline's | 5 percent |
| directional accuracy | holdout directional accuracy at least | 0.55 |
| no fold regression | no fold is clearly worse than the baseline (see below) | abs 0.05 and rel 5 percent |
| max MAE (optional) | holdout MAE below an absolute cap | off |

A noisy fold is not a regression. A fold counts as a regression only if the challenger is worse
than the baseline by more than the absolute tolerance AND by more than the relative tolerance. Two
failure modes motivated this: on a low-volume fold the MAE is small, so a harmless absolute
difference is a large relative one; on a high-volume fold the reverse. Requiring both avoids
failing on either artefact.

Prefer thresholds relative to the baseline (as here) over a fixed absolute MAE. An absolute
limit calibrated for one series is wrong for another, and the same code serves series of very
different volume. The optional `max_mae` exists for a hard business limit.

Decide per bucket. Pooled metrics can hide a model that wins on a high-volume bucket and loses
on a sparse one. `result.for_bucket("name")` returns the same run restricted to one bucket and
`call-pace validate --per-bucket` prints a decision for each. Then set `Config.routing`.

## 4. Ablation

- `feature_subset_search`: every subset of the features (the hour is required by default) crossed
  with the k grid. Ranking: mean MAE on folds 1 to n-1, then fewer features, then smaller k. The final fold
  confirms the winner once, next to the all-features model and the lookup.
- `compare_feature_sets`: MAE of named feature sets fold by fold, for a specific question such as
  "does adding this covariate help?".
- `compare_lookup_variants`: key (pooled hour versus hour and bucket) x statistic (median, mean) x
  outlier treatment (none, winsorised, capped at mean + 3 standard deviations, log scale).
- `covariate_signal_report`: after controlling for bucket, hour and weekday, is the covariate related
  to what is left? Unit of analysis is the bucket-day, because the covariate is constant within a day
  and the hours of one day move together. Pearson, Spearman and Kruskal-Wallis across quantile
  groups. It is a screening tool: a high p-value is a reason to remove the covariate, a low one is a
  reason to investigate.

## 5. End-of-day projection

`projection.evaluate_projection` compares, hour by hour, the share formula
(`cumulative / typical share of the day completed by this hour`) with two neighbour models (hour as a
distance dimension, or as an exact partition key). On data where every day follows the same share
curve the formula is exact by construction and any similarity model is, at best, equal. The
comparison is there so that a claim like "the learned model is better for the daily total" is checked
rather than assumed.

A second question is how finely the share curve should be keyed. `projection.compare_share_keys`
(CLI: `call-pace projection --extra-key weekday_pos`) learns the median share per `(bucket, hour)`
and per `(bucket, hour, extra key)` on the earlier days, projects the last `holdout_days` days with
both and reports the error of each, a paired test and the decision. A row whose finer cell was never
seen, or has a zero median share, falls back to the coarser curve, and the number of such rows is
printed. The finer key is
recommended only if two conditions hold together:

1. it lowers the held-out error by at least `min_gain_pct` (default 5 percent), and
2. at most `max_thin_share` (default 10 percent) of its cells are thin, where a cell is thin when it
   has fewer than `min_obs` training rows (default 5) or does not occur at all (`cell_sparsity`).

A gain with thin cells is refused with that reason: a median over three days is a guess, and the
gain is likely to be noise that will not repeat. Both thresholds are defaults chosen for the
synthetic example.

## 6. Monte Carlo stress test

`call-pace montecarlo` answers a different question from walk-forward: "if volume keeps growing and
we retrain every month, how does the error behave?"

Procedure, per scenario:

1. From history learn the hourly share profile and the pool of daily totals per (bucket, weekday).
2. Draw a monthly growth rate from a triangular distribution (`growth_low`, `growth_mode`,
   `growth_high`).
3. Build the next months: sample a daily total from the pool, scale it by the accumulated growth,
   spread it with the profile, add negative-binomial noise (variance = mean + mean^2 / r).
4. After each simulated month refit on a rolling window of the latest days (real tail plus
   simulated) and score the next month against the lookup baseline.

Output: percentiles of MAE, baseline MAE and directional accuracy per month, the probability that
directional accuracy exceeds a threshold in the last month and the probability that the model beats
the lookup. Scenario `s` uses the random stream `(seed, s)`, so any single scenario can be
reproduced alone.

What it does not tell you: it is not a forecast of real future error. It assumes a stable
intraday and weekly shape, independent days and a fixed dispersion. It measures sensitivity to
growth under those assumptions. All its parameters are defaults chosen for the synthetic example.

## 7. Reading the quantile bands

`QuantileForecaster` fits a lower and an upper quantile (default 0.25 and 0.75) next to the
Poisson centre. Each served hour carries `lower`, `upper`, and, once the hour has closed,
`position`: `below_band`, `within_band` or `above_band`.

- A band is a range of plausible hourly counts, not a guarantee. With quantiles 0.25 and 0.75 the
  nominal coverage is 50 percent: about half of the hours should land inside. Whether that is true
  on unseen days is empirical. `validate` prints the nominal and the empirical coverage on the
  holdout fold, and the mean width. Coverage far above nominal means a band wider than it needs to
  be; far below means the model is overconfident. This library does not recalibrate the bands, so
  check coverage before you let anyone act on one.
- Counts are discrete and rounded up. After `ceil`, lower and upper are whole numbers, and a narrow
  band at low volume can be a single value.
- Quantile models can cross. They are fitted independently, so the lower model can exceed the upper
  for some inputs; `predict_bands` orders them so `lower <= upper`. The centre is not forced inside
  the band.
- When volume is low, read `position` instead of the sign of delta. A delta of -1 against an
  expectation of 3 is noise, and `within_band` says so. `below_band` is the signal worth a look.
- Bands exist only for one bucket. For `all`, bounds are dropped: the sum of per-bucket bounds is
  not a bound on the sum. The projected close has no band for the same reason.
- A recent change in volume level makes a band learned on the old level wrong in both directions.
  Prefer a trailing window (`lookup_window_days`) and re-run `validate` after a known change.

## 8. Reading the pace table

| Status | Meaning |
|---|---|
| `above` / `below` / `ok` | the hour has closed and actual is greater than, less than or equal to expected |
| `in_progress` | the hour running now; no verdict yet |
| `future` | the hour has not started |

Closed hours with no rows count as zero actual. If the data source returned no rows at all for the
day, even the current hour is reported as `future` (the day has not visibly started), and the closed
hours are scored against zero. Expected counts for future hours are conditioned on the cumulative
reached so far.
