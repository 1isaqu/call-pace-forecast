# Data sources

This repository contains no data and no fitted model. This page lists where every row and every
number the code works on comes from, grouped into tiers by how far you can trust and reproduce
them. Last reviewed on 2026-10-13.

| Tier | What it is | In this repository |
|---|---|---|
| 0 | Invented data from a seeded generator, and invented rows written inside the tests | Yes: `src/call_pace/synthetic.py` and `tests/` |
| 1 | A public dataset downloaded at run time | One, fetched only on request: the "Anonymous Bank" call-center data. No test and no CI job uses it |
| 2 | Your own data, through a CSV file, the SQL Server adapter or your own loader | Never committed |

## Tier 0: invented data

`call-pace synth` runs the generator in `src/call_pace/synthetic.py`. `make_synthetic_hourly`
writes one row per (bucket, date, hour) for two buckets, `auto` and `consumer`, in the format of
[data-contract.md](data-contract.md). With `--amounts-out`, `make_synthetic_amounts` also writes
one row per counted event with a log-normal amount. The generator is seeded (`--seed`, default 7;
the amounts use the seed plus one), so the same arguments give the same rows. The README quick
start, the examples in the documentation and most tests run on it.

Every value is invented. The constants in the generator are arbitrary: the daily level of each
bucket, the weekday multipliers, the day-level spread, the negative-binomial dispersion, the
two-peak hourly curve, the optional growth and covariate effect, and the typical amount per bucket.
None comes from a public source. None was fitted to, tuned against or compared with results from
a real system, and the output is not meant to resemble any real operation.

The generator plants two things on purpose so the models have something to find:

- a day-level multiplier, which makes the volume so far informative about the rest of the day, and
- one high-volume bucket and one sparse bucket, because the models behave differently on the two.

Metrics computed on this data describe the generator: they show that the pipeline runs and that
the walk-forward and promotion rules work, and they say nothing about how well any model would do
on your data.

The Monte Carlo test (`call-pace montecarlo`) builds its scenarios from the shape of the history
you give it. On the quick start that history is tier 0 data, and the scenario parameters in
`MonteCarloConfig` (growth range, dispersion, window, number of months) are arbitrary as well.

The unit tests use a second kind of tier 0 data: small frames written inline in the test modules,
and frames that `tests/conftest.py` draws from the same generator with other seeds. There are no
fixture files: `git ls-files tests` lists only Python modules. CI runs on tier 0 only. No test
downloads anything or opens a network connection, and no test connects to a database.

## Tier 1: public datasets

One public dataset is supported, and the code downloads it only when you call
`call_pace.anonymous_bank.fetch` or run `scripts/evaluate_anonymous_bank.py`: the "Anonymous Bank"
call-center data of 1999, published by the Technion SEE Laboratory
(<https://seelab.net.technion.ac.il/data/>). Its terms, checked on 2026-10-13: the data is free for
use, users are asked to notify Avi Mandelbaum and to acknowledge the source, and there is no formal
licence. The repository does not contain the data, and the hash of every file is pinned in the code.
The dataset can test a bank call center with a weekday calendar and six queues of very different
volume. It cannot test outbound calling, collections or any recent operation. The run and its
results are in [real-data-evaluation.md](real-data-evaluation.md).

No test and no CI job uses it. If you add another public dataset, document its source, its licence,
the date you checked the terms and what it cannot test.

## Tier 2: your own data

Connect your own history as described in the
[README](../README.md#connect-your-own-data) and in [data-contract.md](data-contract.md). The rules:

- Provide one row per (bucket, date, hour) with a non-negative count, as a CSV file, through the
  SQL Server adapter or through a loader you write. The adapter reads connection settings from
  environment variables. `.env` is ignored by git and `.env.example` holds placeholders only.
- Use a login that can only read. The adapter only sends `SELECT` statements and opens the
  connection read-only, but it has not been run against a live server in this repository.
- Run `call-pace validate` before trusting any model, and compare the hourly counts for one day by
  hand against your source system.
- `data/`, `artifacts/` and the usual data and model file extensions (`*.csv`, `*.parquet`,
  `*.pkl`, `*.joblib`) are ignored by git. The files `train` writes are fitted on your data, so
  treat them as your data and do not commit them.

## Where the constants come from

Every default in `src/call_pace/config.py` is arbitrary and was picked so the synthetic example
runs: the operating hours, the lookup window, the k grid, the fold sizes, the quantiles, the
LightGBM settings and the model cache lifetime. The same holds for the promotion thresholds and
fold tolerances in `PromotionCriteria` (`validation.py`), the scenario parameters in
`MonteCarloConfig`, the 30 second data cache in `PaceService` and the status ids of the SQL Server
adapter. None comes from a public source and none was calibrated on data. Calibrate them on your
own history, and do not read a number the tool prints with the defaults as a finding.
