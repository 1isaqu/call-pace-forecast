import numpy as np
import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.errors import ConfigError, InsufficientDataError
from call_pace.montecarlo import MonteCarloConfig, ScenarioGenerator, run_monte_carlo

SMALL = MonteCarloConfig(n_scenarios=3, n_months=3, days_per_month=10, window_days=40, k=5, seed=9)


@pytest.mark.parametrize(
    "changes",
    [
        {"growth_low": 0.1, "growth_mode": 0.05, "growth_high": 0.2},
        {"growth_low": 0.05, "growth_mode": 0.05, "growth_high": 0.05},
        {"n_scenarios": 0},
        {"window_days": 0},
        {"dispersion": 0.0},
    ],
)
def test_invalid_scenario_parameters_are_rejected(changes):
    with pytest.raises(ConfigError):
        MonteCarloConfig(**changes)


def test_generated_months_follow_the_calendar_and_the_historical_shape(frame, config):
    generator = ScenarioGenerator(frame, config, SMALL)
    last = frame["date"].max().date()
    month = generator.month(last, 1.0, np.random.default_rng(1))
    dates = sorted(month["date"].unique())
    assert len(dates) == SMALL.days_per_month
    assert dates[0].date() > last and all(d.weekday() < 5 for d in dates)
    assert len(month) == len(dates) * 2 * 12  # two buckets, twelve hours
    assert (month["count"] >= 0).all() and np.array_equal(month["count"], month["count"].round())
    shares = np.array(list(generator.profiles.values()))
    assert np.allclose(shares.sum(axis=1), 1.0)


def test_growth_scale_multiplies_expected_volume(frame, config):
    generator = ScenarioGenerator(frame, config, SMALL)
    last = frame["date"].max().date()
    base = generator.month(last, 1.0, np.random.default_rng(2))["count"].sum()
    doubled = generator.month(last, 2.0, np.random.default_rng(2))["count"].sum()
    assert 1.6 < doubled / base < 2.4


def test_covariates_are_carried_into_simulated_months(config):
    from call_pace.contract import validate_hourly
    from call_pace.features import FeatureSpec, build_frame
    from call_pace.synthetic import make_synthetic_hourly

    cfg = config.replace(covariates=("days_since_event",))
    raw = make_synthetic_hourly(n_days=40, seed=2, config=cfg, with_covariate=True)
    history = build_frame(
        validate_hourly(raw, cfg), FeatureSpec.from_data(validate_hourly(raw, cfg), cfg)
    )
    month = ScenarioGenerator(history, cfg, SMALL).month(
        history["date"].max().date(), 1.0, np.random.default_rng(0)
    )
    assert month["days_since_event"].notna().all()
    assert month.groupby("bucket")["days_since_event"].nunique().eq(1).all()


def test_a_run_has_one_row_per_scenario_and_month_with_sane_metrics(frame, config, spec):
    result = run_monte_carlo(frame, config, spec, SMALL)
    runs = result.runs
    assert len(runs) == SMALL.n_scenarios * SMALL.n_months
    assert sorted(runs["month"].unique()) == [1, 2, 3]
    assert (runs["mae"] >= 0).all() and (runs["mae_lookup"] >= 0).all()
    assert runs["dir_acc"].dropna().between(0, 1).all()


def test_same_seed_same_result_and_scenarios_are_independently_reproducible(frame, config, spec):
    first = run_monte_carlo(frame, config, spec, SMALL)
    again = run_monte_carlo(frame, config, spec, SMALL)
    pd.testing.assert_frame_equal(first.runs, again.runs)
    fewer = run_monte_carlo(
        frame, config, spec, MonteCarloConfig(**{**SMALL.__dict__, "n_scenarios": 2})
    )
    pd.testing.assert_frame_equal(
        fewer.runs, first.runs[first.runs["scenario"] < 2].reset_index(drop=True)
    )
    other = run_monte_carlo(frame, config, spec, MonteCarloConfig(**{**SMALL.__dict__, "seed": 10}))
    assert not np.allclose(other.runs["mae"], first.runs["mae"])


def test_summary_reports_percentiles_per_month(frame, config, spec):
    result = run_monte_carlo(frame, config, spec, SMALL)
    summary = result.summary(percentiles=(10, 50, 90))
    assert summary["month"].tolist() == [1, 2, 3]
    assert (summary["mae_p10"] <= summary["mae_p50"]).all()
    assert (summary["mae_p50"] <= summary["mae_p90"]).all()
    assert {"mae_lookup_p50", "dir_acc_p90"} <= set(summary.columns)


def test_probabilities_are_shares_of_scenarios(frame, config, spec):
    result = run_monte_carlo(frame, config, spec, SMALL)
    for value in (
        result.probability_directional_above(),
        result.probability_directional_above(threshold=0.0, month=1),
        result.probability_beats_baseline(),
        result.probability_beats_baseline(month=2),
    ):
        assert 0.0 <= value <= 1.0
    assert result.probability_directional_above(threshold=2.0) == 0.0
    with pytest.raises(ConfigError):
        result.probability_beats_baseline(month=99)


def test_stronger_growth_means_larger_absolute_errors_later(frame, config, spec):
    flat = MonteCarloConfig(
        **{**SMALL.__dict__, "growth_low": 0.0, "growth_mode": 0.0, "growth_high": 0.001}
    )
    steep = MonteCarloConfig(
        **{**SMALL.__dict__, "growth_low": 0.3, "growth_mode": 0.4, "growth_high": 0.5}
    )
    low = run_monte_carlo(frame, config, spec, flat).runs.query("month == 3")["mae_lookup"].mean()
    high = run_monte_carlo(frame, config, spec, steep).runs.query("month == 3")["mae_lookup"].mean()
    assert high > low


def test_progress_callback_sees_every_scenario(frame, config, spec):
    seen = []
    run_monte_carlo(
        frame, config, spec, SMALL, progress=lambda done, total: seen.append((done, total))
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_history_without_usable_days_is_rejected(frame, config, spec):
    with pytest.raises(InsufficientDataError):
        run_monte_carlo(frame.iloc[:0], Config(), spec, SMALL)
