from datetime import date

import numpy as np
import pandas as pd
import pytest

from call_pace.errors import ArtifactError, ConfigError, InsufficientDataError
from call_pace.synthetic import make_synthetic_amounts
from call_pace.value import (
    TicketGrid,
    TicketLookup,
    amounts_to_counts,
    estimate_tickets,
    expected_value,
    observed_value_by_hour,
    select_ticket_config,
    ticket_estimate,
)

HOURS = list(range(8, 20))


def events(*items):
    df = pd.DataFrame(items, columns=["bucket", "date", "hour", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_median_and_winsorised_mean_by_hand():
    values = [1, 2, 3, 4, 1000]
    assert ticket_estimate(values, "median") == 3
    # Clipping to the 0th..75th percentile turns 1000 into 4: mean of [1, 2, 3, 4, 4].
    assert ticket_estimate(values, "winsor", (0, 75)) == pytest.approx(2.8)
    assert ticket_estimate(values, "winsor") < np.mean(values)


def test_ticket_estimate_rejects_empty_input_and_unknown_estimators():
    with pytest.raises(InsufficientDataError):
        ticket_estimate([])
    with pytest.raises(ConfigError):
        ticket_estimate([1.0], "mode")


def test_per_hour_tickets_need_enough_observations_else_use_the_bucket_value():
    window = events(
        *[("a", "2026-03-02", 9, 10.0)] * 5,  # five events at 9h: enough for its own cell
        ("a", "2026-03-02", 10, 1000.0),  # a single event at 10h: too sparse
    )
    table = estimate_tickets(window, "median", "hour_bucket", min_cell_obs=5)
    assert table.hours == {9: 10.0}
    assert table.for_hour(9) == 10.0
    assert (
        table.for_hour(10) == table.overall == 10.0
    )  # the 10h cell fell back to the bucket median
    assert estimate_tickets(window, "median", "bucket", 5).hours == {}


def test_an_empty_window_is_an_error_unless_a_backtest_asks_for_zero():
    empty = events().astype({"hour": "int64"})
    with pytest.raises(InsufficientDataError, match="invent"):
        estimate_tickets(empty, "median", "bucket", 5)
    assert estimate_tickets(empty, "median", "bucket", 5, allow_empty=True).overall == 0.0
    with pytest.raises(ConfigError):
        estimate_tickets(empty, "median", "daily", 5)


def test_silent_days_still_age_the_window():
    data = events(("a", "2026-03-02", 9, 100.0), ("a", "2026-03-06", 9, 10.0))  # Monday, Friday
    # A 2-day window is Thursday and Friday: only Friday has events.
    assert TicketLookup(window_days=2).fit(data).ticket_for("a", 9) == 10.0
    # A 5-day window is the whole week: the median of both events.
    assert TicketLookup(window_days=5).fit(data).ticket_for("a", 9) == 55.0


def test_a_window_with_no_events_refuses_to_invent_a_ticket():
    # Friday's events are cut by the maturity cutoff, leaving only silent days.
    data = events(("a", "2026-03-02", 9, 100.0), ("a", "2026-03-06", 9, 10.0))
    cutoff_all = TicketLookup(window_days=2, maturity_cutoff_days={"a": 2})
    with pytest.raises(InsufficientDataError, match="invent"):
        cutoff_all.fit(data)
    everything_cut = TicketLookup(window_days=2, maturity_cutoff_days={"a": 9})
    with pytest.raises(InsufficientDataError, match="usable days"):
        everything_cut.fit(data)


def test_maturity_cutoff_drops_the_recent_days_of_that_bucket():
    data = events(
        ("a", "2026-03-02", 9, 100.0),
        ("a", "2026-03-03", 9, 100.0),
        ("a", "2026-03-04", 9, 9999.0),
        ("b", "2026-03-02", 9, 5.0),
        ("b", "2026-03-04", 9, 7.0),
    )
    model = TicketLookup(window_days=10, maturity_cutoff_days={"a": 1}).fit(data)
    assert model.ticket_for("a", 9) == 100.0
    assert model.trained["a"]["to"] == "2026-03-03"
    assert model.ticket_for("b", 9) == 6.0  # untouched: median of 5 and 7


def test_ticket_for_rounds_to_cents_and_rejects_unknown_buckets():
    model = TicketLookup().fit(
        events(("a", "2026-03-02", 9, 10.126), ("a", "2026-03-02", 9, 10.126))
    )
    assert model.ticket_for("a", 17) == 10.13
    with pytest.raises(ArtifactError, match="bucket"):
        model.ticket_for("zzz", 9)


def test_constructor_validates_its_arguments():
    for bad in (
        {"estimator": "mode"},
        {"granularity": "daily"},
        {"window_days": 0},
        {"min_cell_obs": 0},
    ):
        with pytest.raises(ConfigError):
            TicketLookup(**bad)


def test_json_roundtrip_keeps_per_hour_tickets(tmp_path):
    rows = [("a", "2026-03-02", 9, 10.0 + i) for i in range(6)] + [("a", "2026-03-02", 10, 99.0)]
    model = TicketLookup(granularity="hour_bucket", min_cell_obs=3).fit(events(*rows))
    loaded = TicketLookup.load(model.save(tmp_path / "t" / "ticket.json"))
    assert loaded.ticket_for("a", 9) == model.ticket_for("a", 9)
    assert loaded.ticket_for("a", 10) == model.ticket_for("a", 10)
    assert (loaded.granularity, loaded.min_cell_obs) == ("hour_bucket", 3)
    with pytest.raises(ArtifactError):
        TicketLookup.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"model": "lookup"}', encoding="utf-8")
    with pytest.raises(ArtifactError):
        TicketLookup.load(bad)


def test_expected_value_is_count_times_ticket_to_cents():
    assert expected_value([2, 0, 3], [10.005, 50.0, 1.1]) == [20.01, 0.0, 3.3]
    with pytest.raises(ValueError):
        expected_value([1, 2], [1.0])


def test_event_rows_collapse_to_counts_and_to_observed_value():
    data = events(
        ("a", "2026-03-02", 9, 10.0),
        ("a", "2026-03-02", 9, 5.0),
        ("a", "2026-03-02", 10, 1.0),
        ("b", "2026-03-03", 9, 7.0),
    )
    counts = amounts_to_counts(data)
    assert counts.set_index(["bucket", "hour"])["count"].to_dict() == {
        ("a", 9): 2.0,
        ("a", 10): 1.0,
        ("b", 9): 1.0,
    }
    assert observed_value_by_hour(data, date(2026, 3, 2)) == {"a": {9: 15.0, 10: 1.0}}
    assert observed_value_by_hour(data, date(2026, 3, 9)) == {}


# --- configuration search ------------------------------------------------------------------


@pytest.fixture(scope="module")
def amounts(hourly):
    return make_synthetic_amounts(hourly, seed=5)


SMALL = {"n_folds": 3, "fold_size_days": 5, "min_train_days": 20, "min_cell_obs": 3}
GRID = TicketGrid(windows=(30, 15))


def test_selection_scores_every_candidate_on_every_fold_for_every_bucket(amounts):
    outcome = select_ticket_config(amounts, HOURS, grid=GRID, **SMALL)
    n_candidates = len(GRID.candidates())
    assert len(outcome.results) == 2 * n_candidates * 3
    assert set(outcome.winners) == {"auto", "consumer"}
    assert len(outcome.holdout) == 2 and set(outcome.holdout["fold"]) == {3}
    assert len(outcome.baselines) == 2 * 3
    assert {"mae_band", "me_band", "wape_day"} <= set(outcome.results.columns)
    assert {"mae_band_baseline_a", "wape_day_baseline_b"} <= set(outcome.baselines.columns)


def test_the_most_parsimonious_candidate_wins_a_tie():
    flat = events(
        *[("a", d.strftime("%Y-%m-%d"), 9, 100.0) for d in pd.bdate_range("2026-01-05", periods=60)]
    )
    outcome = select_ticket_config(flat, [9], grid=GRID, **SMALL)
    winner = outcome.winners["a"]
    assert (winner.estimator, winner.granularity, winner.window_days) == ("median", "bucket", 30)
    assert outcome.results["mae_band"].eq(0.0).all()


def test_the_holdout_fold_never_influences_the_choice(amounts):
    before = select_ticket_config(amounts, HOURS, grid=GRID, **SMALL).winners
    last_days = sorted(amounts["date"].unique())[-5:]
    distorted = amounts.copy()
    distorted.loc[distorted["date"].isin(last_days), "amount"] *= 500.0
    after = select_ticket_config(distorted, HOURS, grid=GRID, **SMALL).winners
    assert before == after


def test_a_maturity_cutoff_shifts_the_evaluation_calendar(amounts):
    plain = select_ticket_config(amounts, HOURS, grid=GRID, **SMALL)
    cut = select_ticket_config(amounts, HOURS, grid=GRID, maturity_cutoff_days={"auto": 7}, **SMALL)
    plain_auto = plain.baselines[plain.baselines["bucket"] == "auto"]["mae_band"].tolist()
    cut_auto = cut.baselines[cut.baselines["bucket"] == "auto"]["mae_band"].tolist()
    assert plain_auto != cut_auto

    def consumer(frame):
        return frame[frame["bucket"] == "consumer"]["mae_band"].tolist()

    assert consumer(plain.baselines) == consumer(cut.baselines)
