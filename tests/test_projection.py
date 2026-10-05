import numpy as np
import pandas as pd
import pytest

from call_pace.config import Config
from call_pace.errors import ConfigError, InsufficientDataError
from call_pace.features import FeatureSpec, build_frame
from call_pace.projection import (
    NeighborProjector,
    ProportionProjector,
    cell_sparsity,
    compare_share_keys,
    evaluate_projection,
)

CFG = Config()


def flat_days(totals, bucket="a"):
    """Each day spreads its total evenly over 12 hours, so the share curve is identical daily."""
    rows = []
    days = pd.bdate_range("2026-01-05", periods=len(totals))
    for day, total in zip(days, totals, strict=True):
        rows += [(bucket, day, h, total / 12.0) for h in range(8, 20)]
    data = pd.DataFrame(rows, columns=["bucket", "date", "hour", "count"])
    return build_frame(data, FeatureSpec.from_config(CFG, [bucket]))


def test_projection_inverts_the_share_identity_exactly_on_a_regular_curve():
    frame = flat_days([12, 24, 36, 48] * 5)
    projector = ProportionProjector().fit(frame)
    projected = projector.predict(frame)
    assert np.allclose(projected, frame["day_total"])
    assert projector.share_for("a", 8) == pytest.approx(1 / 12)
    assert projector.project_one("a", 19, 40.0) == pytest.approx(40.0)  # the day is over


def test_hours_with_a_zero_typical_share_have_no_projection():
    frame = flat_days([12, 24])
    frame.loc[frame["hour"] == 8, "share"] = 0.0
    projector = ProportionProjector().fit(frame)
    assert np.isnan(projector.project_one("a", 8, 5.0))
    assert np.isnan(projector.project_one("a", 3, 5.0))  # an hour never seen at all


def test_an_unseen_bucket_borrows_the_pooled_curve():
    projector = ProportionProjector().fit(flat_days([12, 24]))
    assert projector.project_one("new", 9, 4.0) == pytest.approx(4.0 / (2 / 12))


def test_trailing_window_ignores_old_days():
    totals = [12] * 6 + [24] * 3
    full = ProportionProjector().fit(flat_days(totals))
    recent = ProportionProjector(window_days=3).fit(flat_days(totals))
    assert full.share_for("a", 8) == pytest.approx(1 / 12)
    assert recent.share_for("a", 8) == pytest.approx(1 / 12)  # shares are identical by design
    assert recent.shares and len(recent.shares) == 12


def test_silent_days_cannot_teach_a_share_curve():
    with pytest.raises(InsufficientDataError):
        ProportionProjector().fit(flat_days([0, 0, 0]))


@pytest.mark.parametrize("mode", ["hour_feature", "hour_partition"])
def test_neighbors_recover_the_total_of_the_most_similar_days(mode):
    frame = flat_days([12, 36] * 8)
    train = frame[frame["date"] < frame["date"].unique()[12]]
    test = frame[frame["date"] >= frame["date"].unique()[12]]
    model = NeighborProjector(k=3, mode=mode).fit(train)
    assert np.allclose(model.predict(test), test["day_total"])


def test_partition_mode_falls_back_to_the_formula_when_a_cell_is_too_small():
    frame = flat_days([12, 24, 36])  # only three days per (bucket, hour) cell
    neighbors = NeighborProjector(k=10, mode="hour_partition").fit(frame)
    formula = ProportionProjector().fit(frame)
    assert np.allclose(neighbors.predict(frame), formula.predict(frame))


def test_queries_outside_the_training_range_are_clipped_and_counted():
    frame = flat_days([12, 24] * 6)
    model = NeighborProjector(k=3, mode="hour_partition").fit(frame)
    probe = frame.iloc[:2].copy()
    probe["cumulative"] = [1e6, -5.0]
    model.predict(probe)
    assert model.ood_count == 2
    model.predict(frame.iloc[:2])
    assert model.ood_count == 0  # the counter describes the latest call only


def test_buckets_are_never_mixed():
    low, high = flat_days([12] * 8, "a"), flat_days([120] * 8, "b")
    both = pd.concat([low, high], ignore_index=True)
    model = NeighborProjector(k=3, mode="hour_feature").fit(both)
    assert np.allclose(model.predict(both), both["day_total"])


def test_invalid_arguments_and_unfitted_use_are_rejected():
    with pytest.raises(ConfigError):
        NeighborProjector(mode="nearest")
    with pytest.raises(ConfigError):
        NeighborProjector(k=0)
    with pytest.raises(InsufficientDataError):
        NeighborProjector().predict(flat_days([12]))
    with pytest.raises(InsufficientDataError):
        NeighborProjector().fit(flat_days([0, 0]))


def test_evaluation_shows_the_formula_winning_when_no_past_day_matches_exactly():
    rng = np.random.default_rng(4)
    totals = list(rng.permutation([12 * i for i in range(1, 41)]))  # every total is unique
    frame = flat_days(totals)
    table = evaluate_projection(frame, holdout_days=8, k=3)
    assert list(table["hour"]) == [str(h) for h in range(8, 20)] + ["all"]
    overall = table.iloc[-1]
    assert overall["mae_proportion"] == pytest.approx(0.0)
    assert overall["mae_hour_feature"] > 1.0 and overall["mae_hour_partition"] > 1.0
    assert overall["n"] == table["n"].iloc[:-1].sum()


def test_evaluation_scores_every_projector_on_the_same_rows(frame):
    table = evaluate_projection(frame, holdout_days=15, k=5)
    assert (
        np.isfinite(table[["mae_proportion", "mae_hour_feature", "mae_hour_partition"]]).all().all()
    )
    assert table["n"].iloc[-1] > 0


def test_evaluation_needs_more_history_than_the_holdout():
    with pytest.raises(InsufficientDataError):
        evaluate_projection(flat_days([12, 24, 36]), holdout_days=3)


# --- finer keys for the share curve ---------------------------------------------------------


def monday_morning_days(weeks):
    """Mondays put their volume in the morning, other days in the afternoon.

    A share curve pooled over weekdays describes the other four days, so it projects Mondays
    wrongly; a curve keyed by weekday is exact. Every day has the same total.
    """
    rows = []
    for day in pd.bdate_range("2026-01-05", periods=5 * weeks):
        morning = day.weekday() == 0
        for hour in range(8, 20):
            early = hour < 14
            rows.append(("a", day, hour, 10.0 if early == morning else 2.0))
    data = pd.DataFrame(rows, columns=["bucket", "date", "hour", "count"])
    return build_frame(data, FeatureSpec.from_config(CFG, ["a"]))


def test_sparsity_counts_cells_and_treats_absent_cells_as_thin():
    frame = monday_morning_days(5)  # every (hour, weekday) cell has five rows
    keys = ["bucket", "hour", "weekday_pos"]
    full = cell_sparsity(frame, keys, min_obs=5)
    assert (full.cells_possible, full.cells_present, full.cells_thin) == (60, 60, 0)
    assert full.median_obs == 5.0 and full.thin_share == 0.0
    gap = frame[~((frame["hour"] == 8) & (frame["weekday_pos"] == 0))]
    assert cell_sparsity(gap, keys, min_obs=5).cells_thin == 1  # the absent cell
    # With a higher bar the 59 present cells are short as well, plus the absent one.
    assert cell_sparsity(gap, keys, min_obs=6).cells_thin == 60


def test_sparsity_needs_real_columns_and_handles_an_empty_frame():
    with pytest.raises(ConfigError):
        cell_sparsity(monday_morning_days(1), ["bucket", "nope"])
    empty = cell_sparsity(monday_morning_days(1).iloc[:0], ["bucket", "hour"])
    assert empty.cells_possible == 0 and empty.thin_share == 0.0


def test_a_finer_key_is_adopted_when_it_helps_and_its_cells_are_populated():
    frame = monday_morning_days(12)
    result = compare_share_keys(frame, ("bucket", "hour", "weekday_pos"), holdout_days=10)
    assert result.mae_fine == pytest.approx(0.0) and result.mae_coarse > 1.0
    assert result.fallbacks == 0 and result.sparsity.cells_thin == 0
    assert result.paired.mean_difference < 0  # negative favours the finer key
    assert result.adopt_fine and result.explain().startswith("ADOPT bucket+hour+weekday_pos")


def test_a_finer_key_is_refused_when_it_does_not_help():
    frame = flat_days([12, 24, 36, 48] * 15)  # one regular curve: nothing for a weekday to add
    result = compare_share_keys(frame, ("bucket", "hour", "weekday_pos"), holdout_days=10)
    assert result.gain_pct == 0.0 and not result.adopt_fine
    assert "gain below the 5% needed" in result.explain()


def test_a_finer_key_that_helps_is_still_refused_when_too_many_cells_are_thin():
    frame = monday_morning_days(12)
    result = compare_share_keys(
        frame, ("bucket", "hour", "weekday_pos"), holdout_days=10, min_obs=500
    )
    assert result.gain_pct > 5.0 and result.sparsity.thin_share == 1.0
    assert not result.adopt_fine and "cells are thin" in result.explain()


def test_missing_fine_cells_fall_back_to_the_coarser_curve_and_are_counted():
    frame = monday_morning_days(12)
    dates = sorted(frame["date"].unique())
    # A weekday value that never occurs in the training days:
    frame.loc[frame["date"].isin(dates[-10:]), "weekday_pos"] = 9
    result = compare_share_keys(frame, ("bucket", "hour", "weekday_pos"), holdout_days=10)
    assert result.fallbacks == result.n > 0
    assert result.mae_fine == pytest.approx(result.mae_coarse)  # a fallback is the coarse answer


def test_the_fine_key_must_refine_the_coarse_key_and_exist():
    frame = monday_morning_days(12)
    with pytest.raises(ConfigError):
        compare_share_keys(frame, ("bucket", "hour"))
    with pytest.raises(ConfigError):
        compare_share_keys(frame, ("bucket", "weekday_pos"))  # drops a coarse key
    with pytest.raises(ConfigError):
        compare_share_keys(frame, ("bucket", "hour", "nope"))
    with pytest.raises(InsufficientDataError):
        compare_share_keys(monday_morning_days(1), ("bucket", "hour", "weekday_pos"))
