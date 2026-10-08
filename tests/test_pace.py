import pytest

from call_pace.pace import (
    Band,
    aggregate_bands,
    band_position,
    build_bands,
    classify_delta,
    summarize,
)

HOURS = list(range(8, 20))
FIVE = [5.0] * 12


def by_hour(bands):
    return {b.hour: b for b in bands}


def test_status_follows_the_sign_of_delta():
    assert [classify_delta(d) for d in (3, -2, 0)] == ["above", "below", "ok"]


def test_closed_open_and_future_hours_at_ten_oclock():
    bands = by_hour(build_bands(FIVE, {8: 7, 9: 5}, current_hour=10, hours=HOURS))
    assert (bands[8].status, bands[8].delta, bands[8].cumulative) == ("above", 2, 7)
    assert (bands[9].status, bands[9].delta, bands[9].cumulative) == ("ok", 0, 12)
    assert bands[10].status == "in_progress" and bands[10].actual is None
    assert all(bands[h].status == "future" for h in range(11, 20))
    assert bands[11].delta is None and bands[11].cumulative is None


def test_the_first_hour_of_the_day_is_in_progress_and_nothing_is_closed():
    bands = by_hour(build_bands(FIVE, {7: 0}, current_hour=8, hours=HOURS))
    assert bands[8].status == "in_progress"
    assert all(bands[h].status == "future" for h in range(9, 20))


def test_the_last_hour_can_be_in_progress_after_a_full_day_on_pace():
    observed = {h: 5 for h in range(8, 19)}
    bands = by_hour(build_bands(FIVE, observed, current_hour=19, hours=HOURS))
    assert all(bands[h].status == "ok" for h in range(8, 19))
    assert bands[19].status == "in_progress"


def test_missing_hours_count_as_zero_once_closed():
    bands = by_hour(build_bands(FIVE, {8: 5}, current_hour=11, hours=HOURS))
    assert (bands[9].actual, bands[9].status) == (0.0, "below")
    assert bands[10].delta == -5.0 and bands[10].cumulative == 5.0


def test_with_no_data_at_all_the_current_hour_is_not_in_progress():
    """Documented quirk: an entirely empty observation means the day has not visibly started."""
    bands = by_hour(build_bands(FIVE, {}, current_hour=12, hours=HOURS))
    assert all(bands[h].status == "below" and bands[h].actual == 0.0 for h in range(8, 12))
    assert all(bands[h].status == "future" for h in range(12, 20))


def test_digits_round_money_deltas_and_running_totals():
    bands = build_bands([10.0] * 12, {8: 10.126, 9: 20.337}, 10, HOURS, digits=2)
    assert bands[0].delta == 0.13
    assert bands[1].cumulative == 30.46


def test_quantile_band_position_of_closed_hours():
    lower, upper = [3.0] * 12, [6.0] * 12
    bands = by_hour(build_bands(FIVE, {8: 1, 9: 3, 10: 5, 11: 6, 12: 9}, 13, HOURS, lower, upper))
    assert [bands[h].position for h in range(8, 13)] == [
        "below_band",
        "within_band",
        "within_band",
        "within_band",
        "above_band",
    ]
    assert bands[13].position is None  # open hours have no verdict yet
    assert (bands[13].lower, bands[13].upper) == (3.0, 6.0)


def test_band_position_includes_both_bounds():
    assert band_position(3, 3, 6) == "within_band"
    assert band_position(6, 3, 6) == "within_band"
    assert band_position(2.99, 3, 6) == "below_band"
    assert band_position(6.01, 3, 6) == "above_band"


def test_bands_serialise_to_plain_dicts():
    band = build_bands(FIVE, {8: 5}, 9, HOURS)[0]
    assert band.to_dict() == {
        "hour": 8,
        "expected": 5.0,
        "actual": 5.0,
        "delta": 0.0,
        "status": "ok",
        "cumulative": 5.0,
        "lower": None,
        "upper": None,
        "position": None,
    }


def test_aggregation_sums_hours_and_reclassifies_from_the_total():
    a = build_bands([5.0] * 12, {8: 9, 9: 1}, 10, HOURS)
    b = build_bands([3.0] * 12, {8: 0, 9: 3}, 10, HOURS)
    merged = by_hour(aggregate_bands([a, b]))
    assert (merged[8].expected, merged[8].actual, merged[8].delta) == (8.0, 9.0, 1.0)
    assert merged[8].status == "above"
    assert (merged[9].actual, merged[9].delta, merged[9].status) == (4.0, -4.0, "below")
    assert merged[9].cumulative == (9 + 1) + (0 + 3)
    assert merged[10].status == "in_progress" and merged[10].actual is None
    assert merged[11].status == "future"


def test_an_open_hour_is_in_progress_when_any_bucket_is_running_it():
    """A bucket with no rows today reports its current hour as future; the sum must not copy it."""
    silent = build_bands([2.0] * 12, {}, 11, HOURS)
    busy = build_bands([3.0] * 12, {8: 1, 9: 2, 10: 4}, 11, HOURS)
    assert by_hour(silent)[11].status == "future"
    for order in ([silent, busy], [busy, silent]):
        merged = by_hour(aggregate_bands(order))
        assert merged[11].status == "in_progress" and merged[11].actual is None
        assert merged[12].status == "future"


def test_aggregation_drops_quantile_bounds_and_passes_single_buckets_through():
    a = build_bands(FIVE, {8: 5}, 9, HOURS, [1.0] * 12, [9.0] * 12)
    assert aggregate_bands([a])[0].lower == 1.0
    merged = aggregate_bands([a, a])
    assert merged[0].lower is None and merged[0].upper is None and merged[0].position is None


def test_aggregation_rounds_money():
    a = build_bands([1.005] * 12, {8: 1.0}, 9, HOURS, digits=2)
    b = build_bands([1.005] * 12, {8: 1.0}, 9, HOURS, digits=2)
    merged = aggregate_bands([a, b], digits=2)
    assert merged[0].expected == 2.01 and merged[0].actual == 2.0


def test_aggregation_rounds_money_for_hours_that_have_not_closed_too():
    """Summing floats leaves noise such as 0.30000000000000004; money must show whole cents."""
    a = build_bands([0.1] * 12, {}, 8, HOURS, digits=2)
    b = build_bands([0.2] * 12, {}, 8, HOURS, digits=2)
    assert 0.1 + 0.2 != 0.3  # the premise: the raw float sum is noisy
    merged = aggregate_bands([a, b], digits=2)
    assert all(m.expected == 0.3 for m in merged)


def test_totals_actual_expected_and_projected_close():
    bands = build_bands([4.0] * 12, {8: 6, 9: 2}, 10, HOURS)
    totals = summarize(bands)
    assert totals.cumulative_actual == 8
    assert totals.expected_total == 48
    # 8 so far plus the expectation of the running hour and the eight hours after it.
    assert totals.projected_close == 8 + 4 * 10


def test_totals_with_money_rounding():
    bands = build_bands([0.333] * 12, {8: 0.1}, 9, HOURS, digits=2)
    assert summarize(bands, digits=2).projected_close == round(0.1 + 0.333 * 11, 2)


def test_empty_band_list_has_zero_totals():
    totals = summarize([])
    assert (totals.cumulative_actual, totals.expected_total, totals.projected_close) == (0, 0, 0)


def test_band_is_immutable():
    band = Band(8, 1.0, None, None, "future", None)
    with pytest.raises(AttributeError):
        band.expected = 2.0
