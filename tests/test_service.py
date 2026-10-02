import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from call_pace.artifacts import ArtifactStore
from call_pace.config import Config
from call_pace.contract import validate_hourly
from call_pace.errors import (
    ArtifactError,
    ConfigError,
    DataSourceError,
    InvalidRequestError,
)
from call_pace.features import serving_frame
from call_pace.loaders import InMemoryLoader
from call_pace.service import PaceService
from call_pace.synthetic import make_synthetic_hourly
from call_pace.training import train_models


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SpyLoader(InMemoryLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def load_hourly(self, start, end):
        self.calls += 1
        return super().load_hourly(start, end)


def clock_at(day, hour):
    return lambda: datetime(day.year, day.month, day.day, hour, 20)


@pytest.fixture
def make(routed, trained_store, hourly, amounts, day):
    def build(hour=11, config=None, loader=None, with_amounts=True, ttl=30.0, tick=None, **extra):
        return PaceService(
            config or routed,
            loader or InMemoryLoader(hourly, amounts),
            trained_store,
            amount_loader=(loader or InMemoryLoader(hourly, amounts)) if with_amounts else None,
            clock=clock_at(day, hour),
            data_ttl_seconds=ttl,
            monotonic=tick or Clock(),
            **extra,
        )

    return build


def rows_of(hourly, day, bucket):
    today = hourly[(hourly["date"] == pd.Timestamp(day)) & (hourly["bucket"] == bucket)]
    return {int(h): float(c) for h, c in zip(today["hour"], today["count"], strict=True)}


def bands_of(payload):
    return {b["hour"]: b for b in payload["data"]["bands"]}


# --- outside operating hours --------------------------------------------------------------


@pytest.mark.parametrize("hour", [0, 7, 20, 23])
def test_outside_the_operating_hours_nothing_is_loaded_or_computed(make, hour):
    class Forbidden:
        def load_hourly(self, start, end):
            raise AssertionError("data must not be read outside operating hours")

    payload = make(hour=hour, loader=Forbidden(), with_amounts=False).pace("auto")
    assert payload["meta"]["in_operation"] is False
    assert payload["data"]["bands"] == [] and payload["errors"] == []


def test_a_weekend_is_closed_even_during_working_hours(routed, trained_store, hourly, day):
    saturday = day + timedelta(days=(5 - day.weekday()) % 7)
    service = PaceService(
        routed, InMemoryLoader(hourly), trained_store, clock=clock_at(saturday, 10)
    )
    assert service.pace("all")["meta"]["in_operation"] is False


def test_an_unknown_bucket_is_a_bad_request_even_when_closed(make):
    with pytest.raises(InvalidRequestError, match="unknown bucket"):
        make(hour=3).pace("nope")
    with pytest.raises(InvalidRequestError):
        make(hour=11).pace("nope")


# --- an open day --------------------------------------------------------------------------


def test_statuses_and_totals_at_eleven_oclock(make, hourly, day):
    payload = make(hour=11).pace("auto")
    bands = bands_of(payload)
    seen = rows_of(hourly, day, "auto")
    assert list(bands) == list(range(8, 20))
    assert all(bands[h]["status"] in ("above", "below", "ok") for h in (8, 9, 10))
    assert bands[11]["status"] == "in_progress" and bands[11]["actual"] is None
    assert all(bands[h]["status"] == "future" for h in range(12, 20))
    assert payload["data"]["cumulative_actual"] == sum(seen[h] for h in (8, 9, 10))
    assert bands[10]["cumulative"] == payload["data"]["cumulative_actual"]
    open_expected = sum(bands[h]["expected"] for h in range(11, 20))
    assert (
        payload["data"]["projected_close"] == payload["data"]["cumulative_actual"] + open_expected
    )
    assert payload["data"]["expected_total"] == sum(b["expected"] for b in bands.values())
    assert payload["meta"]["in_operation"] is True and payload["errors"] == []


def test_each_bucket_is_served_by_the_model_it_is_routed_to(make, trained_store, hourly, day):
    store = trained_store
    spec = store.spec()
    auto = bands_of(make(hour=11).pace("auto"))
    lookup = store.load("lookup").expected_for_day("auto", list(spec.hours))
    assert [auto[h]["expected"] for h in spec.hours] == lookup.tolist()

    consumer = bands_of(make(hour=11).pace("consumer"))
    frame = serving_frame(spec, "consumer", day, rows_of(hourly, day, "consumer"), 11)
    knn = store.load("knn").predict(frame)
    assert [consumer[h]["expected"] for h in spec.hours] == knn.tolist()


def test_a_knn_bucket_reacts_to_the_day_so_far_but_a_lookup_bucket_does_not(make, hourly, day):
    today = hourly["date"] == pd.Timestamp(day)
    doubled = hourly.copy()
    doubled.loc[today & (doubled["hour"] < 11), "count"] *= 2.0
    base_service = make(loader=InMemoryLoader(hourly), with_amounts=False)
    busy_service = make(loader=InMemoryLoader(doubled), with_amounts=False)

    def expected(service, bucket):
        return [b["expected"] for b in service.pace(bucket)["data"]["bands"]]

    assert expected(base_service, "auto") == expected(busy_service, "auto")  # lookup: today ignored
    assert expected(base_service, "consumer") != expected(busy_service, "consumer")  # knn: reacts


def test_all_adds_up_its_buckets_and_reports_a_hybrid_model(make):
    service = make(hour=14, with_amounts=False)
    total, auto, consumer = service.pace("all"), service.pace("auto"), service.pace("consumer")
    for hour, band in bands_of(total).items():
        assert (
            band["expected"]
            == bands_of(auto)[hour]["expected"] + bands_of(consumer)[hour]["expected"]
        )
    assert total["data"]["cumulative_actual"] == (
        auto["data"]["cumulative_actual"] + consumer["data"]["cumulative_actual"]
    )
    assert total["meta"]["model"] == "hybrid(auto=lookup, consumer=knn)"
    assert auto["meta"]["model"] == "lookup" and total["meta"]["buckets"] == ["auto", "consumer"]


def test_quantile_routing_adds_a_band_and_a_verdict_for_closed_hours(make, routed):
    config = routed.replace(routing={"auto": "quantile", "consumer": "quantile"})
    service = make(hour=15, config=config, with_amounts=False)
    single = bands_of(service.pace("auto"))
    assert all(single[h]["lower"] <= single[h]["upper"] for h in single)
    assert all(
        single[h]["position"] in ("below_band", "within_band", "above_band") for h in range(8, 15)
    )
    assert single[15]["position"] is None
    merged = bands_of(service.pace("all"))
    assert all(b["lower"] is None and b["upper"] is None for b in merged.values())


def test_payload_is_plain_json_and_counts_are_whole_numbers(make):
    payload = make(hour=12).pace("all")
    json.dumps(payload)
    first = payload["data"]["bands"][0]
    assert isinstance(first["expected"], int) and isinstance(first["actual"], int)


def test_with_no_data_at_all_the_current_hour_is_not_in_progress(make, hourly):
    empty = hourly.iloc[:0]
    payload = make(hour=12, loader=InMemoryLoader(empty), with_amounts=False).pace("consumer")
    bands = bands_of(payload)
    assert all(bands[h]["status"] == "below" for h in range(8, 12))
    assert all(bands[h]["status"] == "future" for h in range(12, 20))
    assert payload["data"]["cumulative_actual"] == 0


# --- failure policy -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [RuntimeError("anything"), DataSourceError("db down"), TimeoutError()]
)
def test_a_data_failure_degrades_instead_of_raising(make, error, caplog):
    class Failing:
        def load_hourly(self, start, end):
            raise error

    with caplog.at_level(logging.ERROR):
        payload = make(hour=14, loader=Failing(), with_amounts=False).pace("all")
    assert payload["meta"]["in_operation"] is True
    assert payload["data"]["bands"] == [] and payload["errors"]
    assert "could not read today's data" in caplog.text


def test_a_bad_request_while_reading_data_is_never_swallowed(make):
    class Rejecting:
        def load_hourly(self, start, end):
            raise InvalidRequestError("the caller asked for something impossible")

    with pytest.raises(InvalidRequestError):
        make(hour=14, loader=Rejecting(), with_amounts=False).pace("all")


def test_malformed_rows_from_the_loader_also_degrade(make, hourly):
    broken = hourly.assign(count=-1.0)
    payload = make(hour=14, loader=InMemoryLoader(broken), with_amounts=False).pace("auto")
    assert payload["errors"] and payload["data"]["bands"] == []


def test_a_missing_model_is_an_error_not_a_degraded_payload(routed, tmp_path, hourly, config, day):
    store = ArtifactStore(tmp_path)
    train_models(hourly, config, store, kinds=("lookup",))  # nothing for the knn-routed bucket
    spy = SpyLoader(hourly)
    service = PaceService(routed, spy, store, clock=clock_at(day, 11))
    with pytest.raises(ArtifactError, match="knn"):
        service.pace("consumer")
    assert spy.calls == 0  # models are loaded before any data is read


# --- data cache ---------------------------------------------------------------------------


def test_todays_data_is_cached_for_its_ttl(make, hourly):
    spy, clock = SpyLoader(hourly), Clock()
    service = make(loader=spy, with_amounts=False, ttl=30.0, tick=clock)
    service.pace("auto")
    service.pace("consumer")
    assert spy.calls == 1
    clock.now = 31
    service.pace("auto")
    assert spy.calls == 2


def test_a_zero_ttl_disables_the_cache(make, hourly):
    spy = SpyLoader(hourly)
    service = make(loader=spy, with_amounts=False, ttl=0)
    service.pace("auto")
    service.pace("auto")
    assert spy.calls == 2


# --- value --------------------------------------------------------------------------------


def test_value_bands_multiply_expected_count_by_the_ticket(make, trained_store, amounts, day):
    payload = make(hour=13).pace("auto")
    ticket = trained_store.load("ticket")
    bands = bands_of(payload)
    for hour in (9, 15):
        assert bands[hour]["expected_value"] == round(
            bands[hour]["expected"] * ticket.ticket_for("auto", hour), 2
        )
    assert payload["meta"]["value_model"] == "ticket_x_expected_count"
    today = amounts[(amounts["date"] == pd.Timestamp(day)) & (amounts["bucket"] == "auto")]
    closed = today[today["hour"] < 13]["amount"].sum()
    assert payload["data"]["value_cumulative_actual"] == pytest.approx(round(closed, 2))
    open_value = sum(bands[h]["expected_value"] for h in range(13, 20))
    assert payload["data"]["value_projected_close"] == pytest.approx(round(closed + open_value, 2))
    assert bands[13]["status_value"] == "in_progress" and bands[16]["status_value"] == "future"


def test_value_for_all_buckets_is_the_sum_of_the_parts(make):
    service = make(hour=16)
    total, auto, consumer = service.pace("all"), service.pace("auto"), service.pace("consumer")
    assert total["data"]["value_expected_total"] == pytest.approx(
        auto["data"]["value_expected_total"] + consumer["data"]["value_expected_total"], abs=0.02
    )


def test_without_an_amount_loader_there_is_no_value(make):
    payload = make(hour=13, with_amounts=False).pace("auto")
    assert "value_model" not in payload["meta"]
    assert "expected_value" not in payload["data"]["bands"][0]


# --- covariates ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def covariate_setup(tmp_path_factory):
    cfg = Config(
        covariates=("days_since_event",),
        n_folds=3,
        fold_size_days=5,
        min_train_days=20,
        k_grid=(3, 5),
        routing={"auto": "knn", "consumer": "knn"},
    )
    data = validate_hourly(
        make_synthetic_hourly(n_days=60, seed=8, config=cfg, with_covariate=True), cfg
    )
    store = ArtifactStore(tmp_path_factory.mktemp("cov"))
    train_models(data, cfg, store, kinds=("knn",))
    return cfg, store, data


def test_models_with_covariates_need_a_provider(covariate_setup):
    cfg, store, data = covariate_setup
    day = data["date"].max().date()
    service = PaceService(cfg, InMemoryLoader(data), store, clock=clock_at(day, 11))
    with pytest.raises(ConfigError, match="days_since_event"):
        service.pace("auto")


def test_the_provider_is_asked_for_each_model_bucket_and_day(covariate_setup):
    cfg, store, data = covariate_setup
    day = data["date"].max().date()
    asked = []

    def provider(bucket, when):
        asked.append((bucket, when))
        return {"days_since_event": 3.0}

    service = PaceService(
        cfg, InMemoryLoader(data), store, covariate_provider=provider, clock=clock_at(day, 11)
    )
    payload = service.pace("all")
    assert sorted(asked) == [("auto", day), ("consumer", day)]
    assert len(payload["data"]["bands"]) == 12
