import threading
import time

import pytest

from call_pace.artifacts import KINDS, ArtifactStore, TtlCache
from call_pace.baseline import LookupForecaster
from call_pace.errors import ArtifactError
from call_pace.knn import KnnForecaster
from call_pace.quantile import QuantileForecaster
from call_pace.value import TicketLookup


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_cache_loads_once_within_the_ttl_and_again_after_it():
    clock, calls = Clock(), []
    cache = TtlCache(10, clock)

    def load():
        calls.append(1)
        return len(calls)

    assert cache.get_or_load("k", load) == 1
    clock.now = 9
    assert cache.get_or_load("k", load) == 1
    clock.now = 11
    assert cache.get_or_load("k", load) == 2
    assert len(calls) == 2


def test_cache_keys_are_independent_and_clear_empties_everything():
    cache = TtlCache(100, Clock())
    assert cache.get_or_load("a", lambda: 1) == 1
    assert cache.get_or_load("b", lambda: 2) == 2
    assert cache.get_or_load("a", lambda: 99) == 1
    cache.clear()
    assert cache.get_or_load("a", lambda: 99) == 99


def test_a_failed_load_is_not_cached():
    cache = TtlCache(100, Clock())

    def broken():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cache.get_or_load("k", broken)
    assert cache.get_or_load("k", lambda: "ok") == "ok"


def test_concurrent_requests_trigger_a_single_load():
    cache, calls = TtlCache(100, time.monotonic), []

    def slow():
        calls.append(1)
        time.sleep(0.05)
        return "value"

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get_or_load("k", slow)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["value"] * 8 and len(calls) == 1


@pytest.fixture(scope="module")
def models(frame, spec, config, hourly):
    from call_pace.synthetic import make_synthetic_amounts

    return {
        "lookup_frame": frame,
        "lookup": LookupForecaster(30).fit(frame),
        "knn": KnnForecaster(spec, 5).fit(frame),
        "quantile": QuantileForecaster(spec, params=config.lgbm_params).fit(frame),
        "ticket": TicketLookup(window_days=30).fit(make_synthetic_amounts(hourly)),
    }


def test_every_kind_round_trips_through_the_store(tmp_path, models):
    store = ArtifactStore(tmp_path / "artifacts")
    assert store.available() == []
    for kind in KINDS:
        assert not store.exists(kind)
        store.save(kind, models[kind])
        assert store.exists(kind)
        assert type(store.load(kind)) is type(models[kind])
    assert store.available() == list(KINDS)


def test_unknown_kinds_and_missing_files_have_clear_errors(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="unknown artifact kind"):
        store.path("arima")
    with pytest.raises(ArtifactError, match="train first"):
        store.load("knn")


def test_loaded_models_are_cached_until_the_ttl_expires(tmp_path, models):
    clock = Clock()
    store = ArtifactStore(tmp_path, ttl_seconds=10, clock=clock)
    store.save("lookup", models["lookup"])
    first = store.load("lookup")
    assert store.load("lookup") is first
    clock.now = 11
    assert store.load("lookup") is not first


def test_a_model_retrained_by_another_process_is_picked_up_before_the_ttl(tmp_path, models):
    clock = Clock()
    serving = ArtifactStore(tmp_path, ttl_seconds=10_000, clock=clock)
    serving.save("lookup", models["lookup"])
    before = serving.load("lookup")
    ArtifactStore(tmp_path).save("lookup", LookupForecaster(5).fit(models["lookup_frame"]))
    after = serving.load("lookup")
    assert after is not before and after.window_days == 5


def test_saving_again_invalidates_the_cache(tmp_path, models):
    store = ArtifactStore(tmp_path, ttl_seconds=1000, clock=Clock())
    store.save("lookup", models["lookup"])
    first = store.load("lookup")
    store.save("lookup", models["lookup"])
    assert store.load("lookup") is not first


def test_manifest_records_the_feature_layout_and_summary(tmp_path, spec, models):
    store = ArtifactStore(tmp_path)
    store.save("lookup", models["lookup"])
    store.write_manifest(spec, {"rows": 123})
    manifest = store.manifest()
    assert manifest["rows"] == 123 and manifest["kinds"] == ["lookup"]
    assert store.spec() == spec
    assert store.known_buckets() == list(spec.buckets)


def test_a_missing_or_broken_manifest_is_an_artifact_error(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError, match="train first"):
        store.manifest()
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path).manifest()
    (tmp_path / "manifest.json").write_text('{"format_version": 99}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="supported"):
        ArtifactStore(tmp_path).manifest()
