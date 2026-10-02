import json

import numpy as np
import pytest

from call_pace.artifacts import ArtifactStore
from call_pace.config import Config
from call_pace.errors import ConfigError
from call_pace.features import build_frame
from call_pace.synthetic import make_synthetic_amounts
from call_pace.training import select_k, train_models


def test_training_writes_every_requested_model_and_a_manifest(tmp_path, hourly, config):
    store = ArtifactStore(tmp_path / "artifacts")
    report = train_models(hourly, config, store)
    assert store.available() == ["lookup", "knn", "quantile"]
    assert report.buckets == ["auto", "consumer"]
    assert report.rows == hourly["date"].nunique() * 2 * 12
    assert report.days == hourly["date"].nunique() and report.first_day < report.last_day
    manifest = json.loads((tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kinds"] == ["lookup", "knn", "quantile"] and manifest["rows"] == report.rows


def test_k_is_selected_by_cross_validation_from_the_configured_grid(tmp_path, hourly, config):
    report = train_models(hourly, config, ArtifactStore(tmp_path))
    assert report.k in config.k_grid and report.k_selected_by_cv
    assert ArtifactStore(tmp_path).load("knn").k == report.k


def test_an_explicit_k_overrides_selection(tmp_path, hourly, config):
    report = train_models(hourly, config, ArtifactStore(tmp_path), kinds=("knn",), k=11)
    assert (report.k, report.k_selected_by_cv) == (11, False)


def test_short_history_falls_back_to_the_default_k_and_says_so(tmp_path, hourly, config):
    short = hourly[hourly["date"] < hourly["date"].unique()[15]]
    report = train_models(short, config, ArtifactStore(tmp_path), kinds=("knn",))
    assert report.k == config.default_k and not report.k_selected_by_cv
    assert any("too short" in note for note in report.notes)


def test_subset_of_kinds_and_unknown_kind(tmp_path, hourly, config):
    store = ArtifactStore(tmp_path)
    train_models(hourly, config, store, kinds=("lookup",))
    assert store.available() == ["lookup"]
    with pytest.raises(ConfigError, match="unknown model kinds"):
        train_models(hourly, config, store, kinds=("arima",))


def test_amounts_add_a_ticket_lookup(tmp_path, hourly, config):
    store = ArtifactStore(tmp_path)
    report = train_models(
        hourly, config, store, amounts=make_synthetic_amounts(hourly), kinds=("lookup",)
    )
    assert "ticket" in report.kinds and store.exists("ticket")
    assert store.load("ticket").covers(["auto", "consumer"])


def test_the_deployed_model_is_fitted_on_all_the_data(tmp_path, hourly, frame, spec, config):
    store = ArtifactStore(tmp_path)
    train_models(hourly, config, store, kinds=("lookup",))
    lookup = store.load("lookup")
    assert lookup.trained["auto"]["to"] == hourly["date"].max().date().isoformat()


def test_select_k_returns_none_when_the_protocol_does_not_fit(hourly, spec, config):
    short = build_frame(hourly[hourly["date"] < hourly["date"].unique()[10]], spec)
    assert select_k(short, spec, config) is None
    full = build_frame(hourly, spec)
    assert select_k(full, spec, config) in config.k_grid


def test_training_is_deterministic(tmp_path, hourly, config):
    a = ArtifactStore(tmp_path / "a")
    b = ArtifactStore(tmp_path / "b")
    train_models(hourly, config, a)
    train_models(hourly, config, b)
    frame = build_frame(hourly, a.spec())
    for kind in ("lookup", "knn", "quantile"):
        assert np.array_equal(a.load(kind).predict(frame), b.load(kind).predict(frame)), kind


def test_default_config_trains_on_a_realistic_amount_of_history(tmp_path):
    from call_pace.contract import validate_hourly
    from call_pace.synthetic import make_synthetic_hourly

    cfg = Config()
    data = validate_hourly(make_synthetic_hourly(n_days=90, seed=1, config=cfg), cfg)
    report = train_models(data, cfg, ArtifactStore(tmp_path), kinds=("lookup", "knn"))
    assert report.k_selected_by_cv
