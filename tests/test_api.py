from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from call_pace.api import create_app
from call_pace.artifacts import ArtifactStore
from call_pace.errors import DataSourceError
from call_pace.loaders import InMemoryLoader
from call_pace.service import PaceService


def client_for(config, store, loader, day, hour=11, amount_loader=None):
    def clock():
        return datetime(day.year, day.month, day.day, hour, 5)

    service = PaceService(config, loader, store, amount_loader, clock=clock)
    return TestClient(create_app(service))


@pytest.fixture
def client(routed, trained_store, hourly, amounts, day):
    loader = InMemoryLoader(hourly, amounts)
    return client_for(routed, trained_store, loader, day, amount_loader=loader)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_one_bucket_returns_the_envelope_with_twelve_bands(client):
    response = client.get("/pace/consumer")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"meta", "data", "errors"}
    assert body["meta"]["bucket"] == "consumer" and body["meta"]["model"] == "knn"
    assert len(body["data"]["bands"]) == 12 and body["errors"] == []
    assert body["meta"]["value_model"] == "ticket_x_expected_count"


def test_the_default_route_is_every_bucket(client):
    body = client.get("/pace").json()
    assert body["meta"]["bucket"] == "all" and body["meta"]["buckets"] == ["auto", "consumer"]
    assert body["meta"]["model"].startswith("hybrid(")


def test_unknown_bucket_is_a_400_with_an_explanation(client):
    response = client.get("/pace/nope")
    assert response.status_code == 400 and "unknown bucket" in response.json()["detail"]


def test_closed_hours_are_a_normal_response(routed, trained_store, hourly, day):
    client = client_for(routed, trained_store, InMemoryLoader(hourly), day, hour=22)
    body = client.get("/pace/auto").json()
    assert body["meta"]["in_operation"] is False and body["data"]["bands"] == []


def test_a_data_failure_is_still_a_200_with_the_failure_reported(routed, trained_store, day):
    class Down:
        def load_hourly(self, start, end):
            raise DataSourceError("database unreachable")

    response = client_for(routed, trained_store, Down(), day).get("/pace/auto")
    assert response.status_code == 200
    body = response.json()
    assert body["errors"] and body["data"]["bands"] == [] and body["meta"]["in_operation"]


def test_untrained_models_are_a_500_that_does_not_leak_paths(routed, hourly, day, tmp_path):
    empty = ArtifactStore(tmp_path / "secret_directory_name")
    response = client_for(routed, empty, InMemoryLoader(hourly), day).get("/pace/auto")
    assert response.status_code == 500
    assert "secret_directory_name" not in response.text
    assert response.json()["detail"] == "models or configuration unavailable"
