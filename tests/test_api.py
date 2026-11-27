import pytest
from fastapi.testclient import TestClient

from airquality import modeling
from app import main

GOOD = {"pm25_today": 150.0, "wind_kmph": 12.0, "humidity_pct": 55, "temp_c": 30.0,
        "traffic_index": 60, "stubble_fires": 120, "day_type": "weekday",
        "station_zone": "Central"}


@pytest.fixture()
def client(small_train, params, monkeypatch):
    cfg = params["candidates"][0]
    monkeypatch.setattr(main, "_model", modeling.fit(cfg["kind"], cfg["params"], small_train, params))
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_returns_a_number(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_client_rule_temperature_above_75_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "temp_c": 75.1}).status_code == 422
    assert client.post("/predict", json={**GOOD, "temp_c": 75}).status_code == 200


def test_client_rule_negative_wind_is_rejected(client):
    assert client.post("/predict", json={**GOOD, "wind_kmph": -1}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in GOOD.items() if k != "humidity_pct"}
    assert client.post("/predict", json=bad).status_code == 422


def test_log_line_is_json(client):
    """Check the formatter output directly instead of depending on how pytest
    captures stdout (which varies by platform/plugins)."""
    import json
    import logging

    captured = []

    class _Grab(logging.Handler):
        def emit(self, record):
            captured.append(main.logger.handlers[0].formatter.format(record))

    grab = _Grab()
    main.logger.addHandler(grab)
    try:
        client.post("/predict", json=GOOD)
    finally:
        main.logger.removeHandler(grab)
    assert json.loads(captured[-1])["event"] == "predict"


def test_frontend_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Tomorrow's Air Quality" in r.text
