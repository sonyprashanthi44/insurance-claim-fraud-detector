"""Minimal API and band-logic tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from main import app, band_for_proba  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_band_for_proba_ordering():
    thr = {"high": 0.8, "medium": 0.2}
    assert band_for_proba(0.9, thr) == "High"
    assert band_for_proba(0.5, thr) == "Medium"
    assert band_for_proba(0.1, thr) == "Low"
    assert band_for_proba(0.8, thr) == "High"
    assert band_for_proba(0.2, thr) == "Medium"


def test_predict_sample(client):
    samples = client.get("/samples").json()["samples"]
    assert len(samples) >= 1
    claim = samples[0]["claim"]
    r = client.post("/predict", json=claim)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["band"] in {"Low", "Medium", "High"}
    assert len(body["top_contributions"]) <= 6
    assert len(body["top_contributions"]) >= 1
    for c in body["top_contributions"]:
        assert "feature" in c and "shap" in c


def test_explain_template_fallback(client):
    payload = {
        "probability": 0.42,
        "band": "Medium",
        "top_contributions": [
            {"feature": "Fault", "value": "Policy Holder", "shap": 0.2},
            {"feature": "BasePolicy", "value": "All Perils", "shap": 0.1},
        ],
    }
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"]
    assert body["source"] in {"template", "groq"}


def test_metrics_shape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    m = data["metrics"]
    assert "test_pr_auc" in m
    assert "baseline_pr_auc" in m
    assert m["test_pr_auc"] > m["baseline_pr_auc"]
    assert "medium_band_metrics" in m
