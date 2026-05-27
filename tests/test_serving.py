"""Serving API smoke tests (requires fastapi)."""

import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serving.app import create_app  # noqa: E402


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_without_model_returns_400():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/predict", json={"records": [{"f0": 1.0}]})
    assert resp.status_code == 400
