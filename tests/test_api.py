"""
Tests pour l'API FastAPI
------------------------
"""
from fastapi.testclient import TestClient

from src.api.main import app


def test_root():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
