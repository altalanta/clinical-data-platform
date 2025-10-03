from fastapi.testclient import TestClient

from clinical_platform.api.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_endpoint():
    client = TestClient(app)
    r = client.post("/score", json={"AGE": 50, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0})
    assert r.status_code == 200
    assert 0 <= r.json()["risk"] <= 1

