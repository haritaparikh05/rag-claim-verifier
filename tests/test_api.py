import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_verify_returns_verdict_from_verify_claim():
    fake_result = {
        "verdict": "supported",
        "confidence": 0.95,
        "cited_passage": "IPX8 waterproof.",
        "explanation": "matches",
        "retrieved_chunks": [],
    }
    with patch("app.main.verify_claim", return_value=fake_result):
        response = client.post(
            "/verify",
            json={"product_id": "black_diamond_spot_400", "claim": "waterproof to 1 meter"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "supported"
    assert body["confidence"] == 0.95
    assert body["cited_passage"] == "IPX8 waterproof."


def test_verify_rejects_missing_fields():
    response = client.post("/verify", json={"product_id": "black_diamond_spot_400"})
    assert response.status_code == 422


def test_verify_rejects_empty_body():
    response = client.post("/verify", json={})
    assert response.status_code == 422
