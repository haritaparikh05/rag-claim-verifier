import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FAKE_RESULT = {
    "verdict": "supported",
    "confidence": 0.95,
    "cited_passage": "IPX8 waterproof.",
    "explanation": "matches",
    "retrieved_chunks": [{"chunk_text": "a"}, {"chunk_text": "b"}],
}


def test_health_returns_ok_when_database_is_reachable():
    with patch("app.main.check_database", return_value=True):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_returns_503_when_database_is_unreachable():
    with patch("app.main.check_database", return_value=False):
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unreachable"}


def test_verify_returns_verdict_from_verify_claim():
    with patch("app.main.verify_claim", return_value=FAKE_RESULT):
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


def test_response_carries_a_request_id_header():
    with patch("app.main.check_database", return_value=True):
        response = client.get("/health")
    assert response.headers["x-request-id"]


def test_incoming_request_id_is_echoed_back():
    with patch("app.main.check_database", return_value=True):
        response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_verify_logs_a_structured_record(caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        with patch("app.main.verify_claim", return_value=FAKE_RESULT):
            client.post(
                "/verify",
                json={"product_id": "jbl_flip_6", "claim": "IP67 waterproof"},
                headers={"x-request-id": "req-1"},
            )

    records = [r for r in caplog.records if r.getMessage() == "verify"]
    assert len(records) == 1
    fields = records[0].fields
    assert fields["request_id"] == "req-1"
    assert fields["product_id"] == "jbl_flip_6"
    assert fields["verdict"] == "supported"
    assert fields["retrieved_chunks"] == 2
    assert fields["latency_ms"] >= 0


def test_successful_health_checks_are_not_logged(caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        with patch("app.main.check_database", return_value=True):
            client.get("/health")
    assert not [r for r in caplog.records if r.getMessage() == "request"]
