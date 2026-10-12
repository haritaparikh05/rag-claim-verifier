import logging
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psycopg2
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.errors import (
    BUSY_MESSAGE,
    DATABASE_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    NO_KEY_MESSAGE,
    QUOTA_MESSAGE,
    REJECTED_MESSAGE,
)
from app.main import app
from app.products import PRODUCT_CATALOG, PRODUCT_IDS

client = TestClient(app)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FAKE_RESULT = {
    "verdict": "supported",
    "confidence": 0.95,
    "cited_passage": "IPX8 waterproof.",
    "explanation": "matches",
    "retrieved_chunks": [{"chunk_text": "a"}, {"chunk_text": "b"}],
}


def post_verify(claim="waterproof to 1 meter", product_id="black_diamond_spot_400"):
    return client.post("/verify", json={"product_id": product_id, "claim": claim})


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
        response = post_verify()

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "supported"
    assert body["confidence"] == 0.95
    assert body["cited_passage"] == "IPX8 waterproof."
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0


def test_verify_rejects_missing_fields():
    response = client.post("/verify", json={"product_id": "black_diamond_spot_400"})
    assert response.status_code == 422


def test_verify_rejects_empty_body():
    response = client.post("/verify", json={})
    assert response.status_code == 422


def test_empty_claim_gets_a_friendly_message():
    response = post_verify(claim="")
    assert response.status_code == 422
    assert response.json() == {"detail": "Please enter a claim to check."}


def test_whitespace_only_claim_is_treated_as_empty():
    with patch("app.main.verify_claim", return_value=FAKE_RESULT) as mock_verify:
        response = post_verify(claim="   ")
    assert response.status_code == 422
    assert response.json()["detail"] == "Please enter a claim to check."
    mock_verify.assert_not_called()


def test_overlong_claim_gets_a_friendly_message():
    response = post_verify(claim="x" * 501)
    assert response.status_code == 422
    assert "too long" in response.json()["detail"]


def test_missing_product_gets_a_friendly_message():
    response = client.post("/verify", json={"claim": "waterproof"})
    assert response.status_code == 422
    assert response.json()["detail"] == "Please choose a product."


def test_claim_is_trimmed_before_it_reaches_the_judge():
    with patch("app.main.verify_claim", return_value=FAKE_RESULT) as mock_verify:
        post_verify(claim="  waterproof to 1 meter  ", product_id="jbl_flip_6")
    mock_verify.assert_called_once_with("jbl_flip_6", "waterproof to 1 meter")


def test_unknown_product_is_a_404_and_never_reaches_the_judge():
    with patch("app.main.verify_claim", return_value=FAKE_RESULT) as mock_verify:
        response = post_verify(product_id="not_a_real_product")
    assert response.status_code == 404
    assert "Unknown product" in response.json()["detail"]
    mock_verify.assert_not_called()


def test_gemini_rate_limit_becomes_a_429_with_a_clear_message():
    error = genai_errors.APIError(429, {"error": {"message": "quota"}})
    with patch("app.main.verify_claim", side_effect=error):
        response = post_verify()
    assert response.status_code == 429
    assert response.json() == {"detail": QUOTA_MESSAGE}


def test_gemini_server_error_becomes_a_503_with_a_clear_message():
    error = genai_errors.APIError(503, {"error": {"message": "overloaded"}})
    with patch("app.main.verify_claim", side_effect=error):
        response = post_verify()
    assert response.status_code == 503
    assert response.json() == {"detail": BUSY_MESSAGE}


def test_gemini_auth_error_points_at_the_api_key():
    error = genai_errors.APIError(403, {"error": {"message": "denied"}})
    with patch("app.main.verify_claim", side_effect=error):
        response = post_verify()
    assert response.status_code == 502
    assert response.json() == {"detail": REJECTED_MESSAGE}


def test_missing_api_key_is_explained():
    with patch("app.main.verify_claim", side_effect=RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")):
        response = post_verify()
    assert response.status_code == 503
    assert response.json() == {"detail": NO_KEY_MESSAGE}


def test_database_failure_is_explained():
    with patch("app.main.verify_claim", side_effect=psycopg2.OperationalError("connection refused")):
        response = post_verify()
    assert response.status_code == 503
    assert response.json() == {"detail": DATABASE_MESSAGE}


def test_product_with_no_loaded_documents_is_an_error_not_a_fake_answer():
    empty = {
        "verdict": "unverifiable",
        "confidence": 0.0,
        "cited_passage": "",
        "explanation": "No documentation found for product 'x'.",
        "retrieved_chunks": [],
    }
    with patch("app.main.verify_claim", return_value=empty):
        response = post_verify()
    assert response.status_code == 503
    assert response.json() == {"detail": NO_DOCUMENTS_MESSAGE}


def test_unexpected_errors_are_not_disguised_as_friendly_ones():
    crashing_client = TestClient(app, raise_server_exceptions=False)
    with patch("app.main.verify_claim", side_effect=ValueError("a real bug")):
        response = crashing_client.post(
            "/verify", json={"product_id": "black_diamond_spot_400", "claim": "waterproof"}
        )
    assert response.status_code == 500


def test_products_lists_every_product_with_its_chunk_count():
    with patch("app.main.count_chunks_by_product", return_value={"jbl_flip_6": 7}):
        response = client.get("/products")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == len(PRODUCT_CATALOG) == 9
    by_id = {item["product_id"]: item for item in items}
    assert by_id["jbl_flip_6"]["chunks"] == 7
    assert by_id["jbl_flip_6"]["name"] == "JBL Flip 6 Speaker"
    assert by_id["alps_helix_2"]["chunks"] == 0
    for item in items:
        assert item["category"]
        assert len(item["examples"]) == 3


def test_products_reports_an_unreachable_database():
    with patch("app.main.count_chunks_by_product", side_effect=psycopg2.OperationalError("down")):
        response = client.get("/products")
    assert response.status_code == 503
    assert response.json() == {"detail": DATABASE_MESSAGE}


def test_catalog_matches_the_folders_in_data_raw():
    folders = {p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()}
    assert PRODUCT_IDS == folders


def test_home_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Claim Verifier" in response.text


def test_static_assets_are_served():
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


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
