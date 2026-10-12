import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from app.main import app
from app.products import PRODUCT_IDS

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent
RESULT_FILES = {"dev": "eval_results.json", "holdout": "eval_results_holdout.json"}


def load_demo():
    with open(ROOT / "demo" / "claims.json", encoding="utf-8") as f:
        return json.load(f)["claims"]


def eval_record(source_set, product_id, claim):
    with open(ROOT / "data" / "eval" / RESULT_FILES[source_set], encoding="utf-8") as f:
        results = json.load(f)["results"]
    matches = [r for r in results if r["product_id"] == product_id and r["claim"] == claim]
    assert len(matches) == 1, f"{claim!r} is not in the {source_set} result file"
    return matches[0]


def test_demo_claims_have_the_expected_shape():
    claims = load_demo()
    assert len(claims) == 6
    assert [c["order"] for c in claims] == [1, 2, 3, 4, 5, 6]
    for c in claims:
        assert c["product_id"] in PRODUCT_IDS
        assert c["claim"].strip()
        assert c["expected_verdict"] in {"supported", "contradicted", "unverifiable"}
        assert c["source_set"] in RESULT_FILES
        assert c["demonstrates"].strip()


def test_demo_claims_cover_every_verdict():
    assert {c["expected_verdict"] for c in load_demo()} == {"supported", "contradicted", "unverifiable"}


def test_every_demo_claim_comes_from_a_verified_set_with_the_same_label():
    for c in load_demo():
        record = eval_record(c["source_set"], c["product_id"], c["claim"])
        assert record["expected_verdict"] == c["expected_verdict"], c["claim"]


def test_observed_results_are_copied_from_the_committed_result_files():
    for c in load_demo():
        record = eval_record(c["source_set"], c["product_id"], c["claim"])
        assert c["observed_in_eval"]["verdict"] == record["actual_verdict"], c["claim"]
        assert c["observed_in_eval"]["cited_passage"] == record["cited_passage"], c["claim"]


def test_exactly_one_demo_claim_is_a_flagged_known_miss():
    claims = load_demo()
    misses = [c for c in claims if c["known_miss"]]
    assert len(misses) == 1
    assert misses[0]["expected_verdict"] != misses[0]["observed_in_eval"]["verdict"]
    for c in claims:
        if not c["known_miss"]:
            assert c["expected_verdict"] == c["observed_in_eval"]["verdict"], c["claim"]


def test_demo_claims_endpoint_exposes_only_the_product_and_the_claim():
    response = client.get("/api/demo-claims")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 6
    for item in items:
        assert set(item) == {"product_id", "claim"}


def test_demo_claims_endpoint_returns_an_empty_list_when_the_file_is_missing(tmp_path):
    with patch("app.main.DEMO_CLAIMS_PATH", tmp_path / "missing.json"):
        response = client.get("/api/demo-claims")
    assert response.status_code == 200
    assert response.json() == []
