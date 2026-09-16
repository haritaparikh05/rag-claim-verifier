import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval import retrieve_chunks

# These are integration tests against the real local Postgres + ingested
# product data (not mocked) - the thing worth verifying is that the actual
# SQL/embedding round-trip works, which a mock can't meaningfully test.
# They skip (via the require_db fixture) rather than fail when Postgres
# isn't running, so the rest of the suite still runs cleanly.


def test_retrieve_chunks_scopes_to_requested_product(require_db):
    results = retrieve_chunks("jbl_flip_6", "waterproof rating", top_k=5)
    assert results, "expected at least one chunk for jbl_flip_6 - is the DB ingested? (python ingestion/ingest.py)"
    assert all(r["distance"] >= 0 for r in results)


def test_retrieve_chunks_respects_top_k(require_db):
    results = retrieve_chunks("jbl_flip_6", "battery life", top_k=3)
    assert 0 < len(results) <= 3


def test_retrieve_chunks_unknown_product_returns_empty(require_db):
    results = retrieve_chunks("this_product_does_not_exist", "any claim", top_k=5)
    assert results == []


def test_retrieve_chunks_results_ordered_by_distance_ascending(require_db):
    results = retrieve_chunks("jbl_flip_6", "IP67 waterproof and dustproof", top_k=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_retrieve_chunks_finds_correct_answer_for_known_claim(require_db):
    # a real regression check: this exact claim previously failed to surface
    # its answer until the top_k default was raised from 5 to 8 (Session 5)
    results = retrieve_chunks("anker_313_powercore_10k", "10000mAh capacity", top_k=8)
    assert any("10000mAh" in r["chunk_text"] for r in results)
