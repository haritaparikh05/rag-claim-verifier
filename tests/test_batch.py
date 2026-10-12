import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.batch import MAX_BATCH_SIZE, MAX_CONCURRENCY, Throttle, run_batch
from app.errors import BUSY_MESSAGE, NO_DOCUMENTS_MESSAGE, QUOTA_MESSAGE
from app.main import app

client = TestClient(app)


def ok_result(*_args):
    return {
        "verdict": "supported",
        "confidence": 0.9,
        "cited_passage": "a passage",
        "explanation": "because",
        "retrieved_chunks": [{"chunk_text": "c"}],
    }


class Tracker:
    """A fake judge that records how many calls were running at the same moment."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def __call__(self, product_id, claim):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return ok_result(claim)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def by_index(results):
    return {r["index"]: r for r in results}


def parse_stream(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def post_batch(claims, product_id="jbl_flip_6"):
    return client.post("/verify/batch", json={"product_id": product_id, "claims": claims})


# ---------- the engine ----------


def test_throttle_spaces_call_starts_by_the_minimum_interval():
    clock = FakeClock()
    throttle = Throttle(4.0, clock=clock.time, sleep=clock.sleep)
    for _ in range(3):
        throttle.wait()
    assert clock.sleeps == [4.0, 4.0]


def test_throttle_does_not_wait_when_calls_are_already_far_apart():
    clock = FakeClock()
    throttle = Throttle(4.0, clock=clock.time, sleep=clock.sleep)
    throttle.wait()
    clock.now += 10
    throttle.wait()
    assert clock.sleeps == []


def test_never_more_than_three_claims_are_checked_at_once():
    tracker = Tracker()
    results = list(run_batch("p", [f"claim {i}" for i in range(9)], tracker, min_interval=0))
    assert len(results) == 9
    assert tracker.max_active <= MAX_CONCURRENCY == 3


def test_every_claim_comes_back_exactly_once_with_its_index():
    claims = [f"claim {i}" for i in range(7)]
    results = list(run_batch("p", claims, Tracker(), min_interval=0))
    assert sorted(r["index"] for r in results) == list(range(7))
    for r in results:
        assert r["claim"] == claims[r["index"]]
        assert r["error"] is None
        assert r["verdict"] == "supported"
        assert r["latency_ms"] >= 0


def test_a_rate_limit_stops_the_rest_and_keeps_what_finished():
    calls = []

    def judge(product_id, claim):
        calls.append(claim)
        if claim == "claim 1":
            raise genai_errors.APIError(429, {"error": {"message": "quota"}})
        return ok_result(claim)

    results = by_index(run_batch("p", [f"claim {i}" for i in range(5)], judge, max_workers=1, min_interval=0))

    assert results[0]["error"] is None
    assert results[1]["status"] == 429 and results[1]["error"] == QUOTA_MESSAGE
    for i in (2, 3, 4):
        assert results[i]["error"].startswith("Skipped:")
    assert calls == ["claim 0", "claim 1"]


def test_a_busy_error_on_one_claim_does_not_stop_the_others():
    def judge(product_id, claim):
        if claim == "claim 1":
            raise genai_errors.APIError(503, {"error": {"message": "busy"}})
        return ok_result(claim)

    results = by_index(run_batch("p", [f"claim {i}" for i in range(4)], judge, max_workers=1, min_interval=0))

    assert results[1]["status"] == 503 and results[1]["error"] == BUSY_MESSAGE
    assert [results[i]["error"] for i in (0, 2, 3)] == [None, None, None]


def test_missing_documents_stop_the_rest_with_a_clear_message():
    def judge(product_id, claim):
        return {**ok_result(), "retrieved_chunks": []}

    results = by_index(run_batch("p", ["a", "b", "c"], judge, max_workers=1, min_interval=0))

    assert results[0]["error"] == NO_DOCUMENTS_MESSAGE
    assert results[1]["error"].startswith("Skipped:")
    assert results[2]["error"].startswith("Skipped:")


def test_an_unexpected_error_is_reported_on_that_claim_only():
    def judge(product_id, claim):
        if claim == "boom":
            raise ValueError("a real bug")
        return ok_result(claim)

    results = by_index(run_batch("p", ["ok", "boom", "ok too"], judge, max_workers=1, min_interval=0))

    assert results[1]["status"] == 500
    assert results[0]["error"] is None and results[2]["error"] is None


# ---------- the endpoint ----------


def test_endpoint_streams_start_results_and_done():
    with patch("app.main.MIN_SECONDS_BETWEEN_STARTS", 0), patch("app.main.verify_claim", side_effect=ok_result):
        response = post_batch(["one", "two", "three"])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = parse_stream(response)
    assert lines[0] == {"type": "start", "total": 3, "max_concurrency": 3}
    results = [line for line in lines if line["type"] == "result"]
    assert sorted(r["index"] for r in results) == [0, 1, 2]
    assert lines[-1]["type"] == "done"
    assert lines[-1]["succeeded"] == 3 and lines[-1]["failed"] == 0


def test_endpoint_runs_at_most_three_at_once():
    tracker = Tracker()
    with patch("app.main.MIN_SECONDS_BETWEEN_STARTS", 0), patch("app.main.verify_claim", tracker):
        post_batch([f"claim {i}" for i in range(8)])
    assert tracker.calls == 8
    assert tracker.max_active <= 3


def test_endpoint_reports_partial_results_when_the_rate_limit_is_hit():
    def judge(product_id, claim):
        if claim == "claim 2":
            raise genai_errors.APIError(429, {"error": {"message": "quota"}})
        return ok_result(claim)

    with patch("app.main.MIN_SECONDS_BETWEEN_STARTS", 0), patch("app.main.verify_claim", side_effect=judge):
        response = post_batch([f"claim {i}" for i in range(6)])

    lines = parse_stream(response)
    results = [line for line in lines if line["type"] == "result"]
    assert len(results) == 6
    assert any(r["status"] == 429 for r in results if r["error"])
    assert lines[-1]["failed"] >= 1
    assert lines[-1]["succeeded"] + lines[-1]["failed"] == 6


def test_endpoint_trims_claims_before_checking_them():
    calls = []

    def judge(product_id, claim):
        calls.append((product_id, claim))
        return ok_result(claim)

    with patch("app.main.MIN_SECONDS_BETWEEN_STARTS", 0), patch("app.main.verify_claim", side_effect=judge):
        post_batch(["  padded claim  "], product_id=" jbl_flip_6 ")
    assert calls == [("jbl_flip_6", "padded claim")]


def test_endpoint_rejects_more_than_the_maximum_batch_size():
    response = post_batch([f"claim {i}" for i in range(MAX_BATCH_SIZE + 1)])
    assert response.status_code == 422
    assert response.json() == {"detail": "A batch can contain at most 25 claims."}


def test_endpoint_accepts_exactly_the_maximum_batch_size():
    with patch("app.main.MIN_SECONDS_BETWEEN_STARTS", 0), patch("app.main.verify_claim", side_effect=ok_result):
        response = post_batch([f"claim {i}" for i in range(MAX_BATCH_SIZE)])
    assert response.status_code == 200
    assert parse_stream(response)[-1]["succeeded"] == MAX_BATCH_SIZE


def test_endpoint_rejects_an_empty_list():
    response = post_batch([])
    assert response.status_code == 422
    assert response.json() == {"detail": "Please add at least one claim."}


def test_endpoint_rejects_blank_or_overlong_claims():
    for claims in (["fine", "   "], ["x" * 501]):
        response = post_batch(claims)
        assert response.status_code == 422
        assert "Every claim must be filled in" in response.json()["detail"]


def test_endpoint_rejects_an_unknown_product_before_streaming():
    with patch("app.main.verify_claim", side_effect=ok_result) as judge:
        response = post_batch(["a claim"], product_id="not_a_real_product")
    assert response.status_code == 404
    judge.assert_not_called()
