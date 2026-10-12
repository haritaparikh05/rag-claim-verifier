import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient

from app.main import app
from app.results import load_results, summarize, wilson_interval

client = TestClient(app)


def case(expected, actual, claim="a claim"):
    return {
        "product_id": "jbl_flip_6",
        "claim": claim,
        "expected_verdict": expected,
        "actual_verdict": actual,
        "match": expected == actual,
    }


def write_results(directory, filename, results):
    (directory / filename).write_text(json.dumps({"results": results}), encoding="utf-8")


def test_wilson_interval_matches_a_hand_computed_value():
    low, high = wilson_interval(25, 27)
    assert low == pytest.approx(0.766, abs=0.005)
    assert high == pytest.approx(0.980, abs=0.005)


def test_wilson_interval_for_a_perfect_score_is_capped_at_one():
    low, high = wilson_interval(27, 27)
    assert high == 1.0
    assert 0.85 < low < 0.90


def test_wilson_interval_with_no_cases_is_empty():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_summarize_counts_correct_and_per_class():
    results = [
        case("supported", "supported"),
        case("supported", "contradicted"),
        case("contradicted", "contradicted"),
        case("unverifiable", "unverifiable"),
    ]
    summary = summarize(results)
    assert summary["cases"] == 4
    assert summary["correct"] == 3
    assert summary["accuracy"] == 0.75
    assert summary["per_class"]["supported"] == {"correct": 1, "total": 2}
    assert summary["per_class"]["contradicted"] == {"correct": 1, "total": 1}
    assert summary["per_class"]["unverifiable"] == {"correct": 1, "total": 1}
    assert summary["interval"][0] < 0.75 < summary["interval"][1]


def test_load_results_reads_the_files_and_lists_the_misses(tmp_path):
    write_results(
        tmp_path,
        "eval_results.json",
        [case("supported", "supported"), case("contradicted", "unverifiable", claim="the missed claim")],
    )
    write_results(tmp_path, "fever_eval_results.json", [case("supported", "supported")] * 4)
    write_results(tmp_path, "fever_eval_results_seed777.json", [case("supported", "supported")] * 3 + [case("supported", "contradicted")])

    data = load_results(tmp_path)

    dev, holdout = data["product_sets"]
    assert dev["available"] and dev["correct"] == 1 and dev["cases"] == 2
    assert dev["failures"] == [
        {"product_id": "jbl_flip_6", "claim": "the missed claim", "expected": "contradicted", "actual": "unverifiable"}
    ]
    assert holdout["available"] is False and holdout["failures"] == []

    assert [s["seed"] for s in data["fever"]["samples"]] == [42, 777]
    assert data["fever"]["mean_accuracy"] == pytest.approx((1.0 + 0.75) / 2)
    assert data["fever"]["spread_points"] == pytest.approx(25.0)


def test_notes_state_the_value_of_one_case_from_the_data(tmp_path):
    write_results(tmp_path, "eval_results.json", [case("supported", "supported")] * 4)
    notes = load_results(tmp_path)["notes"]
    assert any("one case is worth 25.0 percentage points" in note for note in notes)


def test_missing_files_are_reported_not_fatal(tmp_path):
    data = load_results(tmp_path)
    assert [s["available"] for s in data["product_sets"]] == [False, False]
    assert data["fever"]["samples"] == []
    assert "mean_accuracy" not in data["fever"]
    assert any("Wilson" in note for note in data["notes"])


def test_a_corrupt_file_is_treated_as_unavailable(tmp_path):
    (tmp_path / "eval_results.json").write_text("{not json", encoding="utf-8")
    assert load_results(tmp_path)["product_sets"][0]["available"] is False


def test_results_endpoint_serves_consistent_numbers_from_the_committed_files():
    response = client.get("/api/results")
    assert response.status_code == 200
    data = response.json()

    assert [s["key"] for s in data["product_sets"]] == ["dev", "holdout"]
    for entry in data["product_sets"]:
        assert entry["available"], entry["file"]
        assert 0 <= entry["correct"] <= entry["cases"]
        assert entry["accuracy"] == pytest.approx(entry["correct"] / entry["cases"])
        assert entry["interval"][0] <= entry["accuracy"] <= entry["interval"][1]
        assert len(entry["failures"]) == entry["cases"] - entry["correct"]

    assert len(data["fever"]["samples"]) == 3
    assert data["notes"]


def test_results_page_is_served():
    response = client.get("/results")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Results" in response.text
    assert client.get("/static/results.js").status_code == 200
