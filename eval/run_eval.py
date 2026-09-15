import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verdict import verify_claim
from eval.metrics import compute_metrics, print_metrics_report

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_results.json"

DELAY_BETWEEN_CALLS_SECONDS = 4  # stay under the Gemini free tier's ~15 requests/minute limit


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    for i, case in enumerate(eval_set, start=1):
        product_id = case["product_id"]
        claim = case["claim"]
        expected = case["expected_verdict"]

        print(f"[{i}/{len(eval_set)}] {product_id!r} - {claim!r} (expected: {expected})")
        verdict_result = verify_claim(product_id, claim)
        actual = verdict_result["verdict"]
        match = actual == expected
        print(f"    -> {actual} [{'PASS' if match else 'FAIL'}]")

        results.append(
            {
                "product_id": product_id,
                "claim": claim,
                "expected_verdict": expected,
                "actual_verdict": actual,
                "confidence": verdict_result["confidence"],
                "cited_passage": verdict_result["cited_passage"],
                "match": match,
            }
        )

        if i < len(eval_set):
            time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    metrics = compute_metrics(results)
    print_metrics_report(results, metrics)

    failures = [r for r in results if not r["match"]]
    if failures:
        print(f"\n=== {len(failures)} failure(s) ===")
        for r in failures:
            print(
                f"  [{r['product_id']}] {r['claim']!r}: "
                f"expected {r['expected_verdict']}, got {r['actual_verdict']}"
            )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)
    print(f"\nSaved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
