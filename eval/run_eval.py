import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verdict import verify_claim

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_results.json"

VERDICTS = ["supported", "contradicted", "unverifiable"]
DELAY_BETWEEN_CALLS_SECONDS = 4  # stay under the Gemini free tier's ~15 requests/minute limit


def compute_metrics(results: list[dict]) -> dict:
    per_class = {v: {"tp": 0, "fp": 0, "fn": 0} for v in VERDICTS}
    correct = 0

    for r in results:
        expected = r["expected_verdict"]
        actual = r["actual_verdict"]
        if actual == expected:
            correct += 1
            per_class[expected]["tp"] += 1
        else:
            if actual in per_class:
                per_class[actual]["fp"] += 1
            per_class[expected]["fn"] += 1

    metrics = {}
    for v, counts in per_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        metrics[v] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    metrics["overall_accuracy"] = correct / len(results) if results else None
    return metrics


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

    print("\n=== Results ===")
    for v in VERDICTS:
        m = metrics[v]
        p = f"{m['precision']:.2f}" if m["precision"] is not None else "n/a"
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "n/a"
        f1 = f"{m['f1']:.2f}" if m["f1"] is not None else "n/a"
        print(f"{v:15s} precision={p}  recall={r}  f1={f1}  (n={m['support']})")
    print(
        f"\nOverall accuracy: {metrics['overall_accuracy']:.2%} "
        f"({sum(r['match'] for r in results)}/{len(results)})"
    )

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
