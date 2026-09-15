VERDICTS = ["supported", "contradicted", "unverifiable"]


def compute_metrics(results: list[dict], verdicts: list[str] = VERDICTS) -> dict:
    """Precision/recall/F1 per class plus overall accuracy, from a list of
    {"expected_verdict": ..., "actual_verdict": ...} dicts."""
    per_class = {v: {"tp": 0, "fp": 0, "fn": 0} for v in verdicts}
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


def print_metrics_report(results: list[dict], metrics: dict, verdicts: list[str] = VERDICTS) -> None:
    print("\n=== Results ===")
    for v in verdicts:
        m = metrics[v]
        p = f"{m['precision']:.2f}" if m["precision"] is not None else "n/a"
        r = f"{m['recall']:.2f}" if m["recall"] is not None else "n/a"
        f1 = f"{m['f1']:.2f}" if m["f1"] is not None else "n/a"
        print(f"{v:15s} precision={p}  recall={r}  f1={f1}  (n={m['support']})")
    print(
        f"\nOverall accuracy: {metrics['overall_accuracy']:.2%} "
        f"({sum(r['match'] for r in results)}/{len(results)})"
    )
