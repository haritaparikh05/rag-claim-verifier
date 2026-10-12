"""Evaluation numbers for the /results page, computed from the result files committed in data/eval."""

import json
import math
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"

VERDICTS = ["supported", "contradicted", "unverifiable"]

PRODUCT_SETS = [
    {
        "key": "dev",
        "title": "Development set",
        "file": "eval_results.json",
        "description": (
            "27 claims, three per product (one supported, one contradicted, one unverifiable), run through the full "
            "pipeline. The judge's prompt rules were written from this set's failures, so treat its score as a "
            "development result."
        ),
    },
    {
        "key": "holdout",
        "title": "Held-out set",
        "file": "eval_results_holdout.json",
        "description": (
            "27 fresh claims written after the prompt was fixed, with every label checked before any score was seen. "
            "Nothing was tuned on them. This is the number to quote for claims the system has not seen."
        ),
    },
]

FEVER_FILES = [
    ("fever_eval_results.json", 42),
    ("fever_eval_results_seed777.json", 777),
    ("fever_eval_results_seed2024.json", 2024),
]
FEVER_DESCRIPTION = (
    "Three random samples of 300 examples from FEVER, an independent fact-verification benchmark labeled by other "
    "people. It tests the judging logic on Wikipedia claims, not product accuracy, and bypasses retrieval."
)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion: a plausible range for the true accuracy given a small sample."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _load(eval_dir: Path, filename: str) -> list[dict] | None:
    try:
        with open(eval_dir / filename, encoding="utf-8") as f:
            results = json.load(f)["results"]
    except (OSError, ValueError, KeyError):
        return None
    return results or None


def summarize(results: list[dict]) -> dict:
    cases = len(results)
    correct = sum(1 for r in results if r["match"])
    low, high = wilson_interval(correct, cases)

    per_class = {}
    for verdict in VERDICTS:
        expected = [r for r in results if r["expected_verdict"] == verdict]
        per_class[verdict] = {"correct": sum(1 for r in expected if r["match"]), "total": len(expected)}

    return {
        "available": True,
        "cases": cases,
        "correct": correct,
        "accuracy": correct / cases,
        "interval": [low, high],
        "per_class": per_class,
    }


def _misses(results: list[dict]) -> list[dict]:
    return [
        {
            "product_id": r.get("product_id"),
            "claim": r["claim"],
            "expected": r["expected_verdict"],
            "actual": r["actual_verdict"],
        }
        for r in results
        if not r["match"]
    ]


def load_results(eval_dir: Path = EVAL_DIR) -> dict:
    product_sets = []
    for spec in PRODUCT_SETS:
        entry = {"key": spec["key"], "title": spec["title"], "description": spec["description"], "file": spec["file"]}
        results = _load(eval_dir, spec["file"])
        if results is None:
            entry.update(available=False, failures=[])
        else:
            entry.update(summarize(results), failures=_misses(results))
        product_sets.append(entry)

    samples = []
    for filename, seed in FEVER_FILES:
        results = _load(eval_dir, filename)
        if results is not None:
            samples.append({"seed": seed, "file": filename, **summarize(results)})

    fever = {"title": "FEVER benchmark sample", "description": FEVER_DESCRIPTION, "samples": samples}
    if samples:
        accuracies = [s["accuracy"] for s in samples]
        fever["mean_accuracy"] = sum(accuracies) / len(accuracies)
        fever["spread_points"] = (max(accuracies) - min(accuracies)) * 100

    return {"product_sets": product_sets, "fever": fever, "notes": _notes(product_sets, fever)}


def _notes(product_sets: list[dict], fever: dict) -> list[str]:
    notes = []
    sizes = [s["cases"] for s in product_sets if s["available"]]
    if sizes:
        notes.append(
            f"Both product sets are small and written by the project author, so one case is worth "
            f"{100 / min(sizes):.1f} percentage points. They are regression checks for the pipeline, not benchmarks."
        )
    notes.append(
        "The 95% range shown beside each score is a Wilson interval: a plausible range for the true accuracy given "
        "how few cases were tested. With about 27 cases it is wide, so a difference of one or two cases is noise."
    )
    if fever.get("samples"):
        notes.append(
            f"FEVER measures generic Wikipedia claims, so it checks the judging logic, not product accuracy. The "
            f"three samples differ by about {fever['spread_points']:.1f} percentage points; tuning stopped once "
            f"changes could not be told apart from that run-to-run noise."
        )
    notes.append(
        "Verdicts come from a hosted AI model, and its behavior can change over time: during development, an "
        "unchanged prompt began misclassifying one case after an upstream model update."
    )
    return notes
