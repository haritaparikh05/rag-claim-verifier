import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verdict import MODEL, build_prompt, get_client, parse_verdict_response, _generate_with_retry
from eval.metrics import compute_metrics, print_metrics_report

PROGRESS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "fever_eval_progress.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "fever_eval_results.json"

# pietrolesci/nli_fever repackages FEVER as (premise, hypothesis, label) NLI
# pairs. Empirically verified (not just trusting the dataset card): `premise`
# is the short FEVER claim, `hypothesis` is the (often multi-sentence)
# Wikipedia evidence text - the reverse of what the card's own summary says.
LABEL_MAP = {0: "supported", 1: "unverifiable", 2: "contradicted"}
DELAY_BETWEEN_CALLS_SECONDS = 4  # stay under the Gemini free tier's ~15 requests/minute limit
SEED = 42  # fixed so the sampled set is reproducible run to run
DEFAULT_N_PER_CLASS = 100


def build_sample(n_per_class: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("pietrolesci/nli_fever", split="dev")

    by_label: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, label in enumerate(ds["label"]):
        if label in by_label:
            by_label[label].append(i)

    rng = random.Random(SEED)
    sample_indices = []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        sample_indices.extend(shuffled[:n_per_class])

    sample = []
    for idx in sample_indices:
        row = ds[idx]
        sample.append(
            {
                "id": idx,
                "claim": row["premise"],
                "evidence": row["hypothesis"],
                "expected_verdict": LABEL_MAP[row["label"]],
            }
        )
    return sample


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(list(progress.values()), f, indent=2)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate the verdict step against a stratified sample of FEVER (NLI-reformatted)"
    )
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    args = parser.parse_args()

    print("Loading FEVER (NLI format) dev split...")
    sample = build_sample(args.n_per_class)
    print(f"Sampled {len(sample)} examples ({args.n_per_class} per class, seed={SEED})")

    progress = load_progress()
    remaining = [case for case in sample if case["id"] not in progress]
    if progress:
        print(f"Resuming: {len(progress)} already done, {len(remaining)} remaining")

    client = get_client()

    for i, case in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] expected={case['expected_verdict']!r}", end="  ")
        chunk = {"source_file": "fever", "chunk_index": 0, "chunk_text": case["evidence"]}
        prompt = build_prompt(case["claim"], [chunk])

        try:
            response = _generate_with_retry(client, MODEL, prompt)
            result = parse_verdict_response(response.text)
            actual = result.get("verdict", "unverifiable")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        match = actual == case["expected_verdict"]
        print(f"-> {actual} [{'PASS' if match else 'FAIL'}]")

        progress[case["id"]] = {
            "id": case["id"],
            "claim": case["claim"],
            "expected_verdict": case["expected_verdict"],
            "actual_verdict": actual,
            "match": match,
        }

        if i % 5 == 0 or i == len(remaining):
            save_progress(progress)

        if i < len(remaining):
            time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    save_progress(progress)

    missing = [case["id"] for case in sample if case["id"] not in progress]
    if missing:
        print(f"\n{len(missing)} example(s) still missing (errored out) - re-run this script to fill them in.")

    results = [progress[case["id"]] for case in sample if case["id"] in progress]
    metrics = compute_metrics(results)
    print_metrics_report(results, metrics)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"n_per_class": args.n_per_class, "seed": SEED, "results": results, "metrics": metrics},
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
