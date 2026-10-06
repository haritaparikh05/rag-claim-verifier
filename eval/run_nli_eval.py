import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval import retrieve_chunks
from eval.metrics import compute_metrics, print_metrics_report
from eval.nli_judge import DECISION_THRESHOLD, MODEL_NAME, NliJudge
from eval.run_fever_eval import DEFAULT_N_PER_CLASS, DEFAULT_SEED, build_sample

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


PRODUCT_SETS = {"dev": "eval_set.json", "holdout": "eval_set_holdout.json"}
RETRIEVAL_TOP_K = 8  # same retrieval as the Gemini pipeline; --top-k only limits how many chunks the judge scores


def run_product(judge: NliJudge, set_name: str, judge_top_k: int) -> list[dict]:
    """A labeled product set, with the same retrieval step the Gemini pipeline uses."""
    with open(EVAL_DIR / PRODUCT_SETS[set_name], encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    for i, case in enumerate(eval_set, start=1):
        chunks = retrieve_chunks(case["product_id"], case["claim"], top_k=RETRIEVAL_TOP_K)
        start = time.perf_counter()
        out = judge.judge(case["claim"], [c["chunk_text"] for c in chunks][:judge_top_k])
        latency_ms = (time.perf_counter() - start) * 1000

        match = out["verdict"] == case["expected_verdict"]
        print(f"[{i}/{len(eval_set)}] {case['product_id']!r} - {case['claim']!r}: "
              f"expected {case['expected_verdict']}, got {out['verdict']} [{'PASS' if match else 'FAIL'}]")
        results.append(
            {
                "product_id": case["product_id"],
                "claim": case["claim"],
                "expected_verdict": case["expected_verdict"],
                "actual_verdict": out["verdict"],
                "confidence": round(out["confidence"], 4),
                "cited_passage": out["cited_passage"],
                "latency_ms": round(latency_ms, 1),
                "match": match,
            }
        )
    return results


def run_fever(judge: NliJudge, n_per_class: int, seed: int) -> list[dict]:
    """The same stratified FEVER sample the Gemini judge was evaluated on (sampling is deterministic by seed)."""
    print("Loading FEVER (NLI format) dev split...")
    sample = build_sample(n_per_class, seed)
    print(f"Sampled {len(sample)} examples ({n_per_class} per class, seed={seed})")

    results = []
    for i, case in enumerate(sample, start=1):
        start = time.perf_counter()
        out = judge.judge(case["claim"], [case["evidence"]])
        latency_ms = (time.perf_counter() - start) * 1000

        match = out["verdict"] == case["expected_verdict"]
        if i % 25 == 0 or i == len(sample):
            print(f"[{i}/{len(sample)}] done")
        results.append(
            {
                "id": case["id"],
                "claim": case["claim"],
                "expected_verdict": case["expected_verdict"],
                "actual_verdict": out["verdict"],
                "confidence": round(out["confidence"], 4),
                "latency_ms": round(latency_ms, 1),
                "match": match,
            }
        )
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate the offline NLI judge (compare with the Gemini judge)")
    parser.add_argument("--task", choices=["product", "fever"], required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="FEVER sampling seed (same seeds as the Gemini runs)")
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--set", choices=sorted(PRODUCT_SETS), default="dev", help="product task only")
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help="product task only: score only the top-K retrieved chunks (default: all 8)",
    )
    args = parser.parse_args()

    judge = NliJudge(args.model)

    model_suffix = "" if args.model == MODEL_NAME else "_" + args.model.split("/")[-1]
    if args.task == "product":
        results = run_product(judge, args.set, args.top_k)
        set_tag = "" if args.set == "dev" else f"_{args.set}"
        topk_tag = "" if args.top_k == RETRIEVAL_TOP_K else f"_top{args.top_k}"
        results_path = EVAL_DIR / f"nli_eval_results{set_tag}{model_suffix}{topk_tag}.json"
    else:
        results = run_fever(judge, args.n_per_class, args.seed)
        seed_suffix = "" if args.seed == DEFAULT_SEED else f"_seed{args.seed}"
        results_path = EVAL_DIR / f"nli_fever_eval_results{seed_suffix}{model_suffix}.json"

    metrics = compute_metrics(results)
    print_metrics_report(results, metrics)
    mean_latency = statistics.mean(r["latency_ms"] for r in results)
    print(f"Mean judge latency: {mean_latency:.0f} ms per example (model scoring only, excludes retrieval)")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "decision_threshold": DECISION_THRESHOLD,
                "task": args.task,
                "seed": args.seed if args.task == "fever" else None,
                "set": args.set if args.task == "product" else None,
                "judge_top_k": args.top_k if args.task == "product" else None,
                "mean_latency_ms": round(mean_latency, 1),
                "results": results,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved detailed results to {results_path}")


if __name__ == "__main__":
    main()
