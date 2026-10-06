import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.metrics import VERDICTS

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
FEVER_SEEDS = [42, 777, 2024]
DEFAULT_SEED = 42
LARGE_MODEL = "DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def load(path: Path) -> dict | None:
    if not path.exists():
        print(f"(missing: {path.name})")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def accuracy(results: list[dict]) -> float:
    return sum(r["match"] for r in results) / len(results)


def recall_by_class(results: list[dict]) -> dict:
    out = {}
    for v in VERDICTS:
        rows = [r for r in results if r["expected_verdict"] == v]
        out[v] = sum(r["match"] for r in rows) / len(rows) if rows else float("nan")
    return out


def product_files(set_name: str) -> dict[str, str]:
    tag = "" if set_name == "dev" else f"_{set_name}"
    return {
        "Gemini": "eval_results.json" if set_name == "dev" else f"eval_results_{set_name}.json",
        "NLI base": f"nli_eval_results{tag}.json",
        "NLI large": f"nli_eval_results{tag}_{LARGE_MODEL}.json",
        "NLI large, top 3 chunks": f"nli_eval_results{tag}_{LARGE_MODEL}_top3.json",
    }


def compare_product() -> None:
    titles = {
        "dev": "Dev set (27 cases the Gemini prompt was developed against)",
        "holdout": "Held-out set (27 fresh cases, never used to change any prompt or rule)",
    }
    for set_name, title in titles.items():
        loaded = {label: load(EVAL_DIR / name) for label, name in product_files(set_name).items()}
        loaded = {label: data for label, data in loaded.items() if data}
        if not loaded:
            continue

        keyed = {label: {(r["product_id"], r["claim"]): r for r in d["results"]} for label, d in loaded.items()}
        common = set.intersection(*(set(k) for k in keyed.values()))

        print(f"\n=== {title}: {len(common)} cases, same retrieval, different judge ===")
        print(f"{'judge':26s} | {'accuracy':>8} | {'supp':>5} {'contr':>5} {'unver':>5} (recall) | mean ms")
        for label, rows in keyed.items():
            picked = [rows[k] for k in common]
            rec = recall_by_class(picked)
            latency = loaded[label].get("mean_latency_ms")
            lat = f"{latency:.0f}" if latency is not None else "-"
            print(
                f"{label:26s} | {accuracy(picked):>8.1%} | {rec['supported']:>5.0%} {rec['contradicted']:>5.0%} "
                f"{rec['unverifiable']:>5.0%}          | {lat}"
            )

        nli_labels = [label for label in keyed if label != "Gemini"]
        if nli_labels:
            best = nli_labels[-1]
            wrong = [k for k in sorted(common) if not keyed[best][k]["match"]]
            if wrong:
                print(f"\nCases '{best}' got wrong:")
                for k in wrong:
                    r = keyed[best][k]
                    gem = keyed.get("Gemini", {}).get(k, {}).get("actual_verdict", "n/a")
                    print(f"  [{k[0]}] {k[1]!r}: expected {r['expected_verdict']}, got {r['actual_verdict']} (Gemini: {gem})")


def compare_fever() -> None:
    print("\n=== FEVER samples (300 examples each, same examples for both judges) ===")
    print(f"{'seed':>6} | {'Gemini acc':>10} | {'NLI acc':>8} | {'NLI mean ms':>11} | per-class recall (Gemini / NLI)")
    for seed in FEVER_SEEDS:
        suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
        gemini = load(EVAL_DIR / f"fever_eval_results{suffix}.json")
        nli = load(EVAL_DIR / f"nli_fever_eval_results{suffix}.json")
        if not gemini or not nli:
            continue

        g = {r["id"]: r for r in gemini["results"]}
        n = {r["id"]: r for r in nli["results"]}
        ids = [i for i in g if i in n]
        gr = recall_by_class([g[i] for i in ids])
        nr = recall_by_class([n[i] for i in ids])
        recalls = "  ".join(f"{v[:4]} {gr[v]:.0%}/{nr[v]:.0%}" for v in VERDICTS)
        print(
            f"{seed:>6} | {accuracy([g[i] for i in ids]):>10.2%} | {accuracy([n[i] for i in ids]):>8.2%} | "
            f"{nli['mean_latency_ms']:>11.0f} | {recalls}"
        )


if __name__ == "__main__":
    compare_product()
    compare_fever()
