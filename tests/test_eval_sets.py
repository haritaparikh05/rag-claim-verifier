import json
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
VERDICTS = {"supported", "contradicted", "unverifiable"}


def load(name):
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def test_holdout_has_one_of_each_verdict_per_product():
    holdout = load("eval_set_holdout.json")
    assert len(holdout) == 27
    by_product = {}
    for case in holdout:
        assert case["expected_verdict"] in VERDICTS
        assert case["note"]
        by_product.setdefault(case["product_id"], []).append(case["expected_verdict"])
    assert len(by_product) == 9
    for product, verdicts in by_product.items():
        assert Counter(verdicts) == Counter(VERDICTS), product


def test_holdout_is_fresh_and_covers_the_same_products_as_dev():
    dev = load("eval_set.json")
    holdout = load("eval_set_holdout.json")

    assert {c["product_id"] for c in dev} == {c["product_id"] for c in holdout}

    dev_claims = {(c["product_id"], c["claim"].lower()) for c in dev}
    for case in holdout:
        assert (case["product_id"], case["claim"].lower()) not in dev_claims
