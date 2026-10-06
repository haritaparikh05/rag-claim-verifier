import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.nli_judge import decide


def passage(text, entailment=0.0, contradiction=0.0):
    return {"text": text, "entailment": entailment, "contradiction": contradiction, "neutral": 1 - entailment - contradiction}


def test_no_passages_is_unverifiable():
    assert decide([])["verdict"] == "unverifiable"


def test_confident_entailment_is_supported_and_cites_that_passage():
    out = decide([passage("a", entailment=0.1), passage("b", entailment=0.9)])
    assert out["verdict"] == "supported"
    assert out["cited_passage"] == "b"
    assert out["confidence"] == 0.9


def test_confident_contradiction_is_contradicted_and_cites_that_passage():
    out = decide([passage("a", contradiction=0.95), passage("b")])
    assert out["verdict"] == "contradicted"
    assert out["cited_passage"] == "a"


def test_low_confidence_everywhere_is_unverifiable_with_no_citation():
    out = decide([passage("a", entailment=0.3, contradiction=0.2)])
    assert out["verdict"] == "unverifiable"
    assert out["cited_passage"] == ""


def test_stronger_relation_wins_when_both_clear_the_threshold():
    out = decide([passage("a", entailment=0.6), passage("b", contradiction=0.8)])
    assert out["verdict"] == "contradicted"
    assert out["cited_passage"] == "b"


def test_threshold_is_inclusive():
    assert decide([passage("a", entailment=0.5)], threshold=0.5)["verdict"] == "supported"
