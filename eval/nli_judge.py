"""Offline NLI judge: a purpose-built classifier used as an alternative to the Gemini judge in app/verdict.py.

Not used by the deployed app; it exists so the two judges can be compared on the same inputs.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Trained on SNLI + MultiNLI only (not FEVER), so a FEVER comparison is not contaminated by training data.
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 512

# Fixed before looking at any results and never tuned on the eval sets: a relation must have probability
# of at least 0.5 on some passage to count; otherwise the claim is treated as unverifiable.
DECISION_THRESHOLD = 0.5


def decide(scored_passages: list[dict], threshold: float = DECISION_THRESHOLD) -> dict:
    """Turn per-passage NLI probabilities into a verdict.

    Each item is {"text", "entailment", "contradiction", "neutral"}. The strongest entailment and the strongest
    contradiction across all passages compete; the larger one wins if it clears the threshold.
    """
    if not scored_passages:
        return {"verdict": "unverifiable", "confidence": 0.0, "cited_passage": ""}

    best_entail = max(scored_passages, key=lambda p: p["entailment"])
    best_contra = max(scored_passages, key=lambda p: p["contradiction"])
    strongest = max(best_entail["entailment"], best_contra["contradiction"])

    if strongest < threshold:
        return {"verdict": "unverifiable", "confidence": 1.0 - strongest, "cited_passage": ""}
    if best_entail["entailment"] >= best_contra["contradiction"]:
        return {"verdict": "supported", "confidence": best_entail["entailment"], "cited_passage": best_entail["text"]}
    return {"verdict": "contradicted", "confidence": best_contra["contradiction"], "cited_passage": best_contra["text"]}


class NliJudge:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

        self.label_index = {name.lower(): int(i) for i, name in self.model.config.id2label.items()}
        missing = {"entailment", "contradiction", "neutral"} - set(self.label_index)
        if missing:
            raise ValueError(f"{model_name} does not expose the labels {sorted(missing)}: {self.label_index}")

    @torch.no_grad()
    def score(self, claim: str, passages: list[str]) -> list[dict]:
        """NLI probabilities with each passage as the premise and the claim as the hypothesis."""
        inputs = self.tokenizer(
            passages,
            [claim] * len(passages),
            truncation="only_first",
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        return [
            {
                "text": text,
                "entailment": row[self.label_index["entailment"]].item(),
                "contradiction": row[self.label_index["contradiction"]].item(),
                "neutral": row[self.label_index["neutral"]].item(),
            }
            for text, row in zip(passages, probs)
        ]

    def judge(self, claim: str, passages: list[str]) -> dict:
        return decide(self.score(claim, passages))


if __name__ == "__main__":
    # Smoke test: loads the model and scores three toy pairs (expect entailment, contradiction, neutral).
    judge = NliJudge()
    pairs = [
        ("The speaker is IP67 waterproof and dustproof.", "IP67 waterproof"),
        ("Music play time: up to 12 hours.", "24 hours of battery life"),
        ("The speaker supports Bluetooth 5.1.", "has a built-in flashlight"),
    ]
    for premise, claim in pairs:
        scores = judge.score(claim, [premise])[0]
        verdict = decide([scores])["verdict"]
        print(
            f"{claim!r}: entail={scores['entailment']:.2f} contra={scores['contradiction']:.2f} "
            f"neutral={scores['neutral']:.2f} -> {verdict}"
        )
