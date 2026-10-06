# LLM judge vs a dedicated NLI model

This page is the full write-up of an offline experiment. The deployed app does not use any of it; the code lives in `eval/` and the results in `data/eval/`.

## The question

The verdict step uses a large language model (Gemini) to decide whether retrieved passages support a claim. Is an LLM needed for that, or would a small purpose-built model do?

Natural language inference (NLI) models are trained to label a premise and a hypothesis as entailment, contradiction or neutral, which maps directly onto supported, contradicted and unverifiable. They run locally, with no API and no quota. I compared two of them against Gemini on the same retrieved chunks and on the same FEVER samples.

## How the NLI judge works

For each retrieved chunk, the model scores the chunk (premise) against the claim (hypothesis). If the strongest entailment or contradiction probability across all chunks reaches 0.5, that decides the verdict and that chunk is the citation; otherwise the claim is unverifiable. This rule was fixed before any results were seen and was never tuned (`eval/nli_judge.py`).

Models:
- **NLI base:** `cross-encoder/nli-deberta-v3-base`, trained on SNLI and MultiNLI only (not FEVER).
- **NLI large:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, trained on several harder datasets including FEVER and adversarial ANLI, so it is not evaluated on FEVER.

## The evaluation sets

- **Dev set:** 27 product claims (one supported, one contradicted and one unverifiable per product). The Gemini prompt rules were developed against these cases, so Gemini's score here is optimistic.
- **Held-out set:** 27 fresh claims written afterwards, with the exact source line recorded for each. The author verified every label before any judge was run on it. Nothing was ever tuned on this set (`data/eval/eval_set_holdout.json`).
- **FEVER:** three random samples of 300 examples each (seeds 42, 777, 2024), the same examples for both judges.

## Results

Product claims (same retrieval for every judge):

| Judge | Dev set | Held-out set | Time per example |
|---|---|---|---|
| Gemini (the app's judge) | 100% | **92.6%** (25/27) | about 1 to 25 s, varies |
| NLI base | 63.0% | 55.6% (15/27) | 0.45 s |
| NLI large | 85.2% | 81.5% (22/27) | 9.7 s on a laptop CPU |
| NLI large, scoring only the top 3 chunks | not run | 77.8% (21/27) | 2.9 s |

FEVER (300 examples per sample, base NLI model only):

| Seed | Gemini | NLI base | Recall on "contradicted" (Gemini / NLI) | Recall on "unverifiable" (Gemini / NLI) |
|---|---|---|---|---|
| 42 | 75.67% | 65.67% | 81% / 85% | 54% / 30% |
| 777 | 75.33% | 64.67% | 85% / 84% | 53% / 29% |
| 2024 | 74.33% | 64.67% | 85% / 87% | 52% / 31% |

## What this shows

- **The LLM judge is more accurate, mostly on "not mentioned".** On FEVER, the NLI base model matches Gemini at spotting contradictions but finds only about 30% of the unverifiable cases, against about 53% for Gemini. NLI models tend to call a statement about an unrelated attribute a contradiction, which is exactly what the "a passage must address the same attribute" rule in the Gemini prompt prevents.
- **Noisy chunks hurt the NLI judge.** Taking the single strongest signal across all chunks lets irrelevant text (a shopping-cart sidebar, a table of contents) produce a confident wrong answer. On the dev set, the remaining large-model errors all cited such text.
- **A bigger NLI model closes much of the gap, but with 27 cases it is not a significant difference.** The large model's 81.5% against Gemini's 92.6% has overlapping 95% confidence ranges (roughly 63 to 92% vs 77 to 98%). The base model is clearly worse than Gemini.
- **Gemini's 100% on the dev set was optimistic.** On fresh claims it scored 92.6%, which is the number to quote. The large NLI model dropped much less (85.2% to 81.5%) because it was never tuned on the dev set.
- **The one improvement tried in advance did not help.** I decided before running it that scoring only the top 3 chunks should reduce false contradictions from noisy chunks. It lowered the score (81.5% to 77.8%): it found more unverifiable claims but missed contradictions when the supporting chunk ranked below the third. It is reported as a negative result and no further tuning was done. (A second idea, requiring the winning label to beat "neutral", was dropped before running because at a 0.5 threshold it can never change a result.)
- **Accuracy and speed trade off.** The base model is fast and weak. The large model is roughly as slow as the Gemini call on a CPU. Neither produces an explanation, only a label and a score.
- **Conclusion.** For this task the explicit instructions in the prompt (same attribute, threshold reasoning, no outside knowledge) matter as much as model size. A local NLI model could serve as a cheap first pass, but it is not a drop-in replacement for the LLM judge.

## Caveats

- Both product sets are small and written by the project author. One case is worth 3.7 points, so differences of one or two cases are noise.
- Latency was measured on one laptop CPU; a GPU or a different batching strategy would change the NLI numbers.
- The large model's FEVER score is not reported because it was trained on FEVER data.

## Reproduce

```bash
./venv/bin/python -m eval.run_eval --set holdout
./venv/bin/python -m eval.run_nli_eval --task product --set holdout
./venv/bin/python -m eval.run_nli_eval --task product --set holdout --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
./venv/bin/python -m eval.run_nli_eval --task fever --seed 42
./venv/bin/python -m eval.compare_judges
```
