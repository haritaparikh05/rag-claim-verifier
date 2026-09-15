import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval import retrieve_chunks

load_dotenv()

# Pinned rather than using the "-latest" alias: that alias resolves to
# Google's newest flagship-adjacent model, which on the free tier has a much
# stricter daily quota (20 requests/day, hit during eval testing) than the
# lighter "flash-lite" line (reported ~1000/day). This is a fast
# classification task, not one that needs flagship reasoning, so flash-lite
# is the better fit anyway.
MODEL = "gemini-3.5-flash-lite"
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2
DEFAULT_TOP_K = 5

_client = None

VERDICT_PROMPT_TEMPLATE = """You are verifying whether a product claim is supported by that product's official documentation.

Claim: "{claim}"

Retrieved passages from the product's official documentation:
{passages}

Based ONLY on the passages above (not on any outside knowledge), classify the claim as one of:
- "supported": the passages directly confirm the claim
- "contradicted": the passages directly conflict with the claim
- "unverifiable": the passages don't contain enough information to confirm or deny the claim

When comparing numbers, remember that meeting or exceeding a documented threshold satisfies a claim of that threshold or less: if the documentation says a spec is "at least X" or "up to X", a claim of X or anything less than X is SUPPORTED, not contradicted, since exceeding a minimum or staying under a maximum still meets it. Only mark a numeric claim as "contradicted" when it asserts a value the documentation shows is not actually met (e.g. claiming MORE than a stated maximum, or LESS than a stated minimum).

A passage only counts as evidence if it actually addresses the same specific attribute as the claim - not merely because it shares a keyword. For example, a safety warning about not letting a power cord touch a stovetop says nothing about whether the appliance itself is compatible with induction cooktops as a cooking surface; sharing the word "stovetop" does not make it relevant evidence. If the passages don't genuinely speak to the claim's actual subject, the verdict is "unverifiable", not "contradicted".

Respond with ONLY a JSON object in this exact format, no other text, no markdown formatting:
{{
  "verdict": "supported" | "contradicted" | "unverifiable",
  "confidence": <number between 0 and 1>,
  "cited_passage": "<the exact quoted text from the passages above that most directly supports your verdict, or an empty string if unverifiable>",
  "explanation": "<one sentence explaining the verdict>"
}}
"""


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(claim: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(
        f"[Passage {i + 1}] (from {c['source_file']})\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )
    return VERDICT_PROMPT_TEMPLATE.format(claim=claim, passages=passages)


def parse_verdict_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _generate_with_retry(client: genai.Client, model: str, prompt: str):
    """The free Gemini tier occasionally returns 503 (server overload) or 429
    (rate limit) - both transient, not real failures, so retry with backoff
    instead of surfacing them as a crash. Other errors (bad request, auth)
    are raised immediately since retrying won't fix them."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except genai_errors.APIError as e:
            if e.code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            print(f"Gemini API busy ({e.code}), retrying in {delay}s...", file=sys.stderr)
            time.sleep(delay)


def verify_claim(product_id: str, claim: str, top_k: int = DEFAULT_TOP_K) -> dict:
    chunks = retrieve_chunks(product_id, claim, top_k=top_k)
    if not chunks:
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "cited_passage": "",
            "explanation": f"No documentation found for product '{product_id}'.",
            "retrieved_chunks": [],
        }

    prompt = build_prompt(claim, chunks)
    client = get_client()
    response = _generate_with_retry(client, MODEL, prompt)

    try:
        result = parse_verdict_response(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        result = {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "cited_passage": "",
            "explanation": f"Failed to parse model response: {e}",
        }

    result["retrieved_chunks"] = chunks
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Verify a claim against a product's documentation")
    parser.add_argument("product_id", help="e.g. black_diamond_spot_400")
    parser.add_argument("claim", help="e.g. 'waterproof to 1 meter for 30 minutes'")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    result = verify_claim(args.product_id, args.claim, args.top_k)

    print(f"Claim: {args.claim!r}")
    print(f"Product: {args.product_id}\n")
    print(f"Verdict: {result['verdict'].upper()}")
    print(f"Confidence: {result['confidence']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Cited passage: {result['cited_passage']!r}")


if __name__ == "__main__":
    main()
