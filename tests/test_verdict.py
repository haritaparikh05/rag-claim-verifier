import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.verdict import build_prompt, parse_verdict_response, verify_claim

# Nothing here touches the real database or the real Gemini API - retrieval
# and the model client are mocked, so this suite is fast, free, and doesn't
# consume API quota or depend on Postgres being up.


def test_build_prompt_includes_claim_and_passages():
    chunks = [{"source_file": "manual.pdf", "chunk_index": 0, "chunk_text": "IPX8 waterproof to 1.1 meters."}]
    prompt = build_prompt("waterproof to 1 meter", chunks)
    assert "waterproof to 1 meter" in prompt
    assert "IPX8 waterproof to 1.1 meters." in prompt
    assert "manual.pdf" in prompt


def test_parse_verdict_response_plain_json():
    text = '{"verdict": "supported", "confidence": 0.9, "cited_passage": "x", "explanation": "y"}'
    result = parse_verdict_response(text)
    assert result["verdict"] == "supported"
    assert result["confidence"] == 0.9


def test_parse_verdict_response_strips_markdown_fence():
    text = '```json\n{"verdict": "contradicted", "confidence": 1.0, "cited_passage": "x", "explanation": "y"}\n```'
    result = parse_verdict_response(text)
    assert result["verdict"] == "contradicted"


def test_verify_claim_returns_unverifiable_when_no_chunks_found():
    with patch("app.verdict.retrieve_chunks", return_value=[]):
        result = verify_claim("unknown_product", "some claim")
    assert result["verdict"] == "unverifiable"
    assert result["retrieved_chunks"] == []


def test_verify_claim_calls_gemini_and_returns_parsed_verdict():
    fake_chunks = [{"source_file": "manual.pdf", "chunk_index": 0, "chunk_text": "IPX8 waterproof."}]
    fake_response = MagicMock()
    fake_response.text = (
        '{"verdict": "supported", "confidence": 0.95, '
        '"cited_passage": "IPX8 waterproof.", "explanation": "matches"}'
    )

    with (
        patch("app.verdict.retrieve_chunks", return_value=fake_chunks),
        patch("app.verdict.get_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        result = verify_claim("black_diamond_spot_400", "waterproof to 1 meter")

    assert result["verdict"] == "supported"
    assert result["confidence"] == 0.95
    assert result["retrieved_chunks"] == fake_chunks
    mock_client.models.generate_content.assert_called_once()


def test_verify_claim_handles_unparseable_model_response_gracefully():
    fake_chunks = [{"source_file": "manual.pdf", "chunk_index": 0, "chunk_text": "some text"}]
    fake_response = MagicMock()
    fake_response.text = "not valid json at all"

    with (
        patch("app.verdict.retrieve_chunks", return_value=fake_chunks),
        patch("app.verdict.get_client") as mock_get_client,
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response
        mock_get_client.return_value = mock_client

        result = verify_claim("some_product", "some claim")

    assert result["verdict"] == "unverifiable"
    assert "Failed to parse" in result["explanation"]
