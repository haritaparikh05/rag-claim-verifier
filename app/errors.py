"""Turn low-level failures into an HTTP status and a message a person can act on."""

import psycopg2
from google.genai import errors as genai_errors

QUOTA_MESSAGE = (
    "The AI service has reached its free-tier limit for now. Wait a minute and try again; "
    "if it keeps happening, the daily limit may be used up."
)
BUSY_MESSAGE = "The AI service is busy right now. Please try again in a moment."
REJECTED_MESSAGE = "The AI service rejected the request. Check that GEMINI_API_KEY in the .env file is valid."
NO_KEY_MESSAGE = "No Gemini API key is configured. Add GEMINI_API_KEY to the .env file and restart."
DATABASE_MESSAGE = "The document database is not reachable. Make sure the database is running and try again."
NO_DOCUMENTS_MESSAGE = (
    "No documents are loaded for this product yet. Run the ingestion step "
    "(docker compose up, or python -m ingestion.ingest) and try again."
)


def friendly_error(exc: Exception) -> tuple[int, str] | None:
    """(http_status, message) for failures we know how to explain, or None for anything unexpected."""
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return 429, QUOTA_MESSAGE
        if exc.code in (500, 502, 503, 504):
            return 503, BUSY_MESSAGE
        if exc.code in (400, 401, 403):
            return 502, REJECTED_MESSAGE
        return 502, f"The AI service returned an unexpected error (code {exc.code})."
    if isinstance(exc, RuntimeError) and "GEMINI_API_KEY" in str(exc):
        return 503, NO_KEY_MESSAGE
    if isinstance(exc, psycopg2.Error):
        return 503, DATABASE_MESSAGE
    return None
