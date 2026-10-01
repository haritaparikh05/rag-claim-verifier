import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Libraries (httpx, huggingface_hub, sentence_transformers, ...) log every request at INFO; keep them at WARNING
    # so only this app's own records use the configurable level.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # Unactionable per-request advisory about a Gemini feature this app doesn't use.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    # The request middleware logs every request as JSON, so uvicorn's plain-text access log would only duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
