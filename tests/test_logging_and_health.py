import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.logging_config import JsonFormatter
from app.retrieval import check_database


def test_json_formatter_emits_valid_json_with_extra_fields():
    record = logging.LogRecord("app", logging.INFO, __file__, 1, "verify", None, None)
    record.fields = {"product_id": "jbl_flip_6", "latency_ms": 12}

    parsed = json.loads(JsonFormatter().format(record))

    assert parsed["message"] == "verify"
    assert parsed["level"] == "INFO"
    assert parsed["product_id"] == "jbl_flip_6"
    assert parsed["latency_ms"] == 12
    assert "ts" in parsed


def test_check_database_false_when_connection_fails():
    with patch("app.retrieval.get_db_connection", side_effect=psycopg2.OperationalError("down")):
        assert check_database() is False


def test_check_database_true_when_query_succeeds():
    conn = MagicMock()
    with patch("app.retrieval.get_db_connection", return_value=conn):
        assert check_database() is True
    conn.close.assert_called_once()
