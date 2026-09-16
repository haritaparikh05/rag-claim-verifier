import os
import sys
from pathlib import Path

import psycopg2
import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()


def _db_reachable() -> bool:
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "claimverifier"),
            user=os.getenv("DB_USER", "claimverifier"),
            password=os.getenv("DB_PASSWORD", "claimverifier"),
            connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_db():
    """Skip a test instead of failing it when Postgres isn't running locally -
    these are integration tests that depend on the real database + ingested
    data, not something to fake with a mock."""
    if not _db_reachable():
        pytest.skip("Postgres is not reachable - start it with `docker compose up -d`")
