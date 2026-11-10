"""Shared test setup.

Settings are read when app.core.config is first imported, so the test environment must be in place BEFORE any test
module imports the app. Setting it here (conftest loads first) means the whole suite can run together with
`pytest -q`, always on a throwaway SQLite database and the mock LLM, never on your real PostgreSQL.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/suite.db"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["RECOVER_ON_START"] = "false"
os.environ["EMBED_PROVIDER"] = "hash"  # no embedding model needed in tests
os.environ.setdefault("AUTH_REQUIRED", "false")  # tests/test_auth.py switches it back on for its own checks

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database():
    """Every test starts with empty tables."""
    from app import models  # noqa: F401  (registers the tables)
    from app.core.database import Base, engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
