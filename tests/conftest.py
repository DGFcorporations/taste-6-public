import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    """Fresh throwaway db per test + offline-deterministic embedder."""
    db = tmp_path / "t.db"
    monkeypatch.setenv("TASTE6_DB", str(db))
    monkeypatch.setenv("TASTE_EMBED", "hash")  # deterministic, offline-fast
    return db


@pytest.fixture()
def store(db_path):
    from taste.store import Store

    return Store(db_path)


@pytest.fixture()
def engine(store):
    from taste.core import StandardsEngine
    from taste.embed import get_embedder

    get_embedder.cache_clear()
    return StandardsEngine(store)


@pytest.fixture()
def app(db_path):
    from taste.api import create_app

    return create_app(db_path)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
