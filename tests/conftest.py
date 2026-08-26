from collections.abc import Generator

import pytest
from starlette.testclient import TestClient

from src.main import app


@pytest.fixture(autouse=True)
def _isolate_test_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures unit tests run against isolated in-memory SQLite."""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture
def client() -> Generator[TestClient]:
    """Yields a TestClient with proper ASGI lifespan management."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
