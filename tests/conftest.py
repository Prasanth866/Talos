from collections.abc import Generator

import pytest
from starlette.testclient import TestClient

from src.main import app


@pytest.fixture
def client() -> Generator[TestClient]:
    """Yields a TestClient with proper ASGI lifespan management."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
