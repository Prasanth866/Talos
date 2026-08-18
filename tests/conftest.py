from starlette.testclient import TestClient

from src.main import app


def client() -> TestClient:
    """Returns a TestClient that does not raise server exceptions."""
    return TestClient(app, raise_server_exceptions=False)
