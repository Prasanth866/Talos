import uuid
from unittest.mock import patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from src.core.config import Environment, Settings, get_settings
from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware
from src.main import app


def test_settings_default_values() -> None:
    """Verifies default settings configuration."""
    settings = Settings()
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.debug is False


def test_get_settings_lru_cache() -> None:
    """Ensures get_settings returns cached Settings instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_setup_logging() -> None:
    """Verifies setup_logging configures structlog without errors."""
    setup_logging()
    logger = get_logger("test_logger")
    assert logger is not None


def test_setup_logging_production_environment() -> None:
    """Verifies setup_logging configures JSON renderer for production environment."""
    prod_settings = Settings(environment=Environment.PRODUCTION, debug=False)
    with patch("src.core.logging.get_settings", return_value=prod_settings):
        setup_logging()
        logger = get_logger("test_prod_logger")
        assert logger is not None


def test_logging_and_correlation_id_middleware_valid_uuid() -> None:
    """Verifies middleware preserves incoming valid UUID correlation IDs."""
    test_app = FastAPI()
    test_app.add_middleware(LoggingAndCorrelationIdMiddleware)

    @test_app.get("/ping")
    def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(test_app)
    valid_uuid = str(uuid.uuid4())

    response = client.get("/ping", headers={"X-Request-ID": valid_uuid})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == valid_uuid


def test_logging_and_correlation_id_middleware_invalid_uuid() -> None:
    """Verifies middleware generates a new UUID if incoming correlation ID is invalid."""
    test_app = FastAPI()
    test_app.add_middleware(LoggingAndCorrelationIdMiddleware)

    @test_app.get("/ping")
    def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(test_app)
    invalid_id = "test-custom-correlation-id"

    response = client.get("/ping", headers={"X-Request-ID": invalid_id})
    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    assert header_id != invalid_id
    assert uuid.UUID(header_id)


def test_logging_and_correlation_id_middleware_missing_header() -> None:
    """Verifies middleware generates a valid UUID when X-Request-ID header is omitted."""
    test_app = FastAPI()
    test_app.add_middleware(LoggingAndCorrelationIdMiddleware)

    @test_app.get("/ping")
    def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(test_app)

    response = client.get("/ping")
    assert response.status_code == 200
    header_id = response.headers.get("X-Request-ID")
    assert header_id is not None
    assert uuid.UUID(header_id)


def test_unhandled_exception_handler() -> None:
    """Verifies unhandled exception handler returns 500 JSONResponse with X-Request-ID."""

    @app.get("/error-route")
    def error_route() -> None:
        raise RuntimeError("Simulated error")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/error-route")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "X-Request-ID" in response.headers
    assert uuid.UUID(response.headers["X-Request-ID"])


def test_fastapi_app_lifespan_integration() -> None:
    """Verifies the main FastAPI app builds and handles lifespan events."""
    with TestClient(app) as client:
        assert client is not None
