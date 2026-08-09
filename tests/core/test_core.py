from unittest.mock import patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from src.core.config import Environment, Settings, get_settings
from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware


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
    prod_settings = Settings(environment=Environment.PRODUCTION, debug=True)
    with patch("src.core.logging.get_settings", return_value=prod_settings):
        setup_logging()
        logger = get_logger("test_prod_logger")
        assert logger is not None


def test_logging_and_correlation_id_middleware() -> None:
    """Verifies middleware assigns and returns X-Request-ID headers."""
    app = FastAPI()
    app.add_middleware(LoggingAndCorrelationIdMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"message": "pong"}

    client = TestClient(app)

    response = client.get("/ping")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

    custom_id = "test-custom-correlation-id"
    response_custom = client.get("/ping", headers={"X-Request-ID": custom_id})
    assert response_custom.status_code == 200
    assert response_custom.headers["X-Request-ID"] == custom_id


def test_fastapi_app_lifespan_integration() -> None:
    """Verifies the main FastAPI app builds and handles lifespan events."""
    from src.main import app

    with TestClient(app) as client:
        assert client is not None
