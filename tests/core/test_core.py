import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from starlette.testclient import TestClient

from src.core.config import Environment, Settings, _find_root, get_settings
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
    prod_settings = Settings(
        environment=Environment.PRODUCTION,
        debug=False,
        llm_api_key=SecretStr("prod-key-123"),
    )
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
    """Verifies middleware generates a new UUID for invalid correlation IDs."""
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
    """Verifies middleware generates a valid UUID when X-Request-ID is omitted."""
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
    """Verifies unhandled exception handler returns 500 with X-Request-ID."""

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


def test_find_root_with_app_root_override(tmp_path: Path) -> None:
    """Verifies _find_root respects APP_ROOT environment variable."""
    with patch.dict("os.environ", {"APP_ROOT": str(tmp_path)}):
        root = _find_root()
        assert root == tmp_path.resolve()


def test_find_root_raises_when_marker_missing() -> None:
    """Verifies _find_root raises FileNotFoundError when marker is absent."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(FileNotFoundError, match="not found in any parent"),
    ):
        _find_root(marker="non_existent_marker_file_12345.xyz")


def test_tool_error_handler_responses() -> None:
    """Verifies tool error handler maps exception classes to expected status codes."""
    from src.tools.exceptions import (
        CommandExecutionError,
        ExecutionTimeoutError,
        PathTraversalError,
        ToolError,
    )

    test_app = FastAPI()
    test_app.add_middleware(LoggingAndCorrelationIdMiddleware)

    from src.api.exception_handlers import tool_error_handler

    test_app.add_exception_handler(ToolError, tool_error_handler)  # type: ignore[arg-type]

    @test_app.get("/tool-error")
    def trigger_tool_error() -> None:
        raise ToolError("Generic error", tool_name="TestTool")

    @test_app.get("/traversal-error")
    def trigger_traversal() -> None:
        raise PathTraversalError(
            "Access denied", tool_name="FileSystemTool", attempted_path="/etc/passwd"
        )

    @test_app.get("/timeout-error")
    def trigger_timeout() -> None:
        raise ExecutionTimeoutError(
            "Timed out", tool_name="ShellTool", timeout_seconds=5.0
        )

    @test_app.get("/command-error")
    def trigger_command() -> None:
        raise CommandExecutionError(
            "Execution failed", tool_name="ShellTool", exit_code=1, stderr="err"
        )

    client = TestClient(test_app, raise_server_exceptions=False)

    resp1 = client.get("/tool-error")
    assert resp1.status_code == 400
    assert resp1.json()["error"] == "ToolError"
    assert "X-Request-ID" in resp1.headers
    assert uuid.UUID(resp1.headers["X-Request-ID"])

    resp2 = client.get("/traversal-error")
    assert resp2.status_code == 403
    assert resp2.json()["error"] == "PathTraversalError"
    assert "X-Request-ID" in resp2.headers
    assert uuid.UUID(resp2.headers["X-Request-ID"])

    resp3 = client.get("/timeout-error")
    assert resp3.status_code == 504
    assert resp3.json()["error"] == "ExecutionTimeoutError"
    assert "X-Request-ID" in resp3.headers
    assert uuid.UUID(resp3.headers["X-Request-ID"])

    resp4 = client.get("/command-error")
    assert resp4.status_code == 422
    assert resp4.json()["error"] == "CommandExecutionError"
    assert "X-Request-ID" in resp4.headers
    assert uuid.UUID(resp4.headers["X-Request-ID"])


def test_health_endpoint() -> None:
    """Verifies the /health endpoint returns 200 with expected payload."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers
    assert uuid.UUID(response.headers["X-Request-ID"])


def test_settings_llm_api_key_is_secret_str() -> None:
    """Verifies llm_api_key is SecretStr and masked in repr."""
    settings = Settings(llm_api_key=SecretStr("sk-test-secret-key-123"))
    # SecretStr masks value in string representation
    assert "sk-test-secret-key-123" not in repr(settings)
    assert "sk-test-secret-key-123" not in str(settings.llm_api_key)
    # Actual value accessible via get_secret_value()
    assert settings.llm_api_key.get_secret_value() == "sk-test-secret-key-123"


def test_settings_requires_api_key_in_production() -> None:
    """Verifies that production/staging environments require LLM_API_KEY."""
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        Settings(environment=Environment.PRODUCTION, llm_api_key=SecretStr(""))

    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        Settings(environment=Environment.STAGING, llm_api_key=SecretStr(""))

    # Development allows empty key (for MockLLMClient testing)
    settings = Settings(environment=Environment.DEVELOPMENT, llm_api_key=SecretStr(""))
    assert settings.llm_api_key.get_secret_value() == ""
