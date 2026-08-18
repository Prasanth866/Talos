import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.logging import get_logger
from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)

logger = get_logger(__name__)


def _get_correlation_headers() -> dict[str, str]:
    """Helper to safely extract the correlation ID header from structlog context."""
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    if isinstance(correlation_id, str) and correlation_id:
        return {"X-Request-ID": correlation_id}
    return {}


async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
    """Maps ToolError subclasses to appropriate HTTP status codes."""
    status_code = 400
    if isinstance(exc, PathTraversalError):
        status_code = 403
    elif isinstance(exc, ExecutionTimeoutError):
        status_code = 504
    elif isinstance(exc, CommandExecutionError):
        status_code = 422

    logger.warning(
        "tool_execution_failed",
        error_type=exc.__class__.__name__,
        status_code=status_code,
        path=request.url.path,
        method=request.method,
        tool_name=getattr(exc, "tool_name", "UnknownTool"),
        error_message=str(exc),
        details=getattr(exc, "details", None),
    )

    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict(),
        headers=_get_correlation_headers() or None,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled exceptions and returns a generic 500 response."""
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=_get_correlation_headers() or None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers all exception handlers on the FastAPI app instance."""
    app.add_exception_handler(ToolError, tool_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
