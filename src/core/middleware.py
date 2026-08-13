import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger("http_request")


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class LoggingAndCorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        structlog.contextvars.clear_contextvars()

        incoming_id = request.headers.get("X-Request-ID")
        correlation_id = (
            incoming_id
            if incoming_id and _is_valid_uuid(incoming_id)
            else str(uuid.uuid4())
        )

        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )

        start_time = time.perf_counter()

        logger.info("request_started")

        try:
            response = await call_next(request)

            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "request_finished",
                status_code=response.status_code,
                duration_ms=process_time_ms,
            )

            response.headers["X-Request-ID"] = correlation_id
            return response

        except Exception:
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.error("request_failed", duration_ms=process_time_ms)
            raise
