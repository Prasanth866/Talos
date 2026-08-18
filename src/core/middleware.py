import time
import uuid

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger("http_request")


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class LoggingAndCorrelationIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()

        headers = Headers(scope=scope)
        incoming_id = headers.get("X-Request-ID")
        correlation_id = (
            incoming_id
            if incoming_id and _is_valid_uuid(incoming_id)
            else str(uuid.uuid4())
        )

        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=scope.get("root_path", "") + scope["path"],
            method=scope["method"],
        )

        start_time = time.perf_counter()
        status_code = 500
        logger.info("request_started")

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                res_headers = MutableHeaders(scope=message)
                res_headers.append("X-Request-ID", correlation_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "request_finished",
                status_code=status_code,
                duration_ms=process_time_ms,
            )
        except Exception:
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("request_failed", duration_ms=process_time_ms)
            raise
