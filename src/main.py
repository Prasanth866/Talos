from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_fast_app: FastAPI) -> AsyncGenerator[None]:
    logger.info("application_startup", status="initializing")
    yield
    logger.info("application_shutdown", status="stopping")


app = FastAPI(lifespan=lifespan)

app.add_middleware(LoggingAndCorrelationIdMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    if not isinstance(correlation_id, str):
        correlation_id = None
    response = JSONResponse(
        status_code=500, content={"detail": "Internal server error"}
    )
    if correlation_id:
        response.headers["X-Request-ID"] = correlation_id
    return response
