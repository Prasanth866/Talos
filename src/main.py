from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes.health import router as health_router
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

register_exception_handlers(app)

app.include_router(health_router)
