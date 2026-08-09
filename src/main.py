from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fastapp: FastAPI):
    logger.info("application_startup", status="initializing")
    yield
    logger.info("application_shutdown", status="stopping")


app = FastAPI(lifespan=lifespan)

app.add_middleware(LoggingAndCorrelationIdMiddleware)
