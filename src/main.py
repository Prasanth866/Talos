from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes.health import router as health_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.websocket import router as websocket_router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware
from src.core.worker import TaskManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fast_app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    logger.info("application_startup", status="initializing")
    settings = get_settings()

    loop_factory = getattr(fast_app.state, "reasoning_loop_factory", None)
    task_manager = TaskManager(settings=settings, loop_factory=loop_factory)
    fast_app.state.task_manager = task_manager

    async with asyncio.TaskGroup() as task_group:
        task_manager.start(task_group)
        logger.info(
            "worker_pool_started",
            workers=settings.worker_concurrency,
            queue_capacity=settings.task_queue_max_size,
        )
        yield
        logger.info("application_shutdown", status="draining_workers")
        await task_manager.stop()

    logger.info("application_shutdown", status="stopped")


app = FastAPI(
    title="Talos Agent API",
    description=(
        "Autonomous software engineering agent with tool-use reasoning loop "
        "and WebSocket streaming"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(LoggingAndCorrelationIdMiddleware)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(tasks_router)
app.include_router(websocket_router)
