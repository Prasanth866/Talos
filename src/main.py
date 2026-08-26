from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.exception_handlers import register_exception_handlers
from src.api.routes.health import router as health_router
from src.api.routes.search import router as search_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.websocket import router as websocket_router
from src.core.config import ROOT_DIR, get_settings
from src.core.database import Database
from src.core.logging import get_logger, setup_logging
from src.core.middleware import LoggingAndCorrelationIdMiddleware
from src.core.worker import TaskManager
from src.db import repository
from src.db.models import Base

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fast_app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    logger.info("application_startup", status="initializing")
    settings = get_settings()

    db: Database = getattr(fast_app.state, "db", None) or Database(
        settings.database_url
    )
    fast_app.state.db = db

    async with db.engine.begin() as conn:
        if db.url.startswith("postgresql"):
            try:
                from sqlalchemy import text

                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            except Exception as exc:
                logger.debug("create_extension_vector_skipped", error=str(exc))
        try:
            await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn))
        except Exception as exc:
            logger.warning("create_all_tables_fallback_without_vector", error=str(exc))
            await conn.run_sync(
                lambda sync_conn: Base.metadata.create_all(
                    sync_conn,
                    tables=[
                        t
                        for name, t in Base.metadata.tables.items()
                        if name != "code_chunks"
                    ],
                )
            )

    async with db.session_factory() as session:
        recovered = await repository.recover_interrupted(session)
        if recovered > 0:
            logger.warning(
                "crash_recovery_completed",
                interrupted_tasks_marked_failed=recovered,
            )
        else:
            logger.info("crash_recovery_check", status="no_interrupted_tasks")

    loop_factory = getattr(fast_app.state, "reasoning_loop_factory", None)
    task_manager = TaskManager(
        settings=settings,
        loop_factory=loop_factory,
        session_factory=db.session_factory,
    )
    fast_app.state.task_manager = task_manager

    task_manager.start()
    logger.info(
        "worker_pool_started",
        workers=settings.worker_concurrency,
        queue_capacity=settings.task_queue_max_size,
    )
    yield
    logger.info("application_shutdown", status="draining_workers")
    await task_manager.stop()

    await db.dispose()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingAndCorrelationIdMiddleware)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(tasks_router)
app.include_router(websocket_router)
app.include_router(search_router)


static_dir = ROOT_DIR / "static"
if static_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/style.css", include_in_schema=False)
    async def serve_style_css() -> FileResponse:
        return FileResponse(static_dir / "style.css")

    @app.get("/app.js", include_in_schema=False)
    async def serve_app_js() -> FileResponse:
        return FileResponse(static_dir / "app.js")
