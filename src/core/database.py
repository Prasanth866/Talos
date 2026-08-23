from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Manages the SQLAlchemy asynchronous engine and session factory."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = url
        connect_args: dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args = {"timeout": 30.0, "check_same_thread": False}

        self.engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            future=True,
            connect_args=connect_args,
        )

        if url.startswith("sqlite"):

            @event.listens_for(self.engine.sync_engine, "connect")
            def set_sqlite_pragma(
                dbapi_connection: Any,
                connection_record: Any,
            ) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.close()

        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    async def dispose(self) -> None:
        """Disposes the engine connection pool cleanly."""
        await self.engine.dispose()
