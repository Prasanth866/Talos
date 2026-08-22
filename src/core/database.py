from __future__ import annotations

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
        self.engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            future=True,
        )
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            autoflush=False,
        )

    async def dispose(self) -> None:
        """Disposes the engine connection pool cleanly."""
        await self.engine.dispose()
