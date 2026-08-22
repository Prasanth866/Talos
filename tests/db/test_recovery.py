from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db import repository
from src.db.models import Base, TaskStatus


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_recover_interrupted_tasks(db_session: AsyncSession) -> None:
    # Set up tasks in various states
    await repository.create_task(db_session, "t-pending", "Pending task")

    await repository.create_task(db_session, "t-running-1", "Running task 1")
    await repository.mark_running(db_session, "t-running-1")

    await repository.create_task(db_session, "t-running-2", "Running task 2")
    await repository.mark_running(db_session, "t-running-2")

    await repository.create_task(db_session, "t-completed", "Completed task")
    await repository.mark_running(db_session, "t-completed")
    await repository.mark_completed(db_session, "t-completed", result="All good")

    # Run crash recovery
    recovered_count = await repository.recover_interrupted(db_session)
    assert recovered_count == 2

    # Verify RUNNING tasks became FAILED
    t1 = await repository.get_task(db_session, "t-running-1")
    assert t1 is not None
    assert t1.status == TaskStatus.FAILED.value
    assert "interrupted by server shutdown or crash" in (t1.error or "")
    assert t1.completed_at is not None

    t2 = await repository.get_task(db_session, "t-running-2")
    assert t2 is not None
    assert t2.status == TaskStatus.FAILED.value

    # Verify PENDING and COMPLETED tasks remain untouched
    tp = await repository.get_task(db_session, "t-pending")
    assert tp is not None
    assert tp.status == TaskStatus.PENDING.value

    tc = await repository.get_task(db_session, "t-completed")
    assert tc is not None
    assert tc.status == TaskStatus.COMPLETED.value
    assert tc.result == "All good"

    # Subsequent recovery runs should return 0
    second_run = await repository.recover_interrupted(db_session)
    assert second_run == 0
