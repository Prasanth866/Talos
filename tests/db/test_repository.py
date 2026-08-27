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
async def test_create_and_get_task(db_session: AsyncSession) -> None:
    task = await repository.create_task(
        session=db_session,
        task_id="test-task-1",
        task="Create a hello world function",
        metadata={"user": "developer"},
    )
    assert task.id == "test-task-1"
    assert task.task == "Create a hello world function"
    assert task.status == TaskStatus.PENDING.value
    assert task.metadata_json == {"user": "developer"}
    assert task.created_at is not None

    fetched = await repository.get_task(db_session, "test-task-1")
    assert fetched is not None
    assert fetched.id == "test-task-1"
    assert fetched.status == TaskStatus.PENDING.value


@pytest.mark.asyncio
async def test_get_nonexistent_task(db_session: AsyncSession) -> None:
    task = await repository.get_task(db_session, "nonexistent-id")
    assert task is None


@pytest.mark.asyncio
async def test_mark_running(db_session: AsyncSession) -> None:
    await repository.create_task(db_session, "task-run", "Run task")
    await repository.mark_running(db_session, "task-run")

    fetched = await repository.get_task(db_session, "task-run")
    assert fetched is not None
    assert fetched.status == TaskStatus.RUNNING.value
    assert fetched.started_at is not None


@pytest.mark.asyncio
async def test_mark_completed(db_session: AsyncSession) -> None:
    await repository.create_task(db_session, "task-comp", "Complete task")
    await repository.mark_running(db_session, "task-comp")
    await repository.mark_completed(
        session=db_session,
        task_id="task-comp",
        result="Success answer",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        total_cost_usd=0.0025,
        duration_seconds=1.23,
    )

    fetched = await repository.get_task(db_session, "task-comp")
    assert fetched is not None
    assert fetched.status == TaskStatus.COMPLETED.value
    assert fetched.result == "Success answer"
    assert fetched.prompt_tokens == 100
    assert fetched.completion_tokens == 50
    assert fetched.total_tokens == 150
    assert fetched.total_cost_usd == 0.0025
    assert fetched.duration_seconds == 1.23
    assert fetched.completed_at is not None


@pytest.mark.asyncio
async def test_mark_failed(db_session: AsyncSession) -> None:
    await repository.create_task(db_session, "task-fail", "Failing task")
    await repository.mark_running(db_session, "task-fail")
    await repository.mark_failed(
        session=db_session,
        task_id="task-fail",
        error="LLM context limit exceeded",
    )

    fetched = await repository.get_task(db_session, "task-fail")
    assert fetched is not None
    assert fetched.status == TaskStatus.FAILED.value
    assert fetched.error == "LLM context limit exceeded"
    assert fetched.completed_at is not None


@pytest.mark.asyncio
async def test_list_tasks_and_filters(db_session: AsyncSession) -> None:
    await repository.create_task(db_session, "task-1", "Task 1")
    await repository.create_task(db_session, "task-2", "Task 2")
    await repository.create_task(db_session, "task-3", "Task 3")

    await repository.mark_running(db_session, "task-2")
    await repository.mark_completed(db_session, "task-2", result="Done")
    await repository.mark_failed(db_session, "task-3", error="Failed")

    all_tasks = await repository.list_tasks(db_session)
    assert len(all_tasks) == 3

    pending_tasks = await repository.list_tasks(
        db_session, status=TaskStatus.PENDING.value
    )
    assert len(pending_tasks) == 1
    assert pending_tasks[0].id == "task-1"

    completed_tasks = await repository.list_tasks(
        db_session, status=TaskStatus.COMPLETED.value
    )
    assert len(completed_tasks) == 1
    assert completed_tasks[0].id == "task-2"

    failed_tasks = await repository.list_tasks(
        db_session, status=TaskStatus.FAILED.value
    )
    assert len(failed_tasks) == 1
    assert failed_tasks[0].id == "task-3"

    paginated = await repository.list_tasks(db_session, limit=2, offset=1)
    assert len(paginated) == 2
