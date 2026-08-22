from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Task, TaskStatus


async def create_task(
    session: AsyncSession,
    task_id: str,
    task: str,
    metadata: dict[str, Any] | None = None,
) -> Task:
    """Inserts a new task in PENDING state."""
    db_task = Task(
        id=task_id,
        task=task,
        status=TaskStatus.PENDING.value,
        metadata_json=metadata or {},
    )
    session.add(db_task)
    await session.commit()
    await session.refresh(db_task)
    return db_task


async def get_task(session: AsyncSession, task_id: str) -> Task | None:
    """Retrieves a single task by ID."""
    stmt = select(Task).where(Task.id == task_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_tasks(
    session: AsyncSession,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Task]:
    """Retrieves a paginated list of tasks, optionally filtered by status,

    ordered by created_at DESC.
    """
    stmt = select(Task).order_by(desc(Task.created_at)).limit(limit).offset(offset)
    if status is not None:
        stmt = stmt.where(Task.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def mark_running(session: AsyncSession, task_id: str) -> None:
    """Transitions a task to RUNNING status."""
    now = datetime.now(UTC)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=TaskStatus.RUNNING.value,
            started_at=now,
            updated_at=now,
        )
    )
    await session.execute(stmt)
    await session.commit()


async def mark_completed(
    session: AsyncSession,
    task_id: str,
    result: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    total_cost_usd: float = 0.0,
    duration_seconds: float = 0.0,
) -> None:
    """Transitions a task to COMPLETED status with final metrics and answer."""
    now = datetime.now(UTC)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=TaskStatus.COMPLETED.value,
            result=result,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            duration_seconds=duration_seconds,
            completed_at=now,
            updated_at=now,
        )
    )
    await session.execute(stmt)
    await session.commit()


async def mark_failed(session: AsyncSession, task_id: str, error: str) -> None:
    """Transitions a task to FAILED status with an error message."""
    now = datetime.now(UTC)
    stmt = (
        update(Task)
        .where(Task.id == task_id)
        .values(
            status=TaskStatus.FAILED.value,
            error=error,
            completed_at=now,
            updated_at=now,
        )
    )
    await session.execute(stmt)
    await session.commit()


async def recover_interrupted(session: AsyncSession) -> int:
    """Recovers any tasks left in RUNNING status during a crash or abrupt shutdown.

    Marks them as FAILED and returns the count of recovered tasks.
    """
    now = datetime.now(UTC)
    stmt = (
        update(Task)
        .where(Task.status == TaskStatus.RUNNING.value)
        .values(
            status=TaskStatus.FAILED.value,
            error="Task was interrupted by server shutdown or crash",
            completed_at=now,
            updated_at=now,
        )
    )
    result = await session.execute(stmt)
    await session.commit()
    return int(getattr(result, "rowcount", 0))
