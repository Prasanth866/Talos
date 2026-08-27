from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Task, TaskStatus


async def create_task(
    session: AsyncSession,
    task_id: str,
    task: str,
    metadata: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
) -> Task:
    """Inserts a new task in PENDING state."""
    db_task = Task(
        id=task_id,
        task=task,
        status=TaskStatus.PENDING.value,
        metadata_json=metadata or {},
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
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


async def mark_failed(
    session: AsyncSession,
    task_id: str,
    error: str,
    result: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    total_cost_usd: float = 0.0,
    duration_seconds: float = 0.0,
) -> None:
    """Transitions a task to FAILED status with error and optional partial result."""
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "status": TaskStatus.FAILED.value,
        "error": error,
        "completed_at": now,
        "updated_at": now,
    }
    if result is not None:
        values["result"] = result
    if prompt_tokens > 0:
        values["prompt_tokens"] = prompt_tokens
    if completion_tokens > 0:
        values["completion_tokens"] = completion_tokens
    if total_tokens > 0:
        values["total_tokens"] = total_tokens
    if total_cost_usd > 0.0:
        values["total_cost_usd"] = total_cost_usd
    if duration_seconds > 0.0:
        values["duration_seconds"] = duration_seconds

    stmt = update(Task).where(Task.id == task_id).values(**values)
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


async def delete_task(session: AsyncSession, task_id: str) -> bool:
    """Deletes a single task from the database by ID."""
    stmt = delete(Task).where(Task.id == task_id)
    result = await session.execute(stmt)
    await session.commit()
    return int(getattr(result, "rowcount", 0)) > 0


async def clear_tasks(session: AsyncSession) -> int:
    """Deletes all tasks from the database."""
    stmt = delete(Task)
    result = await session.execute(stmt)
    await session.commit()
    return int(getattr(result, "rowcount", 0))
