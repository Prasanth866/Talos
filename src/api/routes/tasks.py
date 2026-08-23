from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from src.api.schemas.events import (
    TaskDetailResponse,
    TaskResponse,
    TaskStatus,
    TaskSubmitRequest,
)
from src.core.logging import get_logger
from src.core.worker import TaskManager
from src.db import repository
from src.db.models import Task

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _to_task_detail_response(task: Task) -> TaskDetailResponse:
    return TaskDetailResponse(
        task_id=task.id,
        task=task.task,
        status=TaskStatus(task.status),
        result=task.result,
        error=task.error,
        metadata=task.metadata_json,
        prompt_tokens=task.prompt_tokens,
        completion_tokens=task.completion_tokens,
        total_tokens=task.total_tokens,
        total_cost_usd=task.total_cost_usd,
        duration_seconds=task.duration_seconds,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


@router.post("", response_model=TaskResponse, status_code=202)
async def submit_task(
    request_body: TaskSubmitRequest,
    request: Request,
) -> Response:
    """Submits a new agent task for execution.

    Persists the task to the database in PENDING status, enqueues it to the worker pool,
    and returns the task ID with WebSocket streaming endpoint URL.
    Returns HTTP 503 when the worker queue is full or during shutdown.
    """
    task_id = str(uuid4())
    task_manager: TaskManager | None = getattr(request.app.state, "task_manager", None)

    if task_manager is not None:
        if task_manager.is_shutting_down:
            logger.warning(
                "task_submission_rejected_shutting_down",
                task_id=task_id,
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is shutting down"},
                headers={"Retry-After": "5"},
            )

        if task_manager.queue.full():
            logger.warning("task_submission_rejected_queue_full", task_id=task_id)
            return JSONResponse(
                status_code=503,
                content={"detail": "Task queue is full"},
                headers={"Retry-After": "5"},
            )

        if task_manager.session_factory is not None:
            async with task_manager.session_factory() as session:
                await repository.create_task(
                    session=session,
                    task_id=task_id,
                    task=request_body.task,
                    metadata=request_body.metadata,
                )

        success = task_manager.submit(
            task_id=task_id,
            task=request_body.task,
            metadata=request_body.metadata,
        )
        if not success:
            logger.warning("task_submission_rejected_queue_full", task_id=task_id)
            if task_manager.session_factory is not None:
                async with task_manager.session_factory() as session:
                    await repository.mark_failed(
                        session=session,
                        task_id=task_id,
                        error="Task rejected: worker queue full",
                    )
            return JSONResponse(
                status_code=503,
                content={"detail": "Task queue is full"},
                headers={"Retry-After": "5"},
            )

    logger.info("task_submitted", task_id=task_id, task=request_body.task)

    return JSONResponse(
        status_code=202,
        content=TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            ws_url=f"/ws?task_id={task_id}",
        ).model_dump(mode="json"),
    )


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: str,
    request: Request,
) -> TaskDetailResponse:
    """Fetches full task details by task ID."""
    task_manager: TaskManager | None = getattr(request.app.state, "task_manager", None)
    if task_manager is None or task_manager.session_factory is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    async with task_manager.session_factory() as session:
        task = await repository.get_task(session, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return _to_task_detail_response(task)


@router.get("", response_model=list[TaskDetailResponse])
async def list_tasks(
    request: Request,
    status: Annotated[
        TaskStatus | None,
        Query(description="Filter by status"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Max items to return"),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Items to offset"),
    ] = 0,
) -> list[TaskDetailResponse]:
    """Queries a paginated list of tasks, optionally filtered by status."""
    task_manager: TaskManager | None = getattr(request.app.state, "task_manager", None)
    if task_manager is None or task_manager.session_factory is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    async with task_manager.session_factory() as session:
        tasks = await repository.list_tasks(
            session=session,
            status=status.value if status is not None else None,
            limit=limit,
            offset=offset,
        )
        return [_to_task_detail_response(t) for t in tasks]


@router.delete("/{task_id}", status_code=200)
async def delete_task(
    task_id: str,
    request: Request,
) -> dict[str, Any]:
    """Deletes a single task by ID from the database and prunes in-memory state."""
    task_manager: TaskManager | None = getattr(request.app.state, "task_manager", None)
    if task_manager is None or task_manager.session_factory is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    async with task_manager.session_factory() as session:
        deleted = await repository.delete_task(session, task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    await task_manager.prune_task(task_id)
    return {"status": "deleted", "task_id": task_id}


@router.delete("", status_code=200)
@router.delete("/", status_code=200)
async def clear_all_tasks(
    request: Request,
) -> dict[str, Any]:
    """Clears all tasks from the database."""
    task_manager: TaskManager | None = getattr(request.app.state, "task_manager", None)
    if task_manager is None or task_manager.session_factory is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")

    async with task_manager.session_factory() as session:
        count = await repository.clear_tasks(session)

    return {"status": "cleared", "deleted_count": count}
