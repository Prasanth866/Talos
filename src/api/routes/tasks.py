from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from src.api.schemas.events import TaskResponse, TaskSubmitRequest
from src.core.logging import get_logger
from src.core.worker import TaskManager

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=202)
async def submit_task(
    request_body: TaskSubmitRequest,
    request: Request,
) -> Response:
    """Submits a new agent task for execution.

    Returns the task ID and WebSocket streaming endpoint URL.
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

        success = task_manager.submit(
            task_id=task_id,
            task=request_body.task,
            metadata=request_body.metadata,
        )
        if not success:
            logger.warning("task_submission_rejected_queue_full", task_id=task_id)
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
            status="queued",
            ws_url=f"/ws?task_id={task_id}",
        ).model_dump(mode="json"),
    )
