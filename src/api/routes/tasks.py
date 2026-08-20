from uuid import uuid4

from fastapi import APIRouter

from src.api.schemas.events import TaskResponse, TaskSubmitRequest
from src.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse, status_code=202)
async def submit_task(request: TaskSubmitRequest) -> TaskResponse:
    """Submits a new agent task for execution.

    Returns the task ID and WebSocket streaming endpoint URL.
    """
    task_id = f"task_{uuid4().hex[:12]}"
    logger.info("task_submitted", task_id=task_id, task=request.task)

    return TaskResponse(
        task_id=task_id,
        status="queued",
        ws_url=f"/ws?task_id={task_id}",
    )
