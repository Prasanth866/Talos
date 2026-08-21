from __future__ import annotations

import contextlib
from collections.abc import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.llm_client import BaseLLMClient
from src.agent.loop import ReasoningLoop
from src.api.schemas.events import ErrorEvent
from src.core.logging import get_logger
from src.core.worker import TaskManager, create_default_loop

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])

LoopFactory = Callable[[], ReasoningLoop]


def get_default_loop_factory(
    llm_client_override: BaseLLMClient | None = None,
) -> ReasoningLoop:
    """Creates a ReasoningLoop instance using app configuration."""
    return create_default_loop(llm_client_override=llm_client_override)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    task_id: str | None = None,
) -> None:
    """WebSocket subscriber endpoint for streaming reasoning loop events.

    Listens for events matching the given task_id.
    """
    await websocket.accept()
    logger.info("websocket_connected", client=str(websocket.client), task_id=task_id)

    if not task_id:
        await websocket.send_text(
            ErrorEvent(error="task_id query parameter is required.").model_dump_json()
        )
        await websocket.close()
        return

    task_manager: TaskManager | None = getattr(
        websocket.app.state, "task_manager", None
    )
    if task_manager is None:
        await websocket.send_text(
            ErrorEvent(
                error="TaskManager is not initialized.",
                task_id=task_id,
            ).model_dump_json()
        )
        await websocket.close()
        return

    subscriber_queue = await task_manager.subscribe(task_id)

    try:
        while True:
            event = await subscriber_queue.get()
            if event is None:
                # Task processing finished
                logger.debug(
                    "websocket_task_stream_ended",
                    task_id=task_id,
                    client=str(websocket.client),
                )
                break
            await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        logger.info(
            "websocket_disconnected",
            client=str(websocket.client),
            task_id=task_id,
        )
    except Exception as exc:
        logger.exception("websocket_error", error=str(exc), task_id=task_id)
        with contextlib.suppress(Exception):
            await websocket.send_text(
                ErrorEvent(
                    error=f"Internal server error: {exc}",
                    task_id=task_id,
                ).model_dump_json()
            )
    finally:
        await task_manager.unsubscribe(task_id, subscriber_queue)
        with contextlib.suppress(Exception):
            await websocket.close()
