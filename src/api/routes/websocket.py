from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import BaseLLMClient, HTTPLLMClient, MockLLMClient
from src.agent.loop import ReasoningLoop
from src.api.schemas.events import AgentEvent, ErrorEvent
from src.core.config import ROOT_DIR, get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])

LoopFactory = Callable[[], ReasoningLoop]


def get_default_loop_factory(
    llm_client_override: BaseLLMClient | None = None,
) -> ReasoningLoop:
    """Creates a ReasoningLoop instance using app configuration."""
    settings = get_settings()
    dispatcher = create_default_dispatcher(ROOT_DIR)

    if llm_client_override is not None:
        llm_client: BaseLLMClient = llm_client_override
    elif settings.llm_api_key.get_secret_value():
        llm_client = HTTPLLMClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            initial_delay=settings.llm_retry_initial_delay,
            backoff_factor=settings.llm_retry_backoff_factor,
        )
    else:
        # Default mock client for development/testing when no key is set
        llm_client = MockLLMClient(
            responses=[
                {
                    "thought": "Processing received task in development mode.",
                    "final_answer": "Task executed successfully (mock mode).",
                }
            ]
        )

    return ReasoningLoop(
        llm_client=llm_client,
        dispatcher=dispatcher,
        max_steps=settings.llm_max_steps,
    )


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    task_id: str | None = None,
) -> None:
    """Bidirectional WebSocket endpoint for streaming reasoning loop events."""
    await websocket.accept()
    logger.info("websocket_connected", client=str(websocket.client), task_id=task_id)

    async def emit_to_ws(event: AgentEvent) -> None:
        await websocket.send_text(event.model_dump_json())

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data: dict[str, Any] = json.loads(raw_data)
            except json.JSONDecodeError:
                data = {"task": raw_data}

            task = data.get("task")
            if not task or not isinstance(task, str):
                await emit_to_ws(
                    ErrorEvent(
                        error="Invalid message format: 'task' string field is required."
                    )
                )
                continue

            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Check if app state has custom loop factory or client override
            loop_factory: LoopFactory = getattr(
                websocket.app.state,
                "reasoning_loop_factory",
                get_default_loop_factory,
            )
            loop = loop_factory()

            logger.info(
                "websocket_running_task",
                task=task,
                client=str(websocket.client),
            )
            await loop.run(task=task, metadata=metadata, on_event=emit_to_ws)

    except WebSocketDisconnect:
        logger.info(
            "websocket_disconnected",
            client=str(websocket.client),
            task_id=task_id,
        )
    except Exception as exc:
        logger.exception("websocket_error", error=str(exc))
        with contextlib.suppress(Exception):
            await emit_to_ws(ErrorEvent(error=f"Internal server error: {exc}"))
