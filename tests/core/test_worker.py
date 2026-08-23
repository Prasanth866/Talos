from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
import structlog

from src.agent.dispatcher import ToolDispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.api.schemas.events import AgentEvent, TaskCompleteEvent, ThoughtEvent
from src.core.config import Settings
from src.core.worker import TaskItem, TaskManager


@pytest.mark.asyncio
async def test_task_item_creation() -> None:
    """Verifies TaskItem data structure initialization."""
    task_id = str(uuid4())
    item = TaskItem(task_id=task_id, task="Analyze repo", metadata={"env": "test"})
    assert item.task_id == task_id
    assert item.task == "Analyze repo"
    assert item.metadata == {"env": "test"}


@pytest.mark.asyncio
async def test_submit_returns_true_when_capacity_available() -> None:
    """Verifies submit returns True when queue is below capacity."""
    settings = Settings(task_queue_max_size=5, worker_concurrency=1)
    tm = TaskManager(settings=settings)
    task_id = str(uuid4())

    success = tm.submit(task_id=task_id, task="Test task")
    assert success is True
    assert tm.queue_size == 1
    assert tm.queue_capacity == 5


@pytest.mark.asyncio
async def test_submit_returns_false_when_queue_full() -> None:
    """Verifies submit returns False (backpressure) when queue reaches capacity."""
    settings = Settings(task_queue_max_size=2, worker_concurrency=1)
    tm = TaskManager(settings=settings)

    assert tm.submit(str(uuid4()), "Task 1") is True
    assert tm.submit(str(uuid4()), "Task 2") is True
    # Queue is now at capacity (2/2)
    assert tm.submit(str(uuid4()), "Task 3") is False


@pytest.mark.asyncio
async def test_submit_returns_false_when_shutting_down() -> None:
    """Verifies submit rejects new tasks when manager is shutting down."""
    settings = Settings(task_queue_max_size=10, worker_concurrency=1)
    tm = TaskManager(settings=settings)
    tm._shutting_down = True

    assert tm.submit(str(uuid4()), "Task after shutdown") is False


@pytest.mark.asyncio
async def test_worker_processes_task_and_broadcasts_events() -> None:
    """Verifies workers pull tasks from queue, execute loop, and broadcast events."""
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Thinking about the task",
                "final_answer": "Task is completed successfully.",
            }
        ]
    )
    dispatcher = ToolDispatcher()

    def custom_loop_factory() -> ReasoningLoop:
        return ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)

    settings = Settings(task_queue_max_size=5, worker_concurrency=1)
    tm = TaskManager(settings=settings, loop_factory=custom_loop_factory)
    task_id = str(uuid4())

    sub_queue = await tm.subscribe(task_id)

    async with asyncio.TaskGroup() as tg:
        tm.start(tg)
        assert tm.submit(task_id=task_id, task="Execute workflow") is True

        received_events: list[AgentEvent | None] = []
        while True:
            evt = await sub_queue.get()
            received_events.append(evt)
            if evt is None:  # Sentinel indicating task completed
                break

        await tm.stop()

    # Filter non-None events
    events = [e for e in received_events if e is not None]
    assert len(events) >= 2
    assert isinstance(events[0], ThoughtEvent)
    assert events[0].task_id == task_id
    assert events[0].thought == "Thinking about the task"

    assert isinstance(events[1], TaskCompleteEvent)
    assert events[1].task_id == task_id
    assert events[1].final_answer == "Task is completed successfully."


@pytest.mark.asyncio
async def test_worker_binds_task_id_to_structlog_contextvars() -> None:
    """Verifies that task_id is bound in structlog contextvars during task execution."""
    captured_contextvars: list[dict[str, Any]] = []

    class CapturingLLM(MockLLMClient):
        async def generate_response(
            self,
            messages: list[Any],
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            captured_contextvars.append(dict(structlog.contextvars.get_contextvars()))
            return await super().generate_response(messages, tools=tools)

    mock_llm = CapturingLLM(
        responses=[
            {
                "thought": "Logged thought",
                "final_answer": "Done",
            }
        ]
    )
    dispatcher = ToolDispatcher()

    def custom_loop_factory() -> ReasoningLoop:
        return ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)

    settings = Settings(task_queue_max_size=5, worker_concurrency=1)
    tm = TaskManager(settings=settings, loop_factory=custom_loop_factory)
    task_id = str(uuid4())

    async with asyncio.TaskGroup() as tg:
        tm.start(tg)
        tm.submit(task_id=task_id, task="ContextVar task")
        await tm.stop()

    assert len(captured_contextvars) > 0
    assert captured_contextvars[0].get("task_id") == task_id


@pytest.mark.asyncio
async def test_graceful_shutdown_drains_inflight_tasks() -> None:
    """Integration test: verifies graceful shutdown drains all in-flight tasks."""
    completed_tasks: list[str] = []

    class SlowLLM(MockLLMClient):
        async def generate_response(
            self,
            messages: list[Any],
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            await asyncio.sleep(0.05)  # Simulate in-flight work
            cv = structlog.contextvars.get_contextvars()
            completed_tasks.append(cv.get("task_id", ""))
            return await super().generate_response(messages, tools=tools)

    mock_llm = SlowLLM(
        responses=[
            {"thought": "Working...", "final_answer": "Done 1"},
            {"thought": "Working...", "final_answer": "Done 2"},
            {"thought": "Working...", "final_answer": "Done 3"},
        ]
    )
    dispatcher = ToolDispatcher()

    def custom_loop_factory() -> ReasoningLoop:
        return ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)

    settings = Settings(
        task_queue_max_size=10,
        worker_concurrency=2,
        shutdown_drain_timeout_seconds=5.0,
    )
    tm = TaskManager(settings=settings, loop_factory=custom_loop_factory)

    t1, t2, t3 = str(uuid4()), str(uuid4()), str(uuid4())
    tm.submit(t1, "Task 1")
    tm.submit(t2, "Task 2")
    tm.submit(t3, "Task 3")

    async with asyncio.TaskGroup() as tg:
        tm.start(tg)
        await tm.stop()

    assert set(completed_tasks) == {t1, t2, t3}


@pytest.mark.asyncio
async def test_subscribe_unsubscribe_lifecycle() -> None:
    """Verifies subscribe and unsubscribe correctly update manager subscriber map."""
    tm = TaskManager()
    task_id = str(uuid4())

    q1 = await tm.subscribe(task_id)
    assert task_id in tm._subscribers
    assert len(tm._subscribers[task_id]) == 1

    q2 = await tm.subscribe(task_id)
    assert len(tm._subscribers[task_id]) == 2

    await tm.unsubscribe(task_id, q1)
    assert len(tm._subscribers[task_id]) == 1

    await tm.unsubscribe(task_id, q2)
    assert task_id not in tm._subscribers


@pytest.mark.asyncio
async def test_task_events_bounded_retention() -> None:
    """Verifies that TaskManager evicts oldest completed task events
    when limit is exceeded.
    """
    tm = TaskManager(max_retained_task_events=2)

    t1, t2, t3 = str(uuid4()), str(uuid4()), str(uuid4())

    evt1 = ThoughtEvent(thought="Thought 1", step=1, task_id=t1)
    evt2 = ThoughtEvent(thought="Thought 2", step=1, task_id=t2)
    evt3 = ThoughtEvent(thought="Thought 3", step=1, task_id=t3)

    # Broadcast events for t1 and complete t1
    await tm.broadcast_event(t1, evt1)
    await tm.broadcast_event(t1, None)

    # Broadcast events for t2 and complete t2
    await tm.broadcast_event(t2, evt2)
    await tm.broadcast_event(t2, None)

    assert t1 in tm._task_events
    assert t2 in tm._task_events

    # Broadcast events for t3 and complete t3 (should evict t1 since limit is 2)
    await tm.broadcast_event(t3, evt3)
    await tm.broadcast_event(t3, None)

    assert t1 not in tm._task_events
    assert t1 not in tm._task_completed
    assert t2 in tm._task_events
    assert t3 in tm._task_events


@pytest.mark.asyncio
async def test_prune_task_removes_in_memory_state() -> None:
    """Verifies that prune_task explicitly removes events and completion status."""
    tm = TaskManager()
    task_id = str(uuid4())
    evt = ThoughtEvent(thought="Thought", step=1, task_id=task_id)

    await tm.broadcast_event(task_id, evt)
    await tm.broadcast_event(task_id, None)

    assert task_id in tm._task_events
    assert task_id in tm._task_completed

    await tm.prune_task(task_id)

    assert task_id not in tm._task_events
    assert task_id not in tm._task_completed
