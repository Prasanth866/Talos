from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.agent.dispatcher import ToolDispatcher
from src.agent.graph import LangGraphAgent
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import (
    AgentStatus,
    CostRates,
    LLMResponse,
    TokenUsage,
    ToolCall,
    TrajectoryStatus,
)
from src.agent.token_tracker import TokenTracker
from src.api.routes.tasks import _to_task_detail_response
from src.api.schemas.events import (
    AgentEvent,
    BudgetExceededEvent,
    EventType,
)
from src.core.config import Settings
from src.core.worker import TaskManager
from src.db import repository
from src.db.models import Base, Task, TaskStatus


@pytest.mark.asyncio
async def test_budget_exceeded_triggers_budget_exceeded_event() -> None:
    """Unit test: Budget exceeded triggers BudgetExceededEvent on event stream."""
    dispatcher = ToolDispatcher()
    dispatcher.register_tool(
        name="do_work",
        description="Works",
        handler=lambda: "done",
    )

    token_tracker = TokenTracker(
        max_tokens=100,
        custom_rates=CostRates(prompt_cost_per_1m=1.0, completion_cost_per_1m=2.0),
    )

    # First response uses 120 tokens -> Exceeds 100 max_tokens
    resp1 = LLMResponse(
        thought="Starting work",
        tool_call=ToolCall(tool_name="do_work", arguments={}),
        token_usage=TokenUsage(
            prompt_tokens=80, completion_tokens=40, total_tokens=120
        ),
    )
    resp2 = LLMResponse(
        thought="Step 2 should never be called",
        final_answer="Done",
        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
    )

    llm = MockLLMClient(responses=[resp1, resp2], token_tracker=token_tracker)
    loop = ReasoningLoop(llm_client=llm, dispatcher=dispatcher, max_steps=5)

    events: list[AgentEvent] = []

    async def on_event(evt: AgentEvent) -> None:
        events.append(evt)

    trajectory = await loop.run(
        task="Heavy calculation task",
        on_event=on_event,
        task_id="task-budget-1",
    )

    # Trajectory must fail due to budget exceeded
    assert trajectory.status == TrajectoryStatus.FAILED
    assert "budget_exceeded" in (trajectory.error or "")
    assert "PARTIAL RESULT" in (trajectory.final_answer or "")

    # BudgetExceededEvent must have been emitted
    budget_events = [e for e in events if e.event_type == EventType.BUDGET_EXCEEDED]
    assert len(budget_events) == 1
    bev = budget_events[0]
    assert isinstance(bev, BudgetExceededEvent)
    assert bev.budget_type == "tokens"
    assert bev.tokens_used >= 120
    assert bev.max_tokens == 100
    assert bev.partial_result is not None


@pytest.mark.asyncio
async def test_partial_result_written_to_db_on_budget_exhaustion(
    tmp_path: Path,
) -> None:
    """Unit test: Partial result is written to DB on budget exhaustion."""
    db_file = tmp_path / "budget_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    task_id = str(uuid4())
    async with session_factory() as session:
        await repository.create_task(
            session=session,
            task_id=task_id,
            task="Perform multi-step refactoring",
            max_tokens=50,
        )

    dispatcher = ToolDispatcher()
    dispatcher.register_tool(
        name="step1_tool",
        description="Step 1",
        handler=lambda: "completed step 1",
    )

    # Step 1 consumes 80 tokens (exceeding 50 limit)
    tracker = TokenTracker(max_tokens=50)
    resp1 = LLMResponse(
        thought="Executing step 1",
        tool_call=ToolCall(tool_name="step1_tool", arguments={}),
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    )
    resp2 = LLMResponse(
        thought="Step 2",
        final_answer="Finished",
    )

    llm = MockLLMClient(responses=[resp1, resp2], token_tracker=tracker)

    def custom_loop_factory() -> ReasoningLoop:
        return ReasoningLoop(llm_client=llm, dispatcher=dispatcher)

    settings = Settings(task_queue_max_size=5, worker_concurrency=1)
    task_manager = TaskManager(
        settings=settings,
        loop_factory=custom_loop_factory,
        session_factory=session_factory,
    )

    async with asyncio.TaskGroup() as tg:
        task_manager.start(tg)
        task_manager.submit(
            task_id=task_id,
            task="Perform multi-step refactoring",
            max_tokens=50,
        )
        await task_manager.stop()

    # Verify DB persistence
    async with session_factory() as session:
        saved_task = await repository.get_task(session, task_id)
        assert saved_task is not None
        assert saved_task.status == TaskStatus.FAILED.value
        assert "budget_exceeded" in (saved_task.error or "")
        assert saved_task.result is not None
        assert "PARTIAL RESULT" in saved_task.result
        assert saved_task.total_tokens >= 80

    await engine.dispose()


@pytest.mark.asyncio
async def test_pre_call_budget_check_prevents_initial_llm_call() -> None:
    """Unit test: When budget is already exhausted, 0 LLM calls are executed."""
    dispatcher = ToolDispatcher()
    token_tracker = TokenTracker(max_tokens=100)
    # Pre-record 100 tokens consumed prior to call
    token_tracker.record_usage(prompt_tokens=60, completion_tokens=40)

    mock_llm = MockLLMClient(
        responses=[LLMResponse(thought="Should not run", final_answer="Done")],
        token_tracker=token_tracker,
    )
    loop = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)

    events: list[AgentEvent] = []

    async def on_event(evt: AgentEvent) -> None:
        events.append(evt)

    trajectory = await loop.run(
        task="Task with zero remaining budget",
        on_event=on_event,
        task_id="zero-budget-task",
    )

    assert trajectory.status == TrajectoryStatus.FAILED
    assert "budget_exceeded" in (trajectory.error or "")
    assert len(mock_llm.call_history) == 0
    assert len(events) == 1
    assert isinstance(events[0], BudgetExceededEvent)


def test_task_status_api_shows_tokens_cost_and_budget_remaining_pct() -> None:
    """Unit test: Task status API serializer calculates budget_remaining_pct."""
    task = Task(
        id="test-api-task",
        task="Analyze performance",
        status=TaskStatus.RUNNING.value,
        total_tokens=400,
        total_cost_usd=0.002,
        max_tokens=1000,
        max_cost_usd=0.01,
    )
    detail = _to_task_detail_response(task)

    assert detail.tokens_used == 400
    assert detail.cost_usd == 0.002
    assert detail.max_tokens == 1000
    assert detail.max_cost_usd == 0.01
    # 400/1000 tokens (60% rem); 0.002/0.01 cost (80% rem) -> min = 60.0%
    assert detail.budget_remaining_pct == 60.0

    # Test exhausted budget (0.0% remaining)
    task_exhausted = Task(
        id="exhausted-task",
        task="Analyze performance",
        status=TaskStatus.FAILED.value,
        total_tokens=1500,
        total_cost_usd=0.015,
        max_tokens=1000,
        max_cost_usd=0.01,
    )
    exhausted_detail = _to_task_detail_response(task_exhausted)
    assert exhausted_detail.budget_remaining_pct == 0.0


def test_task_transitions_to_failed_with_reason_budget_exceeded_in_graph() -> None:
    """Unit test: LangGraph state machine hard-stops on budget exhaustion."""
    dispatcher = ToolDispatcher()
    dispatcher.register_tool(
        name="dummy_tool",
        description="Dummy",
        handler=lambda: "ok",
    )

    plan_json = json.dumps(
        {
            "task": "Long budget task",
            "rationale": "Plan steps",
            "steps": [
                {"step_id": 1, "description": "Step 1", "expected_output": "out"}
            ],
        }
    )

    # Tracker with max_tokens=100
    tracker = TokenTracker(max_tokens=100)
    mock_responses = [
        LLMResponse(
            raw_content=plan_json,
            thought="Plan created",
            token_usage=TokenUsage(
                prompt_tokens=90, completion_tokens=30, total_tokens=120
            ),
        ),
        LLMResponse(
            thought="Execution should be blocked by pre-call check",
            tool_call=ToolCall(tool_name="dummy_tool", arguments={}),
        ),
    ]
    llm = MockLLMClient(responses=mock_responses, token_tracker=tracker)
    agent = LangGraphAgent(llm_client=llm, dispatcher=dispatcher)

    final_state = agent.run_task(
        task_id="graph-budget-task",
        task="Long budget task",
        max_tokens=100,
    )

    assert final_state["status"] == AgentStatus.FAILED.value
    assert "budget_exceeded" in (final_state["error"] or "")
    assert "PARTIAL RESULT" in (final_state["partial_result"] or "")
