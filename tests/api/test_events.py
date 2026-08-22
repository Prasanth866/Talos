import pytest
from pydantic import TypeAdapter, ValidationError

from src.api.schemas.events import (
    AgentEvent,
    ErrorEvent,
    EventType,
    HealthResponse,
    ReadinessResponse,
    TaskCompleteEvent,
    TaskDetailResponse,
    TaskResponse,
    TaskStatus,
    TaskSubmitRequest,
    ThoughtEvent,
    ToolCallEvent,
    ToolOutputEvent,
)


def test_thought_event_schema() -> None:
    """Verifies ThoughtEvent validates fields, default version, and task_id."""
    event = ThoughtEvent(thought="Analyzing task", step=1, task_id="test-task-123")
    assert event.event_type == EventType.THOUGHT
    assert event.version == "v1"
    assert event.thought == "Analyzing task"
    assert event.step == 1
    assert event.task_id == "test-task-123"
    assert event.timestamp is not None

    dumped = event.model_dump(mode="json")
    assert dumped["event_type"] == "thought"
    assert dumped["version"] == "v1"
    assert dumped["task_id"] == "test-task-123"


def test_tool_call_event_schema() -> None:
    """Verifies ToolCallEvent validates required fields."""
    event = ToolCallEvent(
        tool_name="read_file",
        tool_call_id="call_123",
        arguments={"path": "main.py"},
        step=2,
    )
    assert event.event_type == EventType.TOOL_CALL
    assert event.tool_name == "read_file"
    assert event.tool_call_id == "call_123"
    assert event.arguments == {"path": "main.py"}
    assert event.step == 2


def test_tool_output_event_schema() -> None:
    """Verifies ToolOutputEvent validates output and success fields."""
    event = ToolOutputEvent(
        tool_name="read_file",
        tool_call_id="call_123",
        output="file content",
        success=True,
        duration_seconds=0.05,
        step=2,
    )
    assert event.event_type == EventType.TOOL_OUTPUT
    assert event.success is True
    assert event.duration_seconds == 0.05
    assert event.step == 2


def test_task_complete_event_schema() -> None:
    """Verifies TaskCompleteEvent validates completion metrics."""
    event = TaskCompleteEvent(
        task="Build server",
        final_answer="Server built successfully",
        total_steps=3,
        total_tokens=150,
        total_cost_usd=0.002,
        duration_seconds=1.25,
    )
    assert event.event_type == EventType.TASK_COMPLETE
    assert event.final_answer == "Server built successfully"
    assert event.total_steps == 3
    assert event.total_tokens == 150
    assert event.total_cost_usd == 0.002


def test_error_event_schema() -> None:
    """Verifies ErrorEvent schema and optional step/details."""
    event = ErrorEvent(
        error="LLM failed",
        details={"reason": "rate_limit"},
        step=3,
    )
    assert event.event_type == EventType.ERROR
    assert event.error == "LLM failed"
    assert event.details == {"reason": "rate_limit"}
    assert event.step == 3


def test_discriminated_union_adapter() -> None:
    """Verifies AgentEvent discriminated union parses different event types."""
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)

    thought_raw = {
        "event_type": "thought",
        "thought": "Thinking...",
        "step": 1,
    }
    parsed_thought = adapter.validate_python(thought_raw)
    assert isinstance(parsed_thought, ThoughtEvent)
    assert parsed_thought.thought == "Thinking..."

    tool_call_raw = {
        "event_type": "tool_call",
        "tool_name": "list_dir",
        "tool_call_id": "c1",
        "arguments": {},
        "step": 2,
    }
    parsed_call = adapter.validate_python(tool_call_raw)
    assert isinstance(parsed_call, ToolCallEvent)
    assert parsed_call.tool_name == "list_dir"

    complete_raw = {
        "event_type": "task_complete",
        "task": "Test task",
        "final_answer": "Done",
        "total_steps": 2,
        "total_tokens": 100,
        "total_cost_usd": 0.001,
        "duration_seconds": 0.5,
    }
    parsed_complete = adapter.validate_python(complete_raw)
    assert isinstance(parsed_complete, TaskCompleteEvent)


def test_event_validation_errors() -> None:
    """Verifies validation errors when required fields are missing."""
    with pytest.raises(ValidationError):
        ThoughtEvent.model_validate({"thought": "missing step"})

    with pytest.raises(ValidationError):
        ToolCallEvent.model_validate({"tool_name": "tool"})

    with pytest.raises(ValidationError):
        TaskSubmitRequest.model_validate({"task": ""})


def test_http_response_schemas() -> None:
    """Verifies HealthResponse, ReadinessResponse, and TaskResponse models."""
    health = HealthResponse()
    assert health.status == "ok"

    readiness = ReadinessResponse(services={"db": "ok"})
    assert readiness.status == "ready"
    assert readiness.services == {"db": "ok"}

    task_resp = TaskResponse(task_id="task_1", ws_url="/ws?task_id=task_1")
    assert task_resp.task_id == "task_1"
    assert task_resp.status == TaskStatus.PENDING
    assert task_resp.ws_url == "/ws?task_id=task_1"

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    detail_resp = TaskDetailResponse(
        task_id="task_1",
        task="Test task",
        status=TaskStatus.COMPLETED,
        result="Success",
        created_at=now,
        updated_at=now,
    )
    assert detail_resp.task_id == "task_1"
    assert detail_resp.status == TaskStatus.COMPLETED
    assert detail_resp.result == "Success"
    assert detail_resp.total_tokens == 0
