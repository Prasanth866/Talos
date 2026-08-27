from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from src.db.models import TaskStatus as TaskStatus


class EventType(StrEnum):
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    BUDGET_EXCEEDED = "budget_exceeded"
    SECURITY_ALERT = "security_alert"


class BaseEvent(BaseModel):
    """Base model for all versioned agent stream events."""

    version: str = "v1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    event_type: EventType


class ThoughtEvent(BaseEvent):
    """Emitted when the agent generates a reasoning thought."""

    event_type: Literal[EventType.THOUGHT] = EventType.THOUGHT
    thought: str
    step: int


class ToolCallEvent(BaseEvent):
    """Emitted when the agent initiates a tool execution."""

    event_type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    step: int


class ToolOutputEvent(BaseEvent):
    """Emitted when a tool execution completes."""

    event_type: Literal[EventType.TOOL_OUTPUT] = EventType.TOOL_OUTPUT
    tool_name: str
    tool_call_id: str
    output: str
    success: bool
    duration_seconds: float
    step: int


class TaskCompleteEvent(BaseEvent):
    """Emitted when the task execution finishes successfully."""

    event_type: Literal[EventType.TASK_COMPLETE] = EventType.TASK_COMPLETE
    task: str
    final_answer: str
    total_steps: int
    total_tokens: int
    total_cost_usd: float
    duration_seconds: float


class ErrorEvent(BaseEvent):
    """Emitted when an error occurs during execution."""

    event_type: Literal[EventType.ERROR] = EventType.ERROR
    error: str
    details: dict[str, Any] | None = None
    step: int | None = None


class BudgetExceededEvent(BaseEvent):
    """Emitted when a task exceeds its configured token or cost budget."""

    event_type: Literal[EventType.BUDGET_EXCEEDED] = EventType.BUDGET_EXCEEDED
    tokens_used: int
    cost_usd: float
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    budget_type: str
    partial_result: str | None = None
    step: int | None = None


class SecurityEvent(BaseEvent):
    """Emitted when a security threat (e.g. secret leak, dangerous cmd) is detected."""

    event_type: Literal[EventType.SECURITY_ALERT] = EventType.SECURITY_ALERT
    threat_type: str
    threat_details: dict[str, Any] = Field(default_factory=dict)
    redacted_count: int = 0
    step: int | None = None


AgentEvent = Annotated[
    ThoughtEvent
    | ToolCallEvent
    | ToolOutputEvent
    | TaskCompleteEvent
    | ErrorEvent
    | BudgetExceededEvent
    | SecurityEvent,
    Field(discriminator="event_type"),
]


class TaskSubmitRequest(BaseModel):
    """Request payload to submit a new agent task."""

    task: str = Field(..., min_length=1, description="Task prompt to execute")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional task metadata"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Max total token budget"
    )
    max_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Max cost budget in USD"
    )


class TaskResponse(BaseModel):
    """Response returned upon submitting a task."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    ws_url: str


class TaskDetailResponse(BaseModel):
    """Detailed task representation returned by query endpoints."""

    task_id: str
    task: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_used: int = 0
    total_cost_usd: float = 0.0
    cost_usd: float = 0.0
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    budget_remaining_pct: float | None = None
    duration_seconds: float = 0.0
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    """Health check liveness response."""

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    status: str = "ready"
    version: str = "0.1.0"
    services: dict[str, str] = Field(default_factory=dict)
