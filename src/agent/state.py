from __future__ import annotations

from typing import Any, TypedDict

from src.agent.models import (
    AgentStatus,
    Message,
    Plan,
    TokenUsage,
    ToolExecutionRecord,
)


def append_list(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer helper to append new items to a list in LangGraph state."""
    return list(left) + list(right)


def update_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Reducer helper to accumulate token usage in LangGraph state."""
    if not left:
        return right
    if not right:
        return left
    return left + right


class AgentState(TypedDict, total=False):
    """Typed schema for the LangGraph agent state machine."""

    task_id: str
    task: str
    plan: Plan | None
    current_step_index: int
    tool_history: list[ToolExecutionRecord]
    reflection_history: list[str]
    retry_count: int
    status: str
    messages: list[Message]
    final_answer: str | None
    error: str | None
    last_error: dict[str, Any] | None
    workspace_id: str | None
    total_tokens: TokenUsage
    metadata: dict[str, Any]


def create_initial_agent_state(
    task_id: str,
    task: str,
    workspace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    """Initializes a new AgentState with clean default channels."""
    return {
        "task_id": task_id,
        "task": task,
        "plan": None,
        "current_step_index": 0,
        "tool_history": [],
        "reflection_history": [],
        "retry_count": 0,
        "status": AgentStatus.INITIALIZING.value,
        "messages": [],
        "final_answer": None,
        "error": None,
        "last_error": None,
        "workspace_id": workspace_id,
        "total_tokens": TokenUsage(),
        "metadata": dict(metadata or {}),
    }
