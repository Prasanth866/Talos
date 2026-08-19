from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool call chosen by the LLM."""

    id: str = Field(default_factory=lambda: f"call_{uuid4().hex[:8]}")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Represents the execution outcome of a tool call."""

    tool_name: str
    output: str = ""
    error: str | None = None
    success: bool = True
    duration_seconds: float = 0.0

    @property
    def formatted_content(self) -> str:
        """Standardized string representation to feed back to the LLM."""
        if not self.success or self.error:
            return f"Error ({self.tool_name}): {self.error or self.output}"
        return self.output


class TokenUsage(BaseModel):
    """Tracks token consumption and dollar cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


class CostRates(BaseModel):
    """Pricing rates per 1,000,000 tokens in USD."""

    prompt_cost_per_1m: float = 2.50  # Default ~ GPT-4o input
    completion_cost_per_1m: float = 10.00  # Default ~ GPT-4o output

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (prompt_tokens * self.prompt_cost_per_1m / 1_000_000.0) + (
            completion_tokens * self.completion_cost_per_1m / 1_000_000.0
        )
        return round(cost, 6)


class Message(BaseModel):
    """Standard message in conversational context."""

    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Structured parsed response from an LLM call."""

    thought: str = ""
    tool_call: ToolCall | None = None
    final_answer: str | None = None
    raw_content: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0


class AgentStep(BaseModel):
    """A single step in the reasoning trajectory."""

    step_number: int
    thought: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_summary_dict(self) -> dict[str, Any]:
        """Provides a concise dictionary for structured logging."""
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": (
                {
                    "tool": self.tool_call.tool_name,
                    "arguments": self.tool_call.arguments,
                }
                if self.tool_call
                else None
            ),
            "result": (
                {
                    "success": self.tool_result.success,
                    "output": self.tool_result.formatted_content,
                }
                if self.tool_result
                else None
            ),
            "tokens": self.token_usage.total_tokens,
            "cost_usd": self.token_usage.estimated_cost_usd,
            "duration_s": round(self.duration_seconds, 3),
        }


class TrajectoryStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    FAILED = "failed"


class ReasoningTrajectory(BaseModel):
    """Full execution trajectory of the reasoning loop."""

    task: str
    steps: list[AgentStep] = Field(default_factory=list)
    status: TrajectoryStatus = TrajectoryStatus.IN_PROGRESS
    final_answer: str | None = None
    error: str | None = None
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.total_tokens = self.total_tokens + step.token_usage

    @property
    def total_cost_usd(self) -> float:
        return self.total_tokens.estimated_cost_usd

    @property
    def tool_call_count(self) -> int:
        return sum(1 for step in self.steps if step.tool_call is not None)

    def to_formatted_trajectory(self) -> str:
        """Renders human-readable reasoning trajectory for debugging and review."""
        lines = [
            f"=== Reasoning Trajectory: {self.task} ===",
            f"Status: {self.status.value.upper()}",
            f"Total Steps: {len(self.steps)} | Tool Calls: {self.tool_call_count}",
            (
                f"Tokens: {self.total_tokens.total_tokens} "
                f"(Prompt: {self.total_tokens.prompt_tokens}, "
                f"Completion: {self.total_tokens.completion_tokens})"
            ),
            f"Estimated Cost: ${self.total_cost_usd:.5f} USD",
            f"Duration: {self.total_duration_seconds:.2f}s",
            "-" * 60,
        ]

        for step in self.steps:
            lines.append(f"\n[Step {step.step_number}]")
            if step.thought:
                lines.append(f"  Thought: {step.thought}")
            if step.tool_call:
                lines.append(
                    f"  Action: {step.tool_call.tool_name}({step.tool_call.arguments})"
                )
            if step.tool_result:
                status_str = "SUCCESS" if step.tool_result.success else "FAILED"
                obs = step.tool_result.formatted_content
                lines.append(f"  Observation ({status_str}): {obs}")
            tokens_str = f"{step.token_usage.total_tokens} tokens"
            cost_str = f"${step.token_usage.estimated_cost_usd:.5f}"
            time_str = f"{step.duration_seconds:.2f}s"
            lines.append(f"  Metrics: {tokens_str} | {cost_str} | {time_str}")

        if self.final_answer:
            lines.append("\n" + "=" * 60)
            lines.append(f"Final Answer:\n{self.final_answer}")
        elif self.error:
            lines.append("\n" + "=" * 60)
            lines.append(f"Error:\n{self.error}")

        return "\n".join(lines)
