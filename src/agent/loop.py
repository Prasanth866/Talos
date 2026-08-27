from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from src.agent.dispatcher import ToolDispatcher
from src.agent.llm_client import BaseLLMClient
from src.agent.models import (
    AgentStep,
    Message,
    MessageRole,
    ReasoningTrajectory,
    TrajectoryStatus,
)
from src.agent.prompts import build_system_prompt
from src.agent.token_tracker import format_partial_result
from src.api.schemas.events import (
    AgentEvent,
    BudgetExceededEvent,
    ErrorEvent,
    TaskCompleteEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolOutputEvent,
)

logger = structlog.get_logger(__name__)

MAX_OBSERVATION_CHARS = 3000


class ReasoningLoop:
    """Coordinates the core agent reasoning loop:
    Task -> LLM Tool Selection -> Tool Dispatch -> Result Feedback -> Trajectory.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        dispatcher: ToolDispatcher,
        max_steps: int = 15,
        max_observation_chars: int = MAX_OBSERVATION_CHARS,
        custom_system_instructions: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.dispatcher = dispatcher
        self.max_steps = max_steps
        self.max_observation_chars = max_observation_chars
        self.custom_system_instructions = custom_system_instructions

    @staticmethod
    def _truncate_observation(text: str, limit: int) -> str:
        """Truncates observation text to stay within context budget."""
        if len(text) <= limit:
            return text
        kept = limit - 80
        return text[:kept] + f"\n... [truncated {len(text) - kept} chars]"

    async def run(
        self,
        task: str,
        metadata: dict[str, Any] | None = None,
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
        task_id: str | None = None,
    ) -> ReasoningTrajectory:
        """Executes the autonomous reasoning loop for a given task description."""
        start_time = time.perf_counter()
        trajectory = ReasoningTrajectory(
            task=task,
            metadata=metadata or {},
        )

        async def _emit(event: AgentEvent) -> None:
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception as cb_err:
                    logger.warning(
                        "event_callback_failed",
                        error=str(cb_err),
                        event_type=event.event_type,
                    )

        tools_doc = self.dispatcher.get_tools_documentation()
        tools_schema = self.dispatcher.get_openai_tools_schema()
        system_prompt = build_system_prompt(
            tools_documentation=tools_doc,
            custom_instructions=self.custom_system_instructions,
        )

        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=f"Task:\n{task}"),
        ]

        logger.info(
            "reasoning_loop_started",
            task=task,
            max_steps=self.max_steps,
            tools=self.dispatcher.get_tool_names(),
        )

        for step_num in range(1, self.max_steps + 1):
            step_start = time.perf_counter()

            tracker = getattr(self.llm_client, "token_tracker", None)
            if tracker is not None:
                is_exceeded, b_type, reason = tracker.is_budget_exceeded()
                if is_exceeded:
                    duration = time.perf_counter() - start_time
                    partial = format_partial_result(
                        task=task,
                        tool_history=trajectory.steps,
                        budget_reason=reason,
                    )
                    trajectory.status = TrajectoryStatus.FAILED
                    trajectory.error = f"budget_exceeded: {reason}"
                    trajectory.final_answer = partial
                    trajectory.total_duration_seconds = duration

                    await _emit(
                        BudgetExceededEvent(
                            tokens_used=tracker.cumulative_usage.total_tokens,
                            cost_usd=tracker.cumulative_usage.estimated_cost_usd,
                            max_tokens=tracker.max_tokens,
                            max_cost_usd=tracker.max_cost_usd,
                            budget_type=b_type or "tokens",
                            partial_result=partial,
                            step=step_num,
                            task_id=task_id,
                        )
                    )
                    return trajectory

            try:
                try:
                    llm_response = await self.llm_client.generate_response(
                        messages=messages,
                        tools=tools_schema if tools_schema else None,
                    )
                except TypeError:
                    llm_response = await self.llm_client.generate_response(messages)
            except Exception as exc:
                duration = time.perf_counter() - start_time
                trajectory.status = TrajectoryStatus.FAILED
                trajectory.error = f"LLM generation failed: {exc}"
                trajectory.total_duration_seconds = duration
                await _emit(
                    ErrorEvent(
                        error=f"LLM generation failed: {exc}",
                        step=step_num,
                        task_id=task_id,
                    )
                )
                logger.error(
                    "reasoning_loop_llm_failed",
                    step=step_num,
                    error=str(exc),
                    exc_info=True,
                )
                return trajectory

            if llm_response.thought:
                await _emit(
                    ThoughtEvent(
                        thought=llm_response.thought,
                        step=step_num,
                        task_id=task_id,
                    )
                )

            if llm_response.final_answer is not None and not llm_response.tool_call:
                step_duration = time.perf_counter() - step_start
                step = AgentStep(
                    step_number=step_num,
                    thought=llm_response.thought,
                    tool_call=None,
                    tool_result=None,
                    token_usage=llm_response.token_usage,
                    duration_seconds=step_duration,
                )
                trajectory.add_step(step)
                trajectory.status = TrajectoryStatus.COMPLETED
                trajectory.final_answer = llm_response.final_answer
                trajectory.total_duration_seconds = time.perf_counter() - start_time

                await _emit(
                    TaskCompleteEvent(
                        task=task,
                        final_answer=llm_response.final_answer,
                        total_steps=len(trajectory.steps),
                        total_tokens=trajectory.total_tokens.total_tokens,
                        total_cost_usd=trajectory.total_cost_usd,
                        duration_seconds=trajectory.total_duration_seconds,
                        task_id=task_id,
                    )
                )

                logger.info(
                    "reasoning_loop_completed",
                    steps=len(trajectory.steps),
                    total_tokens=trajectory.total_tokens.total_tokens,
                    total_cost_usd=trajectory.total_cost_usd,
                    duration_s=round(trajectory.total_duration_seconds, 3),
                )
                return trajectory

            if llm_response.final_answer is not None and llm_response.tool_call:
                logger.warning(
                    "reasoning_loop_ambiguous_response",
                    step=step_num,
                    note="Both final_answer and tool_call returned; tool_call wins.",
                    tool_name=llm_response.tool_call.tool_name,
                )

            if llm_response.tool_call is not None:
                tool_call = llm_response.tool_call
                logger.info(
                    "reasoning_step_tool_call",
                    step=step_num,
                    thought=llm_response.thought,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                )

                await _emit(
                    ToolCallEvent(
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.id,
                        arguments=tool_call.arguments,
                        step=step_num,
                        task_id=task_id,
                    )
                )

                tool_result = await self.dispatcher.execute_tool(tool_call)

                step_duration = time.perf_counter() - step_start
                step = AgentStep(
                    step_number=step_num,
                    thought=llm_response.thought,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    token_usage=llm_response.token_usage,
                    duration_seconds=step_duration,
                )
                trajectory.add_step(step)

                await _emit(
                    ToolOutputEvent(
                        tool_name=tool_result.tool_name,
                        tool_call_id=tool_call.id,
                        output=tool_result.formatted_content,
                        success=tool_result.success,
                        duration_seconds=tool_result.duration_seconds,
                        step=step_num,
                        task_id=task_id,
                    )
                )

                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=llm_response.raw_content,
                    )
                )
                observation = self._truncate_observation(
                    tool_result.formatted_content,
                    self.max_observation_chars,
                )
                obs_content = (
                    f"Observation from tool '{tool_call.tool_name}':\n{observation}"
                )
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=obs_content,
                    )
                )
            else:
                step_duration = time.perf_counter() - step_start
                step = AgentStep(
                    step_number=step_num,
                    thought=llm_response.thought
                    or "Model returned text without a tool call or final answer.",
                    tool_call=None,
                    tool_result=None,
                    token_usage=llm_response.token_usage,
                    duration_seconds=step_duration,
                )
                trajectory.add_step(step)

                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=llm_response.raw_content,
                    )
                )
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=(
                            "Please respond in valid JSON format with either a "
                            "'tool_call' object to execute a tool or a "
                            "'final_answer' string to complete the task."
                        ),
                    )
                )

        trajectory.status = TrajectoryStatus.MAX_STEPS_EXCEEDED
        trajectory.total_duration_seconds = time.perf_counter() - start_time
        trajectory.error = (
            f"Agent reached maximum allowed steps ({self.max_steps}) "
            "without completing the task."
        )
        await _emit(
            ErrorEvent(
                error=trajectory.error or "Unknown error occurred",
                step=self.max_steps,
                task_id=task_id,
            )
        )
        logger.warning(
            "reasoning_loop_max_steps_exceeded",
            max_steps=self.max_steps,
            total_tokens=trajectory.total_tokens.total_tokens,
        )
        return trajectory
