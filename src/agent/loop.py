from __future__ import annotations

import time
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

logger = structlog.get_logger(__name__)

MAX_OBSERVATION_CHARS = 8000


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
    ) -> ReasoningTrajectory:
        """Executes the autonomous reasoning loop for a given task description."""
        start_time = time.perf_counter()
        trajectory = ReasoningTrajectory(
            task=task,
            metadata=metadata or {},
        )

        tools_doc = self.dispatcher.get_tools_documentation()
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

            try:
                llm_response = await self.llm_client.generate_response(messages)
            except Exception as exc:
                duration = time.perf_counter() - start_time
                trajectory.status = TrajectoryStatus.FAILED
                trajectory.error = f"LLM generation failed: {exc}"
                trajectory.total_duration_seconds = duration
                logger.error(
                    "reasoning_loop_llm_failed",
                    step=step_num,
                    error=str(exc),
                    exc_info=True,
                )
                return trajectory

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
        logger.warning(
            "reasoning_loop_max_steps_exceeded",
            max_steps=self.max_steps,
            total_tokens=trajectory.total_tokens.total_tokens,
        )
        return trajectory
