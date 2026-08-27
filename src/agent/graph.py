from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, cast

import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.agent.context import ContextManager
from src.agent.dispatcher import ToolDispatcher
from src.agent.llm_client import BaseLLMClient
from src.agent.models import (
    AgentStatus,
    LLMResponse,
    Message,
    MessageRole,
    Plan,
    TokenUsage,
    ToolExecutionRecord,
)
from src.agent.prompts import DEFAULT_AGENT_SYSTEM_PROMPT
from src.agent.reflection import (
    CircuitBreaker,
    calculate_backoff_delay,
    generate_failure_report,
    parse_pytest_output,
)
from src.agent.state import AgentState, create_initial_agent_state
from src.agent.token_tracker import format_partial_result
from src.tools.exceptions import BudgetExceededError

logger = structlog.get_logger(__name__)

PLANNING_SYSTEM_PROMPT = """You are an expert autonomous software engineer and planner.
Given a programming task, analyze the requirements and output a strictly structured
execution plan in JSON format.

Your output MUST be a valid JSON object with the following schema:
{
  "task": "<the task description>",
  "rationale": "<brief explanation of your strategy>",
  "steps": [
    {
      "step_id": 1,
      "description": "<what to do in this step>",
      "tool_hint": "<suggested tool e.g. read_file, execute_command, search_code>",
      "expected_output": "<what output is expected>",
      "status": "pending"
    }
  ]
}

Output ONLY the JSON object. Do not wrap in markdown or add conversational text.
"""


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extracts and decodes JSON from raw text or markdown-fenced blocks."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cast(dict[str, Any], json.loads(cleaned))


def _run_coroutine_sync(coro: Any) -> Any:
    """Safely runs an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


class LangGraphAgent:
    """Graph-based autonomous agent with planning, context trimming, and checkpoints."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        dispatcher: ToolDispatcher,
        context_manager: ContextManager | None = None,
        checkpointer: SqliteSaver | None = None,
        db_path: str | Path | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_retries: int = 3,
        max_steps: int = 30,
    ) -> None:
        self.llm_client = llm_client
        self.dispatcher = dispatcher
        self.context_manager = context_manager or ContextManager()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(failure_threshold=3)
        self.max_retries = max_retries
        self.max_steps = max_steps

        if checkpointer is not None:
            self.checkpointer = checkpointer
        elif db_path is not None:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)
        else:
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.checkpointer = SqliteSaver(conn)

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Constructs the LangGraph StateGraph workflow."""
        builder = StateGraph(cast(Any, AgentState))

        builder.add_node("planning", self._planning_node)
        builder.add_node("execution", self._execution_node)
        builder.add_node("reflection", self._reflection_node)

        builder.add_edge(START, "planning")

        builder.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "planning": "planning",
                "execution": "execution",
                "failed": END,
            },
        )

        builder.add_conditional_edges(
            "execution",
            self._route_after_execution,
            {
                "execution": "execution",
                "reflection": "reflection",
                "completed": END,
                "failed": END,
            },
        )

        builder.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {
                "execution": "execution",
                "completed": END,
                "failed": END,
            },
        )

        return builder.compile(checkpointer=self.checkpointer)

    def _call_llm(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Invokes LLM with async/sync compatibility and pre-call budget checks."""
        tracker = getattr(self.llm_client, "token_tracker", None)
        if tracker is not None:
            is_exceeded, b_type, reason = tracker.is_budget_exceeded()
            if is_exceeded:
                raise BudgetExceededError(
                    message=reason or "Task budget exceeded",
                    budget_type=b_type or "tokens",
                    tokens_used=tracker.cumulative_usage.total_tokens,
                    cost_usd=tracker.cumulative_usage.estimated_cost_usd,
                )

        res = self.llm_client.generate_response(messages, tools=tools)
        if asyncio.iscoroutine(res):
            return cast(LLMResponse, _run_coroutine_sync(res))
        return cast(LLMResponse, res)

    def _planning_node(self, state: AgentState) -> dict[str, Any]:
        """Generates structured Pydantic plan with retry on malformed output."""
        task = state.get("task", "")
        retry_count = state.get("retry_count", 0)
        error = state.get("error")

        logger.info("langgraph.planning_node", task=task, retry_count=retry_count)

        prompt_content = f"Create an execution plan for this task:\n{task}"
        if error and retry_count > 0:
            prompt_content += (
                f"\n\nPrevious attempt failed with error:\n{error}\n"
                "Please fix the formatting and output valid JSON matching the schema."
            )

        messages = [
            Message(role=MessageRole.SYSTEM, content=PLANNING_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=prompt_content),
        ]

        try:
            response: LLMResponse = self._call_llm(messages)
            raw_text = response.raw_content or response.thought
            plan_data = extract_json_from_text(raw_text)
            plan = Plan.model_validate(plan_data)

            total_tokens = (
                state.get("total_tokens", TokenUsage()) + response.token_usage
            )

            return {
                "plan": plan,
                "status": AgentStatus.EXECUTING.value,
                "retry_count": 0,
                "error": None,
                "total_tokens": total_tokens,
            }
        except BudgetExceededError as exc:
            partial_res = format_partial_result(
                task=task,
                plan=state.get("plan"),
                tool_history=state.get("tool_history", []),
                budget_reason=exc.message,
            )
            return {
                "status": AgentStatus.FAILED.value,
                "error": f"budget_exceeded: {exc.message}",
                "partial_result": partial_res,
            }
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            new_retries = retry_count + 1
            logger.warning(
                "langgraph.planning_failed",
                error=str(exc),
                retry_count=new_retries,
            )
            if new_retries >= self.max_retries:
                return {
                    "status": AgentStatus.FAILED.value,
                    "retry_count": new_retries,
                    "error": (
                        f"Plan generation failed: malformed structured output "
                        f"after {new_retries} retries ({exc})"
                    ),
                }
            return {
                "status": AgentStatus.PLANNING.value,
                "retry_count": new_retries,
                "error": str(exc),
            }

    def _execution_node(self, state: AgentState) -> dict[str, Any]:
        """Executes the next planned action or produces the final answer."""
        task = state.get("task", "")
        plan = state.get("plan")
        current_step_index = state.get("current_step_index", 0)
        tool_history = list(state.get("tool_history", []))
        reflection_history = list(state.get("reflection_history", []))
        total_tokens = state.get("total_tokens", TokenUsage())

        step_num = len(tool_history) + 1
        if step_num > self.max_steps:
            return {
                "status": AgentStatus.FAILED.value,
                "error": f"Execution exceeded maximum step limit of {self.max_steps}",
            }

        messages = self.context_manager.build_context_messages(
            system_prompt=DEFAULT_AGENT_SYSTEM_PROMPT,
            task=task,
            plan=plan,
            current_step_index=current_step_index,
            tool_history=tool_history,
            reflection_history=reflection_history,
        )

        tool_schemas = self.dispatcher.get_openai_tools_schema()
        try:
            response = self._call_llm(messages, tools=tool_schemas)
        except BudgetExceededError as exc:
            partial_res = format_partial_result(
                task=task,
                plan=plan,
                tool_history=tool_history,
                budget_reason=exc.message,
            )
            return {
                "status": AgentStatus.FAILED.value,
                "error": f"budget_exceeded: {exc.message}",
                "partial_result": partial_res,
            }
        total_tokens = total_tokens + response.token_usage

        if response.final_answer and not response.tool_call:
            return {
                "final_answer": response.final_answer,
                "status": AgentStatus.COMPLETED.value,
                "total_tokens": total_tokens,
            }

        if not response.tool_call:
            return {
                "final_answer": response.thought,
                "status": AgentStatus.COMPLETED.value,
                "total_tokens": total_tokens,
            }

        tool_call = response.tool_call
        logger.info(
            "langgraph.tool_executing",
            step=step_num,
            tool=tool_call.tool_name,
            arguments=tool_call.arguments,
        )

        tool_result = _run_coroutine_sync(self.dispatcher.execute_tool(tool_call))

        record = ToolExecutionRecord(
            step=step_num,
            tool_name=tool_call.tool_name,
            arguments=tool_call.arguments,
            output=tool_result.formatted_content,
            success=tool_result.success,
            duration_seconds=tool_result.duration_seconds,
        )
        tool_history.append(record)

        last_error: dict[str, Any] | None = None
        if not tool_result.success:
            last_error = {
                "tool_name": tool_call.tool_name,
                "code": tool_result.error_code or "TOOL_ERROR",
                "message": tool_result.error or tool_result.output,
                "details": tool_result.error_details,
            }

        return {
            "tool_history": tool_history,
            "last_error": last_error,
            "status": AgentStatus.REFLECTING.value,
            "total_tokens": total_tokens,
        }

    def _reflection_node(self, state: AgentState) -> dict[str, Any]:
        """Evaluates step progress, parses tests, and manages bounded retries."""
        task = state.get("task", "")
        plan = state.get("plan")
        current_step_index = state.get("current_step_index", 0)
        tool_history = state.get("tool_history", [])
        reflection_history = list(state.get("reflection_history", []))
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error")

        if not tool_history:
            return {"status": AgentStatus.EXECUTING.value}

        last_record = tool_history[-1]

        test_result = None
        output_lower = last_record.output.lower()
        if (
            "passed in" in output_lower
            or "failed in" in output_lower
            or "passed," in output_lower
            or "failed," in output_lower
            or "=== short test summary" in output_lower
        ):
            test_result = parse_pytest_output(last_record.output)

        is_step_success = last_record.success
        if test_result and not test_result.all_passed:
            is_step_success = False

        if is_step_success:
            self.circuit_breaker.record_success()

            reflection = (
                f"Step {last_record.step}: Executed "
                f"{last_record.tool_name} successfully."
            )
            if test_result:
                reflection += f" Tests: {test_result.summary}."
            reflection_history.append(reflection)

            next_step_index = current_step_index
            if plan and plan.steps and current_step_index < len(plan.steps):
                plan.steps[current_step_index].status = "completed"
                next_step_index += 1

            return {
                "current_step_index": next_step_index,
                "reflection_history": reflection_history,
                "retry_count": 0,
                "consecutive_failures": 0,
                "test_result": test_result,
                "error": None,
                "status": AgentStatus.EXECUTING.value,
            }

        tripped = self.circuit_breaker.record_failure()
        consec_failures = self.circuit_breaker.consecutive_failures

        if tripped or self.circuit_breaker.is_open:
            logger.error(
                "langgraph.circuit_breaker_tripped",
                consecutive_failures=consec_failures,
            )
            report = generate_failure_report(
                task=task,
                plan=plan,
                tool_history=tool_history,
                last_error=last_error,
                retry_count=retry_count,
                test_result=test_result,
            )
            circuit_err = (
                f"CircuitOpenError: Circuit breaker tripped after "
                f"{consec_failures} consecutive failures\n\n{report}"
            )
            reflection_history.append(
                f"Circuit breaker tripped after {consec_failures} failures."
            )
            return {
                "status": AgentStatus.FAILED.value,
                "error": circuit_err,
                "consecutive_failures": consec_failures,
                "test_result": test_result,
                "reflection_history": reflection_history,
            }

        new_retry = retry_count + 1
        if new_retry >= self.max_retries:
            logger.warning(
                "langgraph.max_retries_exceeded",
                retry_count=new_retry,
                max_retries=self.max_retries,
            )
            report = generate_failure_report(
                task=task,
                plan=plan,
                tool_history=tool_history,
                last_error=last_error,
                retry_count=new_retry,
                test_result=test_result,
            )
            reflection_history.append(
                f"Step {last_record.step} failed. Reached max retries "
                f"({new_retry}/{self.max_retries})."
            )
            return {
                "status": AgentStatus.FAILED.value,
                "error": report,
                "retry_count": new_retry,
                "consecutive_failures": consec_failures,
                "test_result": test_result,
                "reflection_history": reflection_history,
            }

        backoff_delay = calculate_backoff_delay(new_retry)
        logger.info(
            "langgraph.step_retry_scheduled",
            retry_count=new_retry,
            backoff_delay=backoff_delay,
        )
        reflection = (
            f"Step {last_record.step} failed ({last_record.tool_name}). "
            f"Retrying ({new_retry}/{self.max_retries}) "
            f"with backoff {backoff_delay:.1f}s."
        )
        if test_result:
            reflection += f" Test failures: {test_result.summary}."
        reflection_history.append(reflection)

        return {
            "retry_count": new_retry,
            "consecutive_failures": consec_failures,
            "test_result": test_result,
            "reflection_history": reflection_history,
            "status": AgentStatus.EXECUTING.value,
        }

    def _route_after_planning(self, state: AgentState) -> str:
        status = state.get("status")
        if status == AgentStatus.FAILED.value:
            return "failed"
        if state.get("plan") is not None:
            return "execution"
        return "planning"

    def _route_after_execution(self, state: AgentState) -> str:
        status = state.get("status")
        if status == AgentStatus.COMPLETED.value:
            return "completed"
        if status == AgentStatus.FAILED.value:
            return "failed"
        return "reflection"

    def _route_after_reflection(self, state: AgentState) -> str:
        status = state.get("status")
        if status == AgentStatus.COMPLETED.value:
            return "completed"
        if status == AgentStatus.FAILED.value:
            return "failed"
        return "execution"

    def run_task(
        self,
        task_id: str,
        task: str,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> AgentState:
        """Runs a complete task synchronously with checkpointing and budget tracking."""
        tracker = getattr(self.llm_client, "token_tracker", None)
        if tracker is not None:
            if max_tokens is not None:
                tracker.max_tokens = max_tokens
            if max_cost_usd is not None:
                tracker.max_cost_usd = max_cost_usd

        initial_state = create_initial_agent_state(
            task_id=task_id,
            task=task,
            workspace_id=workspace_id,
            metadata=metadata,
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        )
        config = {"configurable": {"thread_id": task_id}}
        return cast(AgentState, self.graph.invoke(initial_state, config=config))

    def resume_task(self, task_id: str) -> AgentState:
        """Resumes a task from its latest checkpoint in SQLite."""
        config = {"configurable": {"thread_id": task_id}}
        return cast(AgentState, self.graph.invoke(None, config=config))

    def get_state(self, task_id: str) -> AgentState | None:
        """Restores latest saved state for a task from the checkpointer."""
        config = {"configurable": {"thread_id": task_id}}
        snapshot = self.graph.get_state(config)
        if snapshot and snapshot.values:
            return cast(AgentState, snapshot.values)
        return None
