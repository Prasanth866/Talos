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
from src.agent.state import AgentState, create_initial_agent_state

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
        max_retries: int = 3,
        max_steps: int = 30,
    ) -> None:
        self.llm_client = llm_client
        self.dispatcher = dispatcher
        self.context_manager = context_manager or ContextManager()
        self.max_retries = max_retries
        self.max_steps = max_steps

        # Setup SQLite checkpointer
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

        # Routing from planning node
        builder.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {
                "planning": "planning",
                "execution": "execution",
                "failed": END,
            },
        )

        # Routing from execution node
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

        # Routing from reflection node
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
        """Invokes LLM with async/sync compatibility."""
        res = self.llm_client.generate_response(messages, tools=tools)
        if asyncio.iscoroutine(res):
            return cast(LLMResponse, _run_coroutine_sync(res))
        return cast(LLMResponse, res)

    # -------------------------------------------------------------------------
    # GRAPH NODES
    # -------------------------------------------------------------------------
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

        # Build context messages with sliding-window trimmed history
        messages = self.context_manager.build_context_messages(
            system_prompt=DEFAULT_AGENT_SYSTEM_PROMPT,
            task=task,
            plan=plan,
            current_step_index=current_step_index,
            tool_history=tool_history,
            reflection_history=reflection_history,
        )

        tool_schemas = self.dispatcher.get_openai_tools_schema()
        response = self._call_llm(messages, tools=tool_schemas)
        total_tokens = total_tokens + response.token_usage

        # Check if LLM produced final answer
        if response.final_answer and not response.tool_call:
            return {
                "final_answer": response.final_answer,
                "status": AgentStatus.COMPLETED.value,
                "total_tokens": total_tokens,
            }

        if not response.tool_call:
            # If no tool call and no explicit final answer, treat thought as answer
            return {
                "final_answer": response.thought,
                "status": AgentStatus.COMPLETED.value,
                "total_tokens": total_tokens,
            }

        # Execute tool call
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
        """Evaluates step progress and updates plan step status."""
        plan = state.get("plan")
        current_step_index = state.get("current_step_index", 0)
        tool_history = state.get("tool_history", [])
        reflection_history = list(state.get("reflection_history", []))

        if tool_history:
            last_record = tool_history[-1]
            reflection = (
                f"Step {last_record.step}: Executed {last_record.tool_name} "
                f"({'Success' if last_record.success else 'Failed'})."
            )
            reflection_history.append(reflection)

        # Advance plan step if applicable
        next_step_index = current_step_index
        if (
            plan
            and plan.steps
            and current_step_index < len(plan.steps)
            and tool_history
            and tool_history[-1].success
        ):
            plan.steps[current_step_index].status = "completed"
            next_step_index += 1

        return {
            "current_step_index": next_step_index,
            "reflection_history": reflection_history,
            "status": AgentStatus.EXECUTING.value,
        }

    # -------------------------------------------------------------------------
    # ROUTING CONDITIONS
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    def run_task(
        self,
        task_id: str,
        task: str,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        """Runs a complete task synchronously with checkpointing."""
        initial_state = create_initial_agent_state(
            task_id=task_id,
            task=task,
            workspace_id=workspace_id,
            metadata=metadata,
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
