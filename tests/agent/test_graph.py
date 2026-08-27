from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.context import ContextManager
from src.agent.dispatcher import ToolDispatcher
from src.agent.graph import LangGraphAgent
from src.agent.llm_client import MockLLMClient
from src.agent.models import (
    AgentStatus,
    LLMResponse,
    Plan,
    TokenUsage,
    ToolCall,
    ToolExecutionRecord,
)
from src.agent.state import create_initial_agent_state


def test_agent_state_schema_validates_correctly() -> None:
    """Unit test: AgentState schema initializes with correct typed fields."""
    state = create_initial_agent_state(
        task_id="task-123",
        task="Refactor codebase",
        workspace_id="ws-abc",
        metadata={"priority": "high"},
    )

    assert state["task_id"] == "task-123"
    assert state["task"] == "Refactor codebase"
    assert state["workspace_id"] == "ws-abc"
    assert state["metadata"]["priority"] == "high"
    assert state["status"] == AgentStatus.INITIALIZING.value
    assert state["current_step_index"] == 0
    assert len(state["tool_history"]) == 0
    assert len(state["reflection_history"]) == 0
    assert state["retry_count"] == 0
    assert state["plan"] is None
    assert isinstance(state["total_tokens"], TokenUsage)


def test_structured_plan_validation() -> None:
    """Unit test: Plan and PlanStep models parse structured outputs."""
    raw_json = json.dumps(
        {
            "task": "Add user authentication",
            "rationale": "We need to verify user credentials before giving access.",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Create user model in src/db/models.py",
                    "tool_hint": "write_file",
                    "expected_output": "User model defined with hashed_password",
                    "status": "pending",
                },
                {
                    "step_id": 2,
                    "description": "Run authentication unit tests",
                    "tool_hint": "execute_command",
                    "expected_output": "pytest passes",
                    "status": "pending",
                },
            ],
        }
    )

    plan = Plan.model_validate_json(raw_json)
    assert plan.task == "Add user authentication"
    assert len(plan.steps) == 2
    assert plan.steps[0].step_id == 1
    assert plan.steps[0].tool_hint == "write_file"
    assert plan.steps[1].expected_output == "pytest passes"


def test_context_window_trimming_keeps_history_within_limits() -> None:
    """Unit test: ContextManager compacts older history when 10+ calls occur."""
    ctx_mgr = ContextManager(max_recent_records=3, max_history_tokens=1000)

    records: list[ToolExecutionRecord] = []
    for i in range(1, 13):
        records.append(
            ToolExecutionRecord(
                step=i,
                tool_name=f"tool_{i}",
                arguments={"path": f"file_{i}.py"},
                output=f"Extremely verbose content for step {i} " * 50,
                success=True,
                duration_seconds=0.1,
            )
        )

    trimmed = ctx_mgr.trim_tool_history(records, max_recent=3)
    assert len(trimmed) == 12

    for r in trimmed[:9]:
        assert len(r.output) < 250
        assert f"[Step {r.step}]" in r.output

    for r in trimmed[9:]:
        assert len(r.output) > 500

    raw_tokens = ctx_mgr.estimate_tool_history_tokens(records)
    trimmed_tokens = ctx_mgr.estimate_tool_history_tokens(trimmed)
    assert trimmed_tokens < raw_tokens


def test_checkpoint_persistence_saves_and_restores_state(tmp_path: Path) -> None:
    """Unit test: SQLite checkpointer saves and restores state across restarts."""
    db_file = tmp_path / "test_checkpoints.db"

    dispatcher = ToolDispatcher()
    plan_json = json.dumps(
        {
            "task": "Test checkpoint task",
            "rationale": "Step-by-step verification",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Inspect structure",
                    "tool_hint": "search_code",
                    "expected_output": "code files",
                    "status": "pending",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Planning complete"),
        LLMResponse(thought="Task finished successfully", final_answer="Done!"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent1 = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
        db_path=db_file,
    )
    final_state = agent1.run_task(task_id="persist-task-1", task="Test checkpoint task")
    assert final_state["status"] == AgentStatus.COMPLETED.value
    assert final_state["final_answer"] == "Done!"
    assert final_state["plan"] is not None

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    agent2 = LangGraphAgent(
        llm_client=MockLLMClient(),
        dispatcher=dispatcher,
        checkpointer=checkpointer,
    )

    restored_state = agent2.get_state("persist-task-1")
    assert restored_state is not None
    assert restored_state["task_id"] == "persist-task-1"
    assert restored_state["status"] == AgentStatus.COMPLETED.value
    assert restored_state["final_answer"] == "Done!"
    assert restored_state["plan"] is not None
    assert restored_state["plan"].task == "Test checkpoint task"


def test_malformed_output_triggers_retry_up_to_max_then_failed_state() -> None:
    """Unit test: Malformed output triggers retry up to max_retries, then fails."""
    dispatcher = ToolDispatcher()

    mock_responses = [
        LLMResponse(raw_content="not valid json at all", thought="bad 1"),
        LLMResponse(raw_content="{'broken': json}", thought="bad 2"),
        LLMResponse(raw_content="still bad json", thought="bad 3"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
        max_retries=3,
    )

    state = agent.run_task(task_id="retry-fail-task", task="Build feature")
    assert state["status"] == AgentStatus.FAILED.value
    assert state["retry_count"] == 3
    assert "malformed structured output after 3 retries" in (state["error"] or "")


def test_malformed_output_recovers_on_subsequent_retry() -> None:
    """Unit test: Agent recovers if LLM fixes its structured output on retry."""
    dispatcher = ToolDispatcher()

    valid_plan_json = json.dumps(
        {
            "task": "Build feature",
            "rationale": "Fix output on second try",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Step 1",
                    "tool_hint": None,
                    "expected_output": "Success",
                    "status": "pending",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content="invalid json...", thought="bad attempt"),
        LLMResponse(raw_content=valid_plan_json, thought="corrected plan"),
        LLMResponse(thought="All done", final_answer="Success!"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
        max_retries=3,
    )

    state = agent.run_task(task_id="recovery-task", task="Build feature")
    assert state["status"] == AgentStatus.COMPLETED.value
    assert state["final_answer"] == "Success!"
    assert state["plan"] is not None
    assert state["plan"].task == "Build feature"


def test_full_langgraph_task_execution_flow() -> None:
    """Unit test: Full task execution with tool invocation and completion."""
    dispatcher = ToolDispatcher()
    dispatcher.register_tool(
        name="echo_tool",
        description="Echoes input back",
        handler=lambda msg: f"Echo: {msg}",
        parameters_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
        },
    )

    plan_json = json.dumps(
        {
            "task": "Execute echo task",
            "rationale": "Call echo and finish",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Echo hello message",
                    "tool_hint": "echo_tool",
                    "expected_output": "Echo: hello",
                    "status": "pending",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Calling echo tool",
            tool_call=ToolCall(tool_name="echo_tool", arguments={"msg": "hello"}),
        ),
        LLMResponse(
            thought="Observation received, task complete",
            final_answer="The echo returned 'Echo: hello'.",
        ),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
    )

    state = agent.run_task(task_id="echo-task-1", task="Execute echo task")
    assert state["status"] == AgentStatus.COMPLETED.value
    assert len(state["tool_history"]) == 1
    assert state["tool_history"][0].tool_name == "echo_tool"
    assert state["tool_history"][0].output == "Echo: hello"
    assert len(state["reflection_history"]) >= 1
    assert state["final_answer"] == "The echo returned 'Echo: hello'."
