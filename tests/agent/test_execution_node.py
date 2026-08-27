from __future__ import annotations

import json

from src.agent.dispatcher import ToolDispatcher
from src.agent.graph import LangGraphAgent
from src.agent.llm_client import MockLLMClient
from src.agent.models import (
    AgentStatus,
    LLMResponse,
    ToolCall,
)
from src.tools.exceptions import CommandExecutionError


def test_valid_tool_call_dispatches_correctly() -> None:
    """Unit test: Valid tool call passes schema validation and executes."""
    dispatcher = ToolDispatcher()
    called = []

    def sample_tool(filepath: str, count: int = 1) -> str:
        called.append((filepath, count))
        return f"Processed {filepath} {count} times"

    dispatcher.register_tool(
        name="sample_tool",
        description="Processes a file",
        handler=sample_tool,
        parameters_schema={
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["filepath"],
        },
    )

    plan_json = json.dumps(
        {
            "task": "Test valid tool call",
            "rationale": "Direct execution",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Run tool",
                    "expected_output": "Processed",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Executing sample tool",
            tool_call=ToolCall(
                tool_name="sample_tool",
                arguments={"filepath": "test.txt", "count": 3},
            ),
        ),
        LLMResponse(thought="Finished", final_answer="All done"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(llm_client=llm, dispatcher=dispatcher)
    state = agent.run_task(task_id="valid-call-1", task="Test valid tool call")

    assert state["status"] == AgentStatus.COMPLETED.value
    assert state["last_error"] is None
    assert len(state["tool_history"]) == 1
    assert state["tool_history"][0].tool_name == "sample_tool"
    assert state["tool_history"][0].success is True
    assert "Processed test.txt 3 times" in state["tool_history"][0].output
    assert called == [("test.txt", 3)]


async def test_invalid_args_rejected_with_schema_validation_error() -> None:
    """Unit test: Invalid tool arguments are rejected by schema validation."""
    dispatcher = ToolDispatcher()
    handler_executed = []

    def strict_tool(filename: str, mode: str) -> str:
        handler_executed.append((filename, mode))
        return "OK"

    dispatcher.register_tool(
        name="strict_tool",
        description="Strict file handler",
        handler=strict_tool,
        parameters_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "mode": {"type": "string", "enum": ["read", "write"]},
            },
            "required": ["filename", "mode"],
        },
    )

    # 1. Test missing required field "mode"
    tool_call_missing = ToolCall(
        tool_name="strict_tool",
        arguments={"filename": "doc.txt"},
    )
    res1 = await dispatcher.execute_tool(tool_call_missing)
    assert res1.success is False
    assert res1.error_code == "SCHEMA_VALIDATION_ERROR"
    assert "mode" in (res1.error or "")
    assert len(handler_executed) == 0

    # 2. Test wrong enum value for "mode"
    tool_call_enum = ToolCall(
        tool_name="strict_tool",
        arguments={"filename": "doc.txt", "mode": "delete"},
    )
    res2 = await dispatcher.execute_tool(tool_call_enum)
    assert res2.success is False
    assert res2.error_code == "SCHEMA_VALIDATION_ERROR"
    assert "delete" in (res2.error or "")
    assert len(handler_executed) == 0

    # 3. Test wrong type (integer instead of string for filename)
    tool_call_type = ToolCall(
        tool_name="strict_tool",
        arguments={"filename": 12345, "mode": "read"},
    )
    res3 = await dispatcher.execute_tool(tool_call_type)
    assert res3.success is False
    assert res3.error_code == "SCHEMA_VALIDATION_ERROR"
    assert len(handler_executed) == 0


def test_execution_errors_produce_typed_tool_error_in_state() -> None:
    """Unit test: Tool exceptions produce typed ToolError in agent state last_error."""
    dispatcher = ToolDispatcher()

    def failing_tool(cmd: str) -> str:
        raise CommandExecutionError(
            message=f"Command '{cmd}' failed with non-zero exit code",
            tool_name="failing_tool",
            exit_code=127,
            stderr="command not found: " + cmd,
        )

    dispatcher.register_tool(
        name="failing_tool",
        description="A tool that fails during execution",
        handler=failing_tool,
        parameters_schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    )

    plan_json = json.dumps(
        {
            "task": "Test error tracking",
            "rationale": "Trigger tool execution error",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Run failing tool",
                    "expected_output": "error",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Calling failing tool",
            tool_call=ToolCall(
                tool_name="failing_tool", arguments={"cmd": "invalid_cmd"}
            ),
        ),
        LLMResponse(thought="Handling error", final_answer="Caught error cleanly"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(llm_client=llm, dispatcher=dispatcher)
    state = agent.run_task(task_id="error-task-1", task="Test error tracking")

    assert len(state["tool_history"]) == 1
    record = state["tool_history"][0]
    assert record.success is False
    assert record.tool_name == "failing_tool"
    assert "Command 'invalid_cmd' failed" in record.output

    # Check last_error populated with typed error
    last_err = state["last_error"]
    assert last_err is not None
    assert last_err["tool_name"] == "failing_tool"
    assert last_err["code"] == "COMMAND_FAILED"
    assert last_err["details"]["exit_code"] == 127


def test_tool_results_stored_in_tool_history() -> None:
    """Unit test: Sequential tool calls are reliably stored in tool_history."""
    dispatcher = ToolDispatcher()
    dispatcher.register_tool(
        name="step_tool",
        description="Step tracker",
        handler=lambda num: f"Step {num} completed",
        parameters_schema={
            "type": "object",
            "properties": {"num": {"type": "integer"}},
            "required": ["num"],
        },
    )

    plan_json = json.dumps(
        {
            "task": "Multi-step tool history test",
            "rationale": "Execute 2 steps",
            "steps": [
                {"step_id": 1, "description": "Step 1", "expected_output": "Step 1"},
                {"step_id": 2, "description": "Step 2", "expected_output": "Step 2"},
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Call 1",
            tool_call=ToolCall(tool_name="step_tool", arguments={"num": 1}),
        ),
        LLMResponse(
            thought="Call 2",
            tool_call=ToolCall(tool_name="step_tool", arguments={"num": 2}),
        ),
        LLMResponse(thought="Done", final_answer="All steps recorded"),
    ]
    llm = MockLLMClient(responses=mock_responses)

    agent = LangGraphAgent(llm_client=llm, dispatcher=dispatcher)
    state = agent.run_task(
        task_id="history-task-1", task="Multi-step tool history test"
    )

    assert len(state["tool_history"]) == 2
    assert state["tool_history"][0].step == 1
    assert state["tool_history"][0].output == "Step 1 completed"
    assert state["tool_history"][1].step == 2
    assert state["tool_history"][1].output == "Step 2 completed"
    assert state["status"] == AgentStatus.COMPLETED.value
