from src.agent.models import (
    AgentStep,
    CostRates,
    LLMResponse,
    Message,
    MessageRole,
    ReasoningTrajectory,
    TokenUsage,
    ToolCall,
    ToolResult,
    TrajectoryStatus,
)


def test_token_usage_addition() -> None:
    u1 = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        estimated_cost_usd=0.001,
    )
    u2 = TokenUsage(
        prompt_tokens=200,
        completion_tokens=80,
        total_tokens=280,
        estimated_cost_usd=0.002,
    )
    u3 = u1 + u2
    assert u3.prompt_tokens == 300
    assert u3.completion_tokens == 130
    assert u3.total_tokens == 430
    assert abs(u3.estimated_cost_usd - 0.003) < 1e-6


def test_cost_rates_calculation() -> None:
    rates = CostRates(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00)
    cost = rates.calculate_cost(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost == 7.50


def test_tool_result_formatted_content() -> None:
    res_success = ToolResult(tool_name="read_file", output="hello world", success=True)
    assert res_success.formatted_content == "hello world"

    res_fail = ToolResult(tool_name="read_file", error="File not found", success=False)
    assert res_fail.formatted_content == "Error (read_file): File not found"


def test_agent_step_summary_dict() -> None:
    step = AgentStep(
        step_number=1,
        thought="I should inspect files",
        tool_call=ToolCall(tool_name="list_dir", arguments={"path": "."}),
        tool_result=ToolResult(
            tool_name="list_dir", output="file1.py\nfile2.py", success=True
        ),
        token_usage=TokenUsage(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            estimated_cost_usd=0.0003,
        ),
        duration_seconds=0.15,
    )
    summary = step.to_summary_dict()
    assert summary["step"] == 1
    assert summary["thought"] == "I should inspect files"
    assert summary["action"] == {"tool": "list_dir", "arguments": {"path": "."}}
    assert summary["result"] == {"success": True, "output": "file1.py\nfile2.py"}
    assert summary["tokens"] == 70
    assert summary["duration_s"] == 0.15


def test_reasoning_trajectory_formatting() -> None:
    trajectory = ReasoningTrajectory(
        task="Fix the math bug",
        status=TrajectoryStatus.COMPLETED,
        final_answer="Fixed the return value in math.py",
        total_duration_seconds=1.25,
    )
    step1 = AgentStep(
        step_number=1,
        thought="Let's read the file",
        tool_call=ToolCall(tool_name="read_file", arguments={"path": "math.py"}),
        tool_result=ToolResult(
            tool_name="read_file", output="def add(a, b): return a - b", success=True
        ),
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130,
            estimated_cost_usd=0.0005,
        ),
        duration_seconds=0.4,
    )
    trajectory.add_step(step1)

    formatted = trajectory.to_formatted_trajectory()
    assert "=== Reasoning Trajectory: Fix the math bug ===" in formatted
    assert "Status: COMPLETED" in formatted
    assert "Step 1" in formatted
    assert "Action: read_file" in formatted
    assert "Final Answer:" in formatted
    assert "Fixed the return value in math.py" in formatted
    assert trajectory.tool_call_count == 1
    assert trajectory.total_tokens.total_tokens == 130


def test_message_and_llm_response_creation() -> None:
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"

    resp = LLMResponse(
        thought="Thinking...",
        final_answer="Done",
        raw_content='{"thought": "Thinking...", "final_answer": "Done"}',
    )
    assert resp.thought == "Thinking..."
    assert resp.final_answer == "Done"


def test_reasoning_trajectory_error_formatting() -> None:
    trajectory = ReasoningTrajectory(
        task="Fail task",
        status=TrajectoryStatus.FAILED,
        error="LLM API rate limit exceeded",
    )
    formatted = trajectory.to_formatted_trajectory()
    assert "Status: FAILED" in formatted
    assert "Error:\nLLM API rate limit exceeded" in formatted
