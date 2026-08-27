from pathlib import Path
from typing import Any

import pytest

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import TrajectoryStatus


@pytest.mark.asyncio
async def test_real_failing_test_task_end_to_end(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()

    buggy_code = (
        "def calculate_average(numbers: list[float]) -> float:\n"
        "    if not numbers:\n"
        "        return 0.0\n"
        "    # BUG: using len(numbers) - 1 causes ZeroDivisionError\n"
        "    return sum(numbers) / (len(numbers) - 1)\n"
    )
    test_code = (
        "from stats import calculate_average\n\n"
        "def test_average_single_element():\n"
        "    assert calculate_average([10.0]) == 10.0\n\n"
        "def test_average_multiple_elements():\n"
        "    assert calculate_average([2.0, 4.0, 6.0]) == 4.0\n"
    )
    (sandbox_dir / "stats.py").write_text(buggy_code, encoding="utf-8")
    (sandbox_dir / "test_stats.py").write_text(test_code, encoding="utf-8")

    dispatcher = create_default_dispatcher(sandbox_dir)

    fixed_content = (
        "def calculate_average(numbers: list[float]) -> float:\n"
        "    if not numbers:\n"
        "        return 0.0\n"
        "    return sum(numbers) / len(numbers)\n"
    )

    mock_responses: list[dict[str, Any]] = [
        {
            "thought": "First, I need to run pytest to observe what test is failing.",
            "tool_call": {
                "tool_name": "run_shell",
                "arguments": {"command": "pytest test_stats.py"},
            },
        },
        {
            "thought": "Test failed with ZeroDivisionError. Let's inspect stats.py.",
            "tool_call": {
                "tool_name": "read_file",
                "arguments": {"path": "stats.py"},
            },
        },
        {
            "thought": "Bug found: divisor is len(numbers) - 1. Writing fix.",
            "tool_call": {
                "tool_name": "write_file",
                "arguments": {
                    "path": "stats.py",
                    "content": fixed_content,
                },
            },
        },
        {
            "thought": "Now let's verify that the tests pass with pytest.",
            "tool_call": {
                "tool_name": "run_shell",
                "arguments": {"command": "pytest test_stats.py"},
            },
        },
        {
            "thought": "All tests pass. The bug in calculate_average is resolved.",
            "final_answer": (
                "Fixed calculate_average by changing divisor from len-1 to len. "
                "Verified all tests pass with pytest."
            ),
        },
    ]

    mock_llm = MockLLMClient(responses=mock_responses, model_name="gpt-4o")
    runner = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=10)

    task_description = (
        "Fix failing unit tests in test_stats.py by diagnosing and correcting stats.py."
    )
    trajectory = await runner.run(task_description)

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert trajectory.final_answer is not None
    assert "Fixed calculate_average" in trajectory.final_answer
    assert len(trajectory.steps) == 5
    assert trajectory.tool_call_count == 4
    assert trajectory.total_tokens.total_tokens > 0
    assert trajectory.total_cost_usd > 0.0

    step1 = trajectory.steps[0]
    assert step1.tool_call is not None
    assert step1.tool_call.tool_name == "run_shell"
    assert step1.tool_result is not None
    assert (
        not step1.tool_result.success
        or "FAILED" in step1.tool_result.output
        or "Error" in step1.tool_result.formatted_content
    )

    step3 = trajectory.steps[2]
    assert step3.tool_call is not None
    assert step3.tool_call.tool_name == "write_file"
    assert step3.tool_result is not None
    assert step3.tool_result.success is True

    step4 = trajectory.steps[3]
    assert step4.tool_call is not None
    assert step4.tool_call.tool_name == "run_shell"
    assert step4.tool_result is not None
    assert step4.tool_result.success is True
    assert "passed" in step4.tool_result.output

    updated_content = (sandbox_dir / "stats.py").read_text(encoding="utf-8")
    assert "return sum(numbers) / len(numbers)" in updated_content
