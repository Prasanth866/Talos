from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.context import ContextManager
from src.agent.models import MessageRole, ToolExecutionRecord
from src.agent.reflection import parse_pytest_output
from src.tools.patch import PatchTool


def test_regression_parse_pytest_output_ansi_escape_and_params() -> None:
    """Regression Test 1:

    Verifies parse_pytest_output strips ANSI escape codes and extracts tests.
    """
    raw_pytest_with_ansi = (
        "\x1b[1m=== test session starts ===\x1b[0m\n"
        "collected 3 items\n\n"
        "\x1b[31mFAILED\x1b[0m tests/test_cli.py::"
        "\x1b[1mtest_flags[--no-cache]\x1b[0m - Err\n"
        "\x1b[32mPASSED\x1b[0m tests/test_cli.py::test_default\n"
        "\x1b[32mPASSED\x1b[0m tests/test_cli.py::test_verbose\n\n"
        "\x1b[31m=== 1 failed, 2 passed in 0.12s ===\x1b[0m\n"
    )

    result = parse_pytest_output(raw_pytest_with_ansi)
    assert result.passed == 2
    assert result.failed == 1
    assert result.all_passed is False
    assert len(result.failure_details) == 1
    assert "test_flags[--no-cache]" in result.failure_details[0]
    assert "\x1b" not in result.summary


@pytest.mark.asyncio
async def test_regression_patch_tool_mixed_line_endings(tmp_path: Path) -> None:
    """Regression Test 2:

    Verifies PatchTool cleanly applies unified diffs to files with CRLF endings.
    """
    target_file = tmp_path / "crlf_sample.py"
    crlf_content = (
        "def calculate(a, b):\r\n    # initial comment\r\n    return a + b\r\n"
    )
    target_file.write_bytes(crlf_content.encode("utf-8"))

    patch_diff = (
        f"--- a/{target_file.name}\n"
        f"+++ b/{target_file.name}\n"
        "@@ -1,3 +1,3 @@\n"
        " def calculate(a, b):\n"
        "-    # initial comment\n"
        "-    return a + b\n"
        "+    return a * b\n"
    )

    tool = PatchTool(sandbox_dir=tmp_path)
    res = await tool.apply_patch(patch=patch_diff, dry_run=False)

    assert res["success"] is True
    assert target_file.name in res["files_modified"]
    updated_text = target_file.read_text(encoding="utf-8")
    assert "return a * b" in updated_text


def test_regression_context_manager_preserves_task_prompt() -> None:
    """Regression Test 3:

    Verifies ContextManager sliding window preserves the initial task prompt.
    """
    ctx = ContextManager(max_recent_records=4, max_history_tokens=500)

    history = [
        ToolExecutionRecord(
            step=i,
            tool_name=f"tool_{i}",
            arguments={"arg": i},
            output=f"Output for step {i} " * 20,
            success=True,
            duration_seconds=0.1,
        )
        for i in range(1, 21)
    ]

    messages = ctx.build_context_messages(
        system_prompt="Custom system prompt instructions",
        task="Critical initial user problem statement",
        tool_history=history,
    )

    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert "Custom system prompt instructions" in messages[0].content

    assert messages[1].role == MessageRole.USER
    assert "Critical initial user problem statement" in messages[1].content

    assert "tool_20" in messages[-1].content
