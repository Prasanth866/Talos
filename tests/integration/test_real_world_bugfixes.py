from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.agent.dispatcher import create_default_dispatcher
from src.agent.graph import LangGraphAgent
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import (
    AgentStatus,
    CostRates,
    LLMResponse,
    TokenUsage,
    ToolCall,
    TrajectoryStatus,
)
from src.agent.token_tracker import TokenTracker


@dataclass
class BugFixBenchmarkMetrics:
    task_name: str
    passed: bool
    step_count: int
    tool_call_count: int
    retry_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    wall_clock_seconds: float


@pytest.mark.asyncio
async def test_real_world_bugfix_task1_urllib3_headers(tmp_path: Path) -> None:
    """Evaluation Task 1 (psf/requests & urllib3 pattern):

    CaseInsensitiveDict with folded multi-value headers.
    """
    repo_dir = tmp_path / "urllib3_headers_repo"
    src_dir = repo_dir / "src"
    tests_dir = repo_dir / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    buggy_structures = (
        "from collections.abc import MutableMapping\n"
        "from typing import Any, Iterator\n\n"
        "class CaseInsensitiveDict(MutableMapping[str, Any]):\n"
        "    def __init__(self, data: dict[str, Any] | None = None) -> None:\n"
        "        self._store: dict[str, tuple[str, Any]] = {}\n"
        "        if data:\n"
        "            self.update(data)\n\n"
        "    def __setitem__(self, key: str, value: Any) -> None:\n"
        "        # BUG: drops original key casing and strips folded whitespace\n"
        "        self._store[key.lower()] = (key.lower(), value)\n\n"
        "    def __getitem__(self, key: str) -> Any:\n"
        "        return self._store[key.lower()][1]\n\n"
        "    def __delitem__(self, key: str) -> None:\n"
        "        # BUG: unhandled KeyError on case mismatch\n"
        "        del self._store[key]\n\n"
        "    def __iter__(self) -> Iterator[str]:\n"
        "        return (orig_key for orig_key, _ in self._store.values())\n\n"
        "    def __len__(self) -> int:\n"
        "        return len(self._store)\n\n"
        "    def getlist(self, key: str) -> list[Any]:\n"
        "        val = self.get(key)\n"
        "        if val is None:\n"
        "            return []\n"
        "        return [val] if not isinstance(val, list) else list(val)\n"
    )
    (src_dir / "structures.py").write_text(buggy_structures, encoding="utf-8")

    test_code = (
        "import pytest\n"
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "from src.structures import CaseInsensitiveDict\n\n"
        "def test_case_preservation_and_lookup():\n"
        "    d = CaseInsensitiveDict({'Accept': 'application/json'})\n"
        "    assert d['accept'] == 'application/json'\n"
        "    assert d['ACCEPT'] == 'application/json'\n"
        "    assert list(d.keys()) == ['Accept']\n\n"
        "def test_case_insensitive_deletion():\n"
        "    d = CaseInsensitiveDict({'Content-Type': 'text/html'})\n"
        "    del d['content-type']\n"
        "    assert 'Content-Type' not in d\n"
        "    assert len(d) == 0\n\n"
        "def test_getlist_multi_header():\n"
        "    d = CaseInsensitiveDict({'Set-Cookie': ['a=1', 'b=2']})\n"
        "    assert d.getlist('set-cookie') == ['a=1', 'b=2']\n"
        "    assert d.getlist('non-existent') == []\n"
    )
    (tests_dir / "test_structures.py").write_text(test_code, encoding="utf-8")

    patch_diff = (
        "--- a/src/structures.py\n"
        "+++ b/src/structures.py\n"
        "@@ -10,4 +10,4 @@\n"
        "     def __setitem__(self, key: str, value: Any) -> None:\n"
        "-        # BUG: drops original key casing and strips folded whitespace\n"
        "-        self._store[key.lower()] = (key.lower(), value)\n"
        "+        self._store[key.lower()] = (key, value)\n"
        " \n"
        "     def __getitem__(self, key: str) -> Any:\n"
        "@@ -17,3 +17,3 @@\n"
        "     def __delitem__(self, key: str) -> None:\n"
        "-        # BUG: unhandled KeyError on case mismatch\n"
        "-        del self._store[key]\n"
        "+        del self._store[key.lower()]\n"
    )

    dispatcher = create_default_dispatcher(repo_dir)

    mock_responses = [
        LLMResponse(
            thought="Running pytest on test_structures.py.",
            tool_call=ToolCall(
                tool_name="run_shell",
                arguments={"command": "pytest tests/test_structures.py"},
            ),
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=50),
        ),
        LLMResponse(
            thought="Inspecting src/structures.py.",
            tool_call=ToolCall(
                tool_name="read_file",
                arguments={"path": "src/structures.py"},
            ),
            token_usage=TokenUsage(prompt_tokens=220, completion_tokens=60),
        ),
        LLMResponse(
            thought="Applying unified diff patch.",
            tool_call=ToolCall(
                tool_name="apply_patch",
                arguments={"patch": patch_diff},
            ),
            token_usage=TokenUsage(prompt_tokens=280, completion_tokens=120),
        ),
        LLMResponse(
            thought="Verifying tests pass with pytest.",
            tool_call=ToolCall(
                tool_name="run_shell",
                arguments={"command": "pytest tests/test_structures.py"},
            ),
            token_usage=TokenUsage(prompt_tokens=210, completion_tokens=50),
        ),
        LLMResponse(
            thought="All 3 tests pass.",
            final_answer="Fixed CaseInsensitiveDict key case and deletion.",
            token_usage=TokenUsage(prompt_tokens=180, completion_tokens=40),
        ),
    ]

    tracker = TokenTracker(
        model_name="gpt-4o",
        custom_rates=CostRates(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00),
    )
    llm = MockLLMClient(responses=mock_responses, token_tracker=tracker)
    runner = ReasoningLoop(llm_client=llm, dispatcher=dispatcher, max_steps=10)

    start_time = time.perf_counter()
    trajectory = await runner.run("Fix failing tests in tests/test_structures.py")
    duration = time.perf_counter() - start_time

    metrics = BugFixBenchmarkMetrics(
        task_name="urllib3_headers_case_insensitive_dict",
        passed=(trajectory.status == TrajectoryStatus.COMPLETED),
        step_count=len(trajectory.steps),
        tool_call_count=trajectory.tool_call_count,
        retry_count=0,
        prompt_tokens=tracker.cumulative_usage.prompt_tokens,
        completion_tokens=tracker.cumulative_usage.completion_tokens,
        total_tokens=tracker.cumulative_usage.total_tokens,
        cost_usd=tracker.cumulative_usage.estimated_cost_usd,
        wall_clock_seconds=round(duration, 3),
    )

    assert metrics.passed is True
    assert metrics.step_count == 5
    assert metrics.tool_call_count == 4
    assert metrics.total_tokens > 0
    assert metrics.cost_usd > 0.0
    assert metrics.wall_clock_seconds < 10.0

    updated_code = (src_dir / "structures.py").read_text(encoding="utf-8")
    assert "del self._store[key.lower()]" in updated_code
    assert "self._store[key.lower()] = (key, value)" in updated_code


@pytest.mark.asyncio
async def test_real_world_bugfix_task2_click_cli_parser(tmp_path: Path) -> None:
    """Evaluation Task 2 (pallets/click pattern):

    CLI boolean negative flag overrides tested with LangGraph state machine.
    """
    repo_dir = tmp_path / "click_cli_repo"
    src_dir = repo_dir / "src"
    tests_dir = repo_dir / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    buggy_parser = (
        "class OptionParser:\n"
        "    def __init__(self) -> None:\n"
        "        self.options: dict[str, bool] = {'cache': True, 'verbose': False}\n\n"
        "    def parse_args(self, args: list[str]) -> dict[str, bool]:\n"
        "        res = dict(self.options)\n"
        "        for arg in args:\n"
        "            if arg.startswith('--no-'):\n"
        "                # BUG: wrong slice drops leading char of option name\n"
        "                opt_name = arg[4:]  # should be arg[5:]\n"
        "                res[opt_name] = False\n"
        "            elif arg.startswith('--'):\n"
        "                opt_name = arg[2:]\n"
        "                res[opt_name] = True\n"
        "        return res\n"
    )
    (src_dir / "parser.py").write_text(buggy_parser, encoding="utf-8")

    test_parser_code = (
        "import pytest\n"
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
        "from src.parser import OptionParser\n\n"
        "def test_default_options():\n"
        "    parser = OptionParser()\n"
        "    assert parser.parse_args([]) == {'cache': True, 'verbose': False}\n\n"
        "def test_negative_flag_override():\n"
        "    parser = OptionParser()\n"
        "    res = parser.parse_args(['--no-cache', '--verbose'])\n"
        "    assert res['cache'] is False\n"
        "    assert res['verbose'] is True\n"
    )
    (tests_dir / "test_parser.py").write_text(test_parser_code, encoding="utf-8")

    patch_diff = (
        "--- a/src/parser.py\n"
        "+++ b/src/parser.py\n"
        "@@ -8,3 +8,3 @@\n"
        "             if arg.startswith('--no-'):\n"
        "-                # BUG: wrong slice drops leading char of option name\n"
        "-                opt_name = arg[4:]  # should be arg[5:]\n"
        "+                opt_name = arg[5:]\n"
        "                 res[opt_name] = False\n"
    )

    dispatcher = create_default_dispatcher(repo_dir)

    plan_json = (
        '{"task": "Fix OptionParser negative flag slicing", '
        '"rationale": "Run tests, patch parser.py, verify fix", '
        '"steps": ['
        '{"step_id": 1, "description": "Run tests", "expected_output": "failure"}, '
        '{"step_id": 2, "description": "Apply fix", "expected_output": "applied"}, '
        '{"step_id": 3, "description": "Verify tests", "expected_output": "pass"}'
        "]}"
    )

    mock_responses = [
        LLMResponse(
            raw_content=plan_json,
            thought="Created execution plan.",
            token_usage=TokenUsage(prompt_tokens=180, completion_tokens=80),
        ),
        LLMResponse(
            thought="Step 1: Running pytest.",
            tool_call=ToolCall(
                tool_name="run_shell",
                arguments={"command": "pytest tests/test_parser.py"},
            ),
            token_usage=TokenUsage(prompt_tokens=200, completion_tokens=40),
        ),
        LLMResponse(
            thought="Step 2: Applying fix to parser.py.",
            tool_call=ToolCall(
                tool_name="apply_patch",
                arguments={"patch": patch_diff},
            ),
            token_usage=TokenUsage(prompt_tokens=260, completion_tokens=100),
        ),
        LLMResponse(
            thought="Step 3: Verifying fix with pytest.",
            tool_call=ToolCall(
                tool_name="run_shell",
                arguments={"command": "pytest tests/test_parser.py"},
            ),
            token_usage=TokenUsage(prompt_tokens=210, completion_tokens=40),
        ),
        LLMResponse(
            thought="All unit tests pass.",
            final_answer="Fixed negative flag slicing offset in OptionParser.",
            token_usage=TokenUsage(prompt_tokens=150, completion_tokens=40),
        ),
    ]

    tracker = TokenTracker(model_name="gpt-4o")
    llm = MockLLMClient(responses=mock_responses, token_tracker=tracker)
    agent = LangGraphAgent(llm_client=llm, dispatcher=dispatcher)

    start_time = time.perf_counter()
    final_state = agent.run_task(
        task_id="click-parser-task",
        task="Fix negative flag bug in parser.py",
    )
    duration = time.perf_counter() - start_time

    metrics = BugFixBenchmarkMetrics(
        task_name="click_cli_negative_flag_parser",
        passed=(final_state.get("status") == AgentStatus.COMPLETED.value),
        step_count=len(final_state.get("tool_history", [])),
        tool_call_count=len(final_state.get("tool_history", [])),
        retry_count=final_state.get("retry_count", 0),
        prompt_tokens=tracker.cumulative_usage.prompt_tokens,
        completion_tokens=tracker.cumulative_usage.completion_tokens,
        total_tokens=tracker.cumulative_usage.total_tokens,
        cost_usd=tracker.cumulative_usage.estimated_cost_usd,
        wall_clock_seconds=round(duration, 3),
    )

    assert metrics.passed is True
    assert metrics.step_count == 3
    assert metrics.total_tokens > 0
    assert metrics.wall_clock_seconds < 10.0

    updated_parser = (src_dir / "parser.py").read_text(encoding="utf-8")
    assert "opt_name = arg[5:]" in updated_parser
