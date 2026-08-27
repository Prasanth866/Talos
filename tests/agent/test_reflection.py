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
from src.agent.reflection import (
    CircuitBreaker,
    calculate_backoff_delay,
    parse_pytest_output,
)


def test_test_result_parsing_extracts_correct_counts() -> None:
    """Unit test: parse_pytest_output extracts pass, fail, error counts."""
    pytest_sample = """
============================= test session starts ==============================
platform darwin -- Python 3.14.2
collected 10 items

tests/test_auth.py ..F..                                                [ 50%]
tests/test_api.py .E.s.                                                 [100%]

=================================== FAILURES ===================================
_________________________________ test_login ___________________________________
    def test_login():
>       assert login("wrong") is True
E       AssertionError: assert False is True

tests/test_auth.py:25: AssertionError
=========================== short test summary info ============================
FAILED tests/test_auth.py::test_login - AssertionError: assert False is True
ERROR tests/test_api.py::test_db_init - RuntimeError: DB connection failed
================== 1 failed, 7 passed, 1 skipped, 1 error in 1.42s =============
"""
    result = parse_pytest_output(pytest_sample)

    assert result.passed == 7
    assert result.failed == 1
    assert result.errors == 1
    assert result.skipped == 1
    assert result.total == 10
    assert not result.all_passed
    assert len(result.failure_details) >= 1
    assert any("test_login" in detail for detail in result.failure_details)


def test_test_result_parsing_all_passed() -> None:
    """Unit test: parse_pytest_output correctly marks fully passing suites."""
    clean_sample = """
============================= test session starts ==============================
collected 5 items

tests/test_models.py .....                                              [100%]
============================== 5 passed in 0.23s ===============================
"""
    result = parse_pytest_output(clean_sample)
    assert result.passed == 5
    assert result.failed == 0
    assert result.errors == 0
    assert result.total == 5
    assert result.all_passed


def test_exponential_backoff_delays_increase_correctly() -> None:
    """Unit test: calculate_backoff_delay calculates 2^retry progression."""

    assert calculate_backoff_delay(0, base=1.0, factor=2.0) == 1.0

    assert calculate_backoff_delay(1, base=1.0, factor=2.0) == 2.0

    assert calculate_backoff_delay(2, base=1.0, factor=2.0) == 4.0

    assert calculate_backoff_delay(3, base=1.0, factor=2.0) == 8.0

    assert calculate_backoff_delay(4, base=1.0, factor=2.0) == 16.0

    assert calculate_backoff_delay(10, base=1.0, factor=2.0, max_delay=30.0) == 30.0


def test_circuit_breaker_opens_after_3_consecutive_errors() -> None:
    """Unit test: CircuitBreaker opens after 3 consecutive failures."""
    cb = CircuitBreaker(failure_threshold=3, name="test_breaker")
    assert not cb.is_open

    tripped = cb.record_failure()
    assert not tripped
    assert cb.consecutive_failures == 1
    assert not cb.is_open

    tripped = cb.record_failure()
    assert not tripped
    assert cb.consecutive_failures == 2
    assert not cb.is_open

    tripped = cb.record_failure()
    assert tripped
    assert cb.consecutive_failures == 3
    assert cb.is_open

    cb.record_success()
    assert cb.consecutive_failures == 0
    assert not cb.is_open


def test_retry_count_reaches_max_then_transitions_to_failed() -> None:
    """Unit test: MAX_RETRIES transitions state to FAILED with failure report."""
    dispatcher = ToolDispatcher()
    attempt_count = 0

    def bug_tool() -> str:
        nonlocal attempt_count
        attempt_count += 1
        raise RuntimeError("Unfixable hardware/network failure")

    dispatcher.register_tool(
        name="bug_tool",
        description="Fails always",
        handler=bug_tool,
    )

    plan_json = json.dumps(
        {
            "task": "Fix unfixable bug",
            "rationale": "Attempt step repeatedly",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Run bug tool",
                    "expected_output": "Fixed",
                }
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Attempt 1",
            tool_call=ToolCall(tool_name="bug_tool", arguments={}),
        ),
        LLMResponse(
            thought="Attempt 2",
            tool_call=ToolCall(tool_name="bug_tool", arguments={}),
        ),
        LLMResponse(
            thought="Attempt 3",
            tool_call=ToolCall(tool_name="bug_tool", arguments={}),
        ),
    ]
    llm = MockLLMClient(responses=mock_responses)

    cb = CircuitBreaker(failure_threshold=5)
    agent = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
        circuit_breaker=cb,
        max_retries=3,
    )

    state = agent.run_task(task_id="unfixable-task", task="Fix unfixable bug")

    assert state["status"] == AgentStatus.FAILED.value
    assert state["retry_count"] >= 3
    assert "TASK FAILURE REPORT" in (state["error"] or "")
    assert "Unfixable hardware/network failure" in (state["error"] or "")


def test_circuit_breaker_trips_agent_execution() -> None:
    """Unit test: 3 consecutive errors trip circuit breaker with CircuitOpenError."""
    dispatcher = ToolDispatcher()

    def crashing_api() -> str:
        raise ConnectionError("Remote API downstream unreachable")

    dispatcher.register_tool(
        name="crashing_api",
        description="Downstream API",
        handler=crashing_api,
    )

    plan_json = json.dumps(
        {
            "task": "Call crashing API",
            "rationale": "Query API",
            "steps": [
                {"step_id": 1, "description": "Query", "expected_output": "data"}
            ],
        }
    )

    mock_responses = [
        LLMResponse(raw_content=plan_json, thought="Plan created"),
        LLMResponse(
            thought="Try 1",
            tool_call=ToolCall(tool_name="crashing_api", arguments={}),
        ),
        LLMResponse(
            thought="Try 2",
            tool_call=ToolCall(tool_name="crashing_api", arguments={}),
        ),
        LLMResponse(
            thought="Try 3",
            tool_call=ToolCall(tool_name="crashing_api", arguments={}),
        ),
    ]
    llm = MockLLMClient(responses=mock_responses)

    cb = CircuitBreaker(failure_threshold=3)
    agent = LangGraphAgent(
        llm_client=llm,
        dispatcher=dispatcher,
        circuit_breaker=cb,
        max_retries=10,
    )

    state = agent.run_task(task_id="circuit-task", task="Call crashing API")

    assert state["status"] == AgentStatus.FAILED.value
    assert "CircuitOpenError" in (state["error"] or "")
    assert "Circuit breaker tripped after 3 consecutive failures" in (
        state["error"] or ""
    )
    assert cb.is_open
