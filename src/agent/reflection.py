from __future__ import annotations

import re
from typing import Any

import structlog

from src.agent.models import (
    CircuitState,
    Plan,
    TestResult,
    ToolExecutionRecord,
)
from src.tools.exceptions import CircuitOpenError

logger = structlog.get_logger(__name__)


def parse_pytest_output(output: str) -> TestResult:
    """Parses raw pytest console output into a structured TestResult object."""
    cleaned = output.strip()
    if not cleaned:
        return TestResult(summary="No test output produced", all_passed=False)

    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    failure_details: list[str] = []

    # 1. Match standard pytest summary line: e.g. "=== 2 failed, 3 passed in 1.45s ==="
    # or "=== 5 passed in 0.12s ===" or "=== 1 error, 2 passed ==="
    summary_match = re.search(
        r"=+\s*(?:short test summary info|.*?)\s*([0-9\w\s,]+)\s+in\s+[\d\.]+s\s*=+",
        cleaned,
        re.IGNORECASE,
    )
    summary_text = summary_match.group(1) if summary_match else ""

    # Parse counts with regexes
    passed_m = re.search(r"(\d+)\s+passed", cleaned, re.IGNORECASE)
    if passed_m:
        passed = int(passed_m.group(1))

    failed_m = re.search(r"(\d+)\s+failed", cleaned, re.IGNORECASE)
    if failed_m:
        failed = int(failed_m.group(1))

    errors_m = re.search(r"(\d+)\s+error(?:s)?", cleaned, re.IGNORECASE)
    if errors_m:
        errors = int(errors_m.group(1))

    skipped_m = re.search(r"(\d+)\s+skipped", cleaned, re.IGNORECASE)
    if skipped_m:
        skipped = int(skipped_m.group(1))

    total = passed + failed + errors + skipped

    # 2. Extract failure summaries (FAILED test_path::test_name - Reason)
    for line in cleaned.splitlines():
        line_str = line.strip()
        if line_str.startswith("FAILED ") or " ERROR " in line_str:
            failure_details.append(line_str)

    all_passed = total > 0 and failed == 0 and errors == 0

    if not summary_text:
        if total > 0:
            summary_text = f"{passed} passed, {failed} failed, {errors} errors"
        else:
            summary_text = "No tests executed or unrecognized format"

    return TestResult(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        summary=summary_text.strip(),
        all_passed=all_passed,
        failure_details=failure_details,
    )


def calculate_backoff_delay(
    retry_count: int,
    base: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 60.0,
) -> float:
    """Calculates exponential backoff delay in seconds (base * factor^retry_count)."""
    if retry_count < 0:
        return 0.0
    delay = base * (factor**retry_count)
    return min(delay, max_delay)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures on repetitive errors."""

    def __init__(
        self,
        failure_threshold: int = 3,
        name: str = "agent_circuit_breaker",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.name = name
        self.consecutive_failures: int = 0
        self._state: CircuitState = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        """Returns the current operational state of the circuit breaker."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Returns True if the circuit breaker is open (tripped)."""
        return self._state == CircuitState.OPEN

    def record_success(self) -> None:
        """Records a successful operation, resetting failure counter."""
        if self.consecutive_failures > 0 or self._state != CircuitState.CLOSED:
            logger.info(
                "circuit_breaker.reset",
                name=self.name,
                previous_failures=self.consecutive_failures,
            )
        self.consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> bool:
        """Records a failure and trips the breaker if threshold reached.

        Returns True if the circuit breaker just tripped to OPEN.
        """
        self.consecutive_failures += 1
        logger.warning(
            "circuit_breaker.failure_recorded",
            name=self.name,
            consecutive_failures=self.consecutive_failures,
            threshold=self.failure_threshold,
        )
        if self.consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                "circuit_breaker.tripped",
                name=self.name,
                consecutive_failures=self.consecutive_failures,
            )
            return True
        return False

    def check_open(self) -> None:
        """Raises CircuitOpenError if the breaker is open."""
        if self.is_open:
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN after "
                f"{self.consecutive_failures} consecutive failures",
                tool_name=self.name,
                consecutive_failures=self.consecutive_failures,
            )


def generate_failure_report(
    task: str,
    plan: Plan | None,
    tool_history: list[ToolExecutionRecord],
    last_error: dict[str, Any] | None,
    retry_count: int,
    test_result: TestResult | None = None,
) -> str:
    """Generates a structured, human-readable diagnostic report for a failed task."""
    lines = [
        "=" * 60,
        "TASK FAILURE REPORT",
        "=" * 60,
        f"Task: {task}",
        f"Total Retries Attempted: {retry_count}",
        f"Total Steps Recorded: {len(tool_history)}",
    ]

    if plan and plan.steps:
        lines.append("\nPlan Execution State:")
        for step in plan.steps:
            lines.append(
                f"  [{step.step_id}] ({step.status.upper()}) {step.description}"
            )

    if last_error:
        lines.append("\nRoot Cause / Last Error:")
        lines.append(f"  Tool: {last_error.get('tool_name', 'unknown')}")
        lines.append(f"  Code: {last_error.get('code', 'TOOL_ERROR')}")
        lines.append(f"  Message: {last_error.get('message', 'No details')}")
        if last_error.get("details"):
            lines.append(f"  Details: {last_error['details']}")

    if test_result and test_result.total > 0:
        lines.append("\nTest Execution Diagnostics:")
        lines.append(
            f"  Summary: {test_result.passed} passed, {test_result.failed} failed, "
            f"{test_result.errors} errors (total {test_result.total})"
        )
        if test_result.failure_details:
            lines.append("  Failures:")
            for fail in test_result.failure_details[:5]:
                lines.append(f"    - {fail}")
            if len(test_result.failure_details) > 5:
                lines.append(
                    f"    ... [{len(test_result.failure_details) - 5} more failures]"
                )

    if tool_history:
        last_tool = tool_history[-1]
        lines.append("\nLatest Action Observation:")
        lines.append(f"  Tool: {last_tool.tool_name}({last_tool.arguments})")
        lines.append(f"  Success: {last_tool.success}")
        preview = (
            last_tool.output[:300] + "..."
            if len(last_tool.output) > 300
            else last_tool.output
        )
        lines.append(f"  Output:\n{preview}")

    lines.append("=" * 60)
    return "\n".join(lines)
