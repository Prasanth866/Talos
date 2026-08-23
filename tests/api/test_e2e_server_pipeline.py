from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.main import app


def test_full_pipeline_bugfix_api_ws_db(tmp_path: Path) -> None:
    """End-to-End Integration Test: Full Pipeline.

    POST task -> Queue -> Worker Pool -> Reasoning Loop -> WS Events -> DB Persistence.
    """
    sandbox_dir = tmp_path / "workspace"
    sandbox_dir.mkdir()

    # Create a buggy code file and test inside sandbox
    buggy_code = (
        "def add_tax(price: float, rate: float = 0.1) -> float:\n"
        "    # BUG: subtracting tax instead of adding\n"
        "    return price - (price * rate)\n"
    )
    test_code = (
        "from tax import add_tax\n\n"
        "def test_add_tax():\n"
        "    assert add_tax(100.0, 0.1) == 110.0\n"
    )
    (sandbox_dir / "tax.py").write_text(buggy_code, encoding="utf-8")
    (sandbox_dir / "test_tax.py").write_text(test_code, encoding="utf-8")

    dispatcher = create_default_dispatcher(sandbox_dir)

    fixed_code = (
        "def add_tax(price: float, rate: float = 0.1) -> float:\n"
        "    return price + (price * rate)\n"
    )

    mock_responses: list[dict[str, Any]] = [
        # Step 1: Run pytest
        {
            "thought": "Let's run pytest to observe the failing test.",
            "tool_call": {
                "tool_name": "run_shell",
                "arguments": {"command": "pytest test_tax.py"},
            },
        },
        # Step 2: Read tax.py
        {
            "thought": "Test failed. Reading tax.py to inspect the logic.",
            "tool_call": {
                "tool_name": "read_file",
                "arguments": {"path": "tax.py"},
            },
        },
        # Step 3: Write fix to tax.py
        {
            "thought": "Found bug: minus instead of plus. Writing fix.",
            "tool_call": {
                "tool_name": "write_file",
                "arguments": {"path": "tax.py", "content": fixed_code},
            },
        },
        # Step 4: Run pytest again to verify
        {
            "thought": "Verifying that pytest passes after the fix.",
            "tool_call": {
                "tool_name": "run_shell",
                "arguments": {"command": "pytest test_tax.py"},
            },
        },
        # Step 5: Final completion
        {
            "thought": "All unit tests pass. Task is complete.",
            "final_answer": "Fixed add_tax logic in tax.py and verified with pytest.",
        },
    ]

    mock_llm = MockLLMClient(responses=mock_responses, model_name="gpt-4o")
    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
        max_steps=10,
    )

    with TestClient(app) as client:
        # 1. POST task to API
        submit_res = client.post(
            "/tasks",
            json={
                "task": "Fix failing unit test in test_tax.py",
                "metadata": {"project": "tax-calculator", "env": "ci"},
            },
        )
        assert submit_res.status_code == 202
        submit_data = submit_res.json()
        task_id = submit_data["task_id"]
        assert submit_data["status"] == "PENDING"
        assert submit_data["ws_url"] == f"/ws?task_id={task_id}"

        # 2. Connect to WebSocket and stream events
        events: list[dict[str, Any]] = []
        with client.websocket_connect(f"/ws?task_id={task_id}") as ws:
            while True:
                try:
                    msg = ws.receive_json()
                    events.append(msg)
                    if msg.get("event_type") in ("task_complete", "error"):
                        break
                except Exception:
                    break

        # 3. Assert complete and ordered event stream
        assert len(events) >= 10, f"Expected full event trace, got {len(events)}"

        # Verify task_id correlation across all events
        for evt in events:
            assert evt["task_id"] == task_id
            assert evt["version"] == "v1"
            assert "timestamp" in evt

        event_types = [e["event_type"] for e in events]
        assert "thought" in event_types
        assert "tool_call" in event_types
        assert "tool_output" in event_types
        assert event_types[-1] == "task_complete"

        complete_evt = events[-1]
        assert "Fixed add_tax logic" in complete_evt["final_answer"]
        assert complete_evt["total_steps"] == 5
        assert complete_evt["total_tokens"] > 0
        assert complete_evt["total_cost_usd"] > 0.0

        # 4. Assert Database persistence
        db_task: dict[str, Any] | None = None
        for _ in range(50):
            res = client.get(f"/tasks/{task_id}")
            assert res.status_code == 200
            data = res.json()
            if data.get("status") == "COMPLETED":
                db_task = data
                break
            time.sleep(0.05)

        assert db_task is not None
        assert db_task["task_id"] == task_id
        assert db_task["status"] == "COMPLETED"
        assert "Fixed add_tax logic" in db_task["result"]
        assert db_task["metadata"] == {"project": "tax-calculator", "env": "ci"}
        assert db_task["prompt_tokens"] > 0
        assert db_task["completion_tokens"] > 0
        assert db_task["total_tokens"] == complete_evt["total_tokens"]
        assert db_task["total_cost_usd"] == pytest.approx(
            complete_evt["total_cost_usd"], rel=1e-5
        )
        assert db_task["started_at"] is not None
        assert db_task["completed_at"] is not None

        # 5. Verify the disk file was updated by the agent
        updated_file = (sandbox_dir / "tax.py").read_text(encoding="utf-8")
        assert "return price + (price * rate)" in updated_file


def test_token_costs_stored_in_db_per_task(tmp_path: Path) -> None:
    """Verifies that token costs and counts are accurately stored per task."""
    dispatcher = create_default_dispatcher(tmp_path)

    # Dynamic loop factory based on incoming task
    def dynamic_loop_factory() -> ReasoningLoop:
        mock_llm = MockLLMClient(
            responses=[
                {
                    "thought": "Inspecting directory",
                    "tool_call": {"tool_name": "list_dir", "arguments": {"path": "."}},
                },
                {"thought": "Finished inspection", "final_answer": "Analysis complete"},
            ],
            model_name="gpt-4o",
        )
        return ReasoningLoop(
            llm_client=mock_llm,
            dispatcher=dispatcher,
            max_steps=5,
        )

    app.state.reasoning_loop_factory = dynamic_loop_factory

    with TestClient(app) as client:
        # Submit Task 1
        res1 = client.post("/tasks", json={"task": "Inspect repository files"})
        assert res1.status_code == 202
        id1 = res1.json()["task_id"]

        # Wait for task to complete
        task_data = None
        for _ in range(50):
            r = client.get(f"/tasks/{id1}").json()
            if r.get("status") == "COMPLETED":
                task_data = r
                break
            time.sleep(0.05)

        assert task_data is not None
        assert task_data["task_id"] == id1
        assert task_data["status"] == "COMPLETED"
        assert task_data["prompt_tokens"] > 0
        assert task_data["completion_tokens"] > 0
        assert task_data["total_tokens"] > 0
        assert task_data["total_cost_usd"] > 0.0
        assert task_data["duration_seconds"] > 0.0
        assert task_data["started_at"] is not None
        assert task_data["completed_at"] is not None
