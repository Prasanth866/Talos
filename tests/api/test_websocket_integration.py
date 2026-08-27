from __future__ import annotations

import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.main import app


def test_e2e_submit_websocket_stream_and_db_record(tmp_path: Path) -> None:
    """Integration test: Submits a task, consumes all WebSocket events in real-time,
    and asserts the DB record matches the streamed final answer and metrics.
    """
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "First let's check directory structure.",
                "tool_call": {
                    "tool_name": "list_dir",
                    "arguments": {"path": "."},
                },
            },
            {
                "thought": "Directory inspected. Ready to finish.",
                "final_answer": "Repository analysis complete: everything is in order.",
            },
        ]
    )

    def custom_loop_factory() -> ReasoningLoop:
        return ReasoningLoop(
            llm_client=mock_llm,
            dispatcher=dispatcher,
            max_steps=5,
        )

    app.state.reasoning_loop_factory = custom_loop_factory

    with TestClient(app) as client:
        submit_res = client.post(
            "/tasks",
            json={
                "task": "Analyze repository and report files",
                "metadata": {"test_env": "integration"},
            },
        )
        assert submit_res.status_code == 202
        submit_data = submit_res.json()
        task_id = submit_data["task_id"]
        assert submit_data["status"] == "PENDING"
        assert submit_data["ws_url"] == f"/ws?task_id={task_id}"

        events: list[dict[str, object]] = []
        with client.websocket_connect(f"/ws?task_id={task_id}") as ws:
            while True:
                try:
                    msg = ws.receive_json()
                    events.append(msg)
                    if msg.get("event_type") in ("task_complete", "error"):
                        break
                except Exception:
                    break

        assert len(events) >= 5, f"Expected at least 5 events, got {len(events)}"

        assert events[0]["event_type"] == "thought"
        assert events[0]["thought"] == "First let's check directory structure."
        assert events[0]["step"] == 1
        assert events[0]["task_id"] == task_id
        assert events[0]["version"] == "v1"

        assert events[1]["event_type"] == "tool_call"
        assert events[1]["tool_name"] == "list_dir"
        assert events[1]["step"] == 1
        assert events[1]["task_id"] == task_id

        assert events[2]["event_type"] == "tool_output"
        assert events[2]["tool_name"] == "list_dir"
        assert events[2]["success"] is True
        assert events[2]["step"] == 1
        assert events[2]["task_id"] == task_id

        assert events[3]["event_type"] == "thought"
        assert events[3]["thought"] == "Directory inspected. Ready to finish."
        assert events[3]["step"] == 2
        assert events[3]["task_id"] == task_id

        assert events[4]["event_type"] == "task_complete"
        assert (
            events[4]["final_answer"]
            == "Repository analysis complete: everything is in order."
        )
        assert events[4]["total_steps"] == 2
        assert int(str(events[4]["total_tokens"])) > 0
        assert float(str(events[4]["total_cost_usd"])) > 0.0
        assert float(str(events[4]["duration_seconds"])) > 0.0

        db_task: dict[str, object] | None = None
        for _ in range(50):
            get_res = client.get(f"/tasks/{task_id}")
            assert get_res.status_code == 200
            data = get_res.json()
            if data.get("status") == "COMPLETED":
                db_task = data
                break
            time.sleep(0.05)

        assert db_task is not None
        assert db_task["task_id"] == task_id
        assert db_task["status"] == "COMPLETED"
        assert (
            db_task["result"] == "Repository analysis complete: everything is in order."
        )
        assert db_task["error"] is None
        assert int(str(db_task["prompt_tokens"])) > 0
        assert int(str(db_task["completion_tokens"])) > 0
        assert int(str(db_task["total_tokens"])) == int(str(events[4]["total_tokens"]))
        assert float(str(db_task["total_cost_usd"])) == pytest.approx(
            float(str(events[4]["total_cost_usd"])), rel=1e-5
        )
        assert float(str(db_task["duration_seconds"])) > 0.0
        assert db_task["started_at"] is not None
        assert db_task["completed_at"] is not None


def test_token_cost_persisted_matches_trajectory(tmp_path: Path) -> None:
    """Integration test: Verifies that token costs calculated during execution
    are accurately stored and persisted in the database upon completion.
    """
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Calculating answer with direct reasoning.",
                "final_answer": "Final computed result for token verification.",
            }
        ]
    )

    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
        max_steps=3,
    )

    with TestClient(app) as client:
        submit_res = client.post(
            "/tasks",
            json={"task": "Token cost calculation test"},
        )
        assert submit_res.status_code == 202
        task_id = submit_res.json()["task_id"]

        complete_event: dict[str, object] | None = None
        with client.websocket_connect(f"/ws?task_id={task_id}") as ws:
            while True:
                try:
                    msg = ws.receive_json()
                    if msg.get("event_type") == "task_complete":
                        complete_event = msg
                        break
                except Exception:
                    break

        assert complete_event is not None

        get_res = client.get(f"/tasks/{task_id}")
        assert get_res.status_code == 200
        db_task = get_res.json()

        assert db_task["status"] == "COMPLETED"
        assert int(str(db_task["prompt_tokens"])) > 0
        assert int(str(db_task["completion_tokens"])) > 0
        assert int(str(db_task["total_tokens"])) == int(
            str(complete_event["total_tokens"])
        )
        assert float(str(db_task["total_cost_usd"])) > 0.0
        assert float(str(db_task["total_cost_usd"])) == pytest.approx(
            float(str(complete_event["total_cost_usd"])), rel=1e-5
        )


def test_websocket_replays_events_on_late_connect(tmp_path: Path) -> None:
    """Integration test: Submits a task, waits for it to complete in the background,
    then connects to WebSocket and asserts all past events are replayed in order.
    """
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Thinking before late subscriber connects.",
                "final_answer": "Task concluded early.",
            }
        ]
    )

    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
    )

    with TestClient(app) as client:
        submit_res = client.post(
            "/tasks",
            json={"task": "Event replay test"},
        )
        assert submit_res.status_code == 202
        task_id = submit_res.json()["task_id"]

        for _ in range(50):
            res = client.get(f"/tasks/{task_id}")
            if res.json().get("status") == "COMPLETED":
                break
            time.sleep(0.05)

        replayed_events: list[dict[str, object]] = []
        with client.websocket_connect(f"/ws?task_id={task_id}") as ws:
            while True:
                try:
                    msg = ws.receive_json()
                    replayed_events.append(msg)
                    if msg.get("event_type") == "task_complete":
                        break
                except Exception:
                    break

        assert len(replayed_events) >= 2
        assert replayed_events[0]["event_type"] == "thought"
        assert (
            replayed_events[0]["thought"] == "Thinking before late subscriber connects."
        )
        assert replayed_events[1]["event_type"] == "task_complete"
        assert replayed_events[1]["final_answer"] == "Task concluded early."


def test_failed_task_streams_error_and_persists_failure(tmp_path: Path) -> None:
    """Integration test: Verifies that if reasoning loop encounters an unhandled error,
    an ErrorEvent is streamed over WebSocket and the DB record is marked as FAILED.
    """
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(responses=[RuntimeError("Simulated LLM network failure")])

    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
    )

    with TestClient(app) as client:
        submit_res = client.post(
            "/tasks",
            json={"task": "Failing task execution"},
        )
        assert submit_res.status_code == 202
        task_id = submit_res.json()["task_id"]

        error_event: dict[str, object] | None = None
        with client.websocket_connect(f"/ws?task_id={task_id}") as ws:
            while True:
                try:
                    msg = ws.receive_json()
                    if msg.get("event_type") == "error":
                        error_event = msg
                        break
                except Exception:
                    break

        assert error_event is not None
        assert "Simulated LLM network failure" in str(error_event["error"])

        db_task: dict[str, object] | None = None
        for _ in range(50):
            res = client.get(f"/tasks/{task_id}")
            data = res.json()
            if data.get("status") == "FAILED":
                db_task = data
                break
            time.sleep(0.05)

        assert db_task is not None
        assert db_task["status"] == "FAILED"
        assert "Simulated LLM network failure" in str(db_task["error"])


def test_frontend_static_files_served(client: TestClient) -> None:
    """Verifies that the frontend is served at GET /."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()
    assert "Talos" in response.text
