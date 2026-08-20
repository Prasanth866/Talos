from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.api.routes.websocket import get_default_loop_factory
from src.core.config import Settings
from src.main import app


def test_health_endpoint(client: TestClient) -> None:
    """Verifies /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint(client: TestClient) -> None:
    """Verifies /readiness returns 200 with service status."""
    response = client.get("/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "version" in data
    assert "services" in data


def test_openapi_docs(client: TestClient) -> None:
    """Verifies /docs and /openapi.json are accessible."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    assert schema["info"]["title"] == "Talos Agent API"
    assert "/health" in schema["paths"]
    assert "/readiness" in schema["paths"]
    assert "/tasks" in schema["paths"]


def test_submit_task_success(client: TestClient) -> None:
    """Verifies POST /tasks accepts a task and returns 202 with task_id."""
    payload = {"task": "Refactor auth module", "metadata": {"user": "alice"}}
    response = client.post("/tasks", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["task_id"].startswith("task_")
    assert f"/ws?task_id={data['task_id']}" == data["ws_url"]


def test_submit_task_validation_error(client: TestClient) -> None:
    """Verifies POST /tasks fails with 422 on invalid task string."""
    response = client.post("/tasks", json={"task": ""})
    assert response.status_code == 422


def test_websocket_streaming_flow(tmp_path: Path) -> None:
    """Verifies full WebSocket event streaming with thoughts and tool calls."""
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Need to list directory contents first.",
                "tool_call": {
                    "tool_name": "list_dir",
                    "arguments": {"path": "."},
                },
            },
            {
                "thought": "Directory listed. Ready to conclude.",
                "final_answer": "Task completed successfully.",
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

    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/ws") as ws,
    ):
        ws.send_json({"task": "Inspect repository"})

        # Event 1: Thought from step 1
        msg1 = ws.receive_json()
        assert msg1["event_type"] == "thought"
        assert msg1["thought"] == "Need to list directory contents first."
        assert msg1["version"] == "v1"

        # Event 2: Tool Call from step 1
        msg2 = ws.receive_json()
        assert msg2["event_type"] == "tool_call"
        assert msg2["tool_name"] == "list_dir"
        assert msg2["version"] == "v1"

        # Event 3: Tool Output from step 1
        msg3 = ws.receive_json()
        assert msg3["event_type"] == "tool_output"
        assert msg3["tool_name"] == "list_dir"
        assert msg3["success"] is True

        # Event 4: Thought from step 2
        msg4 = ws.receive_json()
        assert msg4["event_type"] == "thought"
        assert msg4["thought"] == "Directory listed. Ready to conclude."

        # Event 5: Task Complete from step 2
        msg5 = ws.receive_json()
        assert msg5["event_type"] == "task_complete"
        assert msg5["final_answer"] == "Task completed successfully."
        assert msg5["total_steps"] == 2


def test_websocket_raw_text_message() -> None:
    """Verifies WebSocket accepts plain string message as task."""
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Processing raw text task",
                "final_answer": "Finished raw text task",
            }
        ]
    )
    app.state.reasoning_loop_factory = lambda: get_default_loop_factory(
        llm_client_override=mock_llm
    )

    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/ws") as ws,
    ):
        ws.send_text("Simple raw text task")
        msg1 = ws.receive_json()
        assert msg1["event_type"] == "thought"
        msg2 = ws.receive_json()
        assert msg2["event_type"] == "task_complete"


def test_websocket_invalid_message() -> None:
    """Verifies WebSocket returns ErrorEvent when message format is invalid."""
    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/ws") as ws,
    ):
        ws.send_json({"invalid_key": "no task"})
        msg = ws.receive_json()
        assert msg["event_type"] == "error"
        assert "Invalid message format" in msg["error"]


def test_websocket_disconnect_handling(tmp_path: Path) -> None:
    """Verifies server gracefully handles client disconnect mid-stream."""
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "First thought",
                "final_answer": "Done",
            }
        ]
    )
    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
        max_steps=5,
    )

    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/ws") as ws,
    ):
        ws.send_json({"task": "Quick task"})
        msg = ws.receive_json()
        assert msg["event_type"] == "thought"
        # Client disconnects while session is active
        ws.close()


def test_websocket_concurrent_connections(tmp_path: Path) -> None:
    """Verifies server handles multiple concurrent WebSocket connections."""
    dispatcher = create_default_dispatcher(tmp_path)

    def make_loop() -> ReasoningLoop:
        llm = MockLLMClient(
            responses=[
                {
                    "thought": "Thinking for client",
                    "final_answer": "Done for client",
                }
            ]
        )
        return ReasoningLoop(llm_client=llm, dispatcher=dispatcher, max_steps=5)

    app.state.reasoning_loop_factory = make_loop

    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/ws?task_id=1") as ws1,
        test_client.websocket_connect("/ws?task_id=2") as ws2,
    ):
        ws1.send_json({"task": "Task 1"})
        ws2.send_json({"task": "Task 2"})

        msg1_thought = ws1.receive_json()
        msg2_thought = ws2.receive_json()

        assert msg1_thought["event_type"] == "thought"
        assert msg2_thought["event_type"] == "thought"

        msg1_done = ws1.receive_json()
        msg2_done = ws2.receive_json()

        assert msg1_done["event_type"] == "task_complete"
        assert msg2_done["event_type"] == "task_complete"


def test_get_default_loop_factory_branches() -> None:
    """Verifies get_default_loop_factory branches with and without api_key."""
    # 1. With override
    mock_llm = MockLLMClient()
    loop1 = get_default_loop_factory(llm_client_override=mock_llm)
    assert loop1.llm_client is mock_llm

    # 2. With api key configured
    custom_settings = Settings(llm_api_key=SecretStr("sk-test-key-123"))
    with patch("src.api.routes.websocket.get_settings", return_value=custom_settings):
        loop2 = get_default_loop_factory()
        assert loop2.llm_client.__class__.__name__ == "HTTPLLMClient"

    # 3. Default without key
    no_key_settings = Settings(llm_api_key=SecretStr(""))
    with patch("src.api.routes.websocket.get_settings", return_value=no_key_settings):
        loop3 = get_default_loop_factory()
        assert isinstance(loop3.llm_client, MockLLMClient)


@pytest.mark.asyncio
async def test_reasoning_loop_on_event_callback_exception_handling(
    tmp_path: Path,
) -> None:
    """Verifies that an exception in on_event callback does not crash loop."""
    dispatcher = create_default_dispatcher(tmp_path)
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Thinking...",
                "final_answer": "Done",
            }
        ]
    )
    loop = ReasoningLoop(llm_client=mock_llm, dispatcher=dispatcher, max_steps=5)

    async def faulty_callback(_event: object) -> None:
        raise RuntimeError("Callback crashed")

    trajectory = await loop.run("Test task", on_event=faulty_callback)
    assert trajectory.final_answer == "Done"
