from __future__ import annotations

import time

from starlette.testclient import TestClient

from src.agent.dispatcher import create_default_dispatcher
from src.agent.llm_client import MockLLMClient
from src.agent.loop import ReasoningLoop
from src.core.config import ROOT_DIR
from src.main import app


def test_task_query_crud_lifecycle(client: TestClient) -> None:
    # 1. Test 404 for non-existent task
    resp_404 = client.get("/tasks/non-existent-uuid")
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["detail"].lower()

    # 2. Submit a new task
    submit_resp = client.post(
        "/tasks",
        json={"task": "Calculate total repository files", "metadata": {"env": "test"}},
    )
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    task_id = submit_data["task_id"]
    assert submit_data["status"] == "PENDING"

    # 3. Query the task immediately by ID
    get_resp = client.get(f"/tasks/{task_id}")
    assert get_resp.status_code == 200
    task_detail = get_resp.json()
    assert task_detail["task_id"] == task_id
    assert task_detail["task"] == "Calculate total repository files"
    assert task_detail["status"] in ["PENDING", "RUNNING", "COMPLETED"]
    assert task_detail["metadata"] == {"env": "test"}
    assert task_detail["created_at"] is not None

    # 4. List tasks and verify presence
    list_resp = client.get("/tasks")
    assert list_resp.status_code == 200
    tasks_list = list_resp.json()
    assert isinstance(tasks_list, list)
    assert any(t["task_id"] == task_id for t in tasks_list)


def test_task_queries_with_worker_execution() -> None:
    # Override reasoning loop to execute with mock client
    mock_llm = MockLLMClient(
        responses=[
            {
                "thought": "Computing result directly.",
                "final_answer": "Repository analysis completed.",
            }
        ]
    )
    dispatcher = create_default_dispatcher(ROOT_DIR)
    app.state.reasoning_loop_factory = lambda: ReasoningLoop(
        llm_client=mock_llm,
        dispatcher=dispatcher,
    )

    with TestClient(app) as test_client:
        submit_resp = test_client.post(
            "/tasks",
            json={"task": "Perform fast calculation", "metadata": {"test": "true"}},
        )
        assert submit_resp.status_code == 202
        task_id = submit_resp.json()["task_id"]

        # Poll until completed (worker processes asynchronously)
        completed_task = None
        for _ in range(50):
            resp = test_client.get(f"/tasks/{task_id}")
            assert resp.status_code == 200
            data = resp.json()
            if data["status"] == "COMPLETED":
                completed_task = data
                break
            time.sleep(0.05)

        assert completed_task is not None
        assert completed_task["status"] == "COMPLETED"
        assert completed_task["result"] == "Repository analysis completed."
        assert completed_task["started_at"] is not None
        assert completed_task["completed_at"] is not None

        # Query with status filter
        completed_list_resp = test_client.get("/tasks?status=COMPLETED")
        assert completed_list_resp.status_code == 200
        completed_list = completed_list_resp.json()
        assert any(t["task_id"] == task_id for t in completed_list)

        # Query with non-matching status filter
        pending_list_resp = test_client.get("/tasks?status=PENDING")
        assert pending_list_resp.status_code == 200
        pending_list = pending_list_resp.json()
        assert not any(t["task_id"] == task_id for t in pending_list)

        # Pagination test
        paginated_resp = test_client.get("/tasks?limit=1&offset=0")
        assert paginated_resp.status_code == 200
        paginated_list = paginated_resp.json()
        assert len(paginated_list) <= 1
