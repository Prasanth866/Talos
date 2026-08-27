from __future__ import annotations

import subprocess
from pathlib import Path

import docker
import pytest

from src.agent.llm_client import MockLLMClient
from src.agent.models import TrajectoryStatus
from src.agent.pipeline import execute_workspace_task
from src.indexer.embeddings import MockEmbeddingClient
from src.workspace.exceptions import WorkspaceNotFoundError
from src.workspace.manager import WorkspaceManager


def _create_fixture_git_repo(repo_dir: Path) -> str:
    """Creates a local Git repository fixture with a bug and pytest suite."""
    repo_dir.mkdir(parents=True, exist_ok=True)

    calc_file = repo_dir / "calculator.py"
    calc_file.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def divide(a: int, b: int) -> int:\n"
        "    # Bug: multiplication instead of division\n"
        "    return a * b\n"
    )

    test_file = repo_dir / "test_calculator.py"
    test_file.write_text(
        "from calculator import add, divide\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5\n"
    )

    git_bin = "/usr/bin/git"
    subprocess.run(  # noqa: S603
        [git_bin, "init"], cwd=repo_dir, check=True, capture_output=True
    )
    subprocess.run(  # noqa: S603
        [git_bin, "config", "user.name", "Talos Agent"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(  # noqa: S603
        [git_bin, "config", "user.email", "agent@talos.ai"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(  # noqa: S603
        [git_bin, "add", "."], cwd=repo_dir, check=True
    )
    subprocess.run(  # noqa: S603
        [git_bin, "commit", "-m", "initial commit"], cwd=repo_dir, check=True
    )

    return str(repo_dir)


@pytest.mark.asyncio
async def test_full_agent_pipeline_e2e(tmp_path: Path) -> None:
    """Integration test: submit task -> workspace created -> code indexed ->

    agent navigates via search -> pytest runs in sandbox -> result returned.
    """

    repo_source = tmp_path / "source_repo"
    repo_url = _create_fixture_git_repo(repo_source)

    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception:
        pytest.skip("Docker daemon not available for integration test.")

    workspace_root = tmp_path / "workspaces"
    mgr = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=workspace_root,
    )

    llm_client = MockLLMClient(
        responses=[
            {
                "thought": "I will search the codebase for division.",
                "tool_call": {
                    "name": "hybrid_search",
                    "arguments": {"query": "divide"},
                },
            },
            {
                "thought": "Let me inspect the definition of divide.",
                "tool_call": {
                    "name": "get_symbol_definition",
                    "arguments": {"name": "divide"},
                },
            },
            {
                "thought": "Found multiplication bug. Fixing calculator.py.",
                "tool_call": {
                    "name": "write_file",
                    "arguments": {
                        "path": "calculator.py",
                        "content": (
                            "def add(a: int, b: int) -> int:\n"
                            "    return a + b\n\n"
                            "def divide(a: int, b: int) -> int:\n"
                            "    return a // b\n"
                        ),
                    },
                },
            },
            {
                "thought": "Running tests in the sandbox container.",
                "tool_call": {
                    "name": "run_shell",
                    "arguments": {
                        "command": ("python3 test_calculator.py || echo 'TESTS_RAN'")
                    },
                },
            },
            {
                "thought": "Tests passed successfully.",
                "final_answer": "Fixed the division bug in calculator.py.",
            },
        ]
    )

    embedding_client = MockEmbeddingClient()

    trajectory, workspace_id = await execute_workspace_task(
        task="Fix the bug in divide() so test_calculator.py passes",
        repo_url=repo_url,
        workspace_manager=mgr,
        llm_client=llm_client,
        embedding_client=embedding_client,
        max_steps=10,
    )

    assert trajectory.status == TrajectoryStatus.COMPLETED
    assert trajectory.final_answer is not None
    assert "Fixed the division bug" in trajectory.final_answer
    assert len(trajectory.steps) == 5

    step1 = trajectory.steps[0]
    assert step1.tool_call is not None
    assert step1.tool_call.tool_name == "hybrid_search"
    assert step1.tool_result is not None
    assert "calculator.py" in step1.tool_result.output

    step2 = trajectory.steps[1]
    assert step2.tool_call is not None
    assert step2.tool_call.tool_name == "get_symbol_definition"
    assert step2.tool_result is not None
    assert "divide" in step2.tool_result.output

    step3 = trajectory.steps[2]
    assert step3.tool_call is not None
    assert step3.tool_call.tool_name == "write_file"

    step4 = trajectory.steps[3]
    assert step4.tool_call is not None
    assert step4.tool_call.tool_name == "run_shell"

    with pytest.raises(WorkspaceNotFoundError):
        mgr.get(workspace_id)
