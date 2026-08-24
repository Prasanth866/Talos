from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import docker.errors
import git.exc
import pytest

from src.workspace import (
    DockerDaemonError,
    GitCloneError,
    Workspace,
    WorkspaceCreationError,
    WorkspaceDestroyError,
    WorkspaceError,
    WorkspaceExecutionError,
    WorkspaceManager,
    WorkspaceNotFoundError,
    WorkspaceStatus,
)


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Fixture providing a mocked DockerClient."""
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "container_abc123"
    mock_container.status = "running"
    client.containers.run.return_value = mock_container
    client.containers.get.return_value = mock_container
    client.ping.return_value = True
    return client


def test_workspace_error_to_dict() -> None:
    """Verifies WorkspaceError provides structured dictionary representation."""
    err = WorkspaceNotFoundError(
        message="Workspace missing",
        workspace_id="ws-123",
        details={"extra": "data"},
    )
    d = err.to_dict()
    assert d["error"] == "WorkspaceNotFoundError"
    assert d["message"] == "Workspace missing"
    assert d["details"]["workspace_id"] == "ws-123"
    assert d["details"]["extra"] == "data"


def test_create_produces_running_workspace_with_cloned_repo(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: create() produces a running container with cloned repo."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch(
        "src.workspace.manager.shallow_clone", return_value="abcdef1234567890"
    ) as mock_clone:
        ws = manager.create(repo_url="https://github.com/example/repo.git")

        assert isinstance(ws, Workspace)
        assert ws.status == WorkspaceStatus.RUNNING
        assert ws.container_id == "container_abc123"
        assert ws.container_name.startswith("talos-ws-")
        assert ws.commit_sha == "abcdef1234567890"
        assert ws.host_dir.exists()
        assert ws.workspace_id in manager._workspaces

        mock_clone.assert_called_once()
        mock_docker_client.containers.run.assert_called_once()


def test_destroy_removes_all_container_artifacts(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: destroy() removes all container artifacts."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create(repo_url="https://github.com/example/repo.git")
        host_dir = ws.host_dir
        ws_id = ws.workspace_id

        assert host_dir.exists()
        assert ws_id in manager._workspaces

        manager.destroy(ws_id)

        mock_container = mock_docker_client.containers.get.return_value
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)

        assert not host_dir.exists()
        assert ws_id not in manager._workspaces
        assert ws.status == WorkspaceStatus.DESTROYED


def test_get_workspace_and_not_found(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Verifies get() returns active workspace or raises WorkspaceNotFoundError."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")
        fetched = manager.get(ws.workspace_id)
        assert fetched.workspace_id == ws.workspace_id

    with pytest.raises(WorkspaceNotFoundError) as exc_info:
        manager.get("non-existent-id")
    assert exc_info.value.workspace_id == "non-existent-id"
    assert isinstance(exc_info.value, WorkspaceError)


def test_docker_errors_wrapped_as_workspace_error(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: Docker SDK errors are wrapped as WorkspaceError."""
    mock_docker_client.containers.run.side_effect = docker.errors.APIError(
        "Docker API failure"
    )
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with (
        patch("src.workspace.manager.shallow_clone", return_value="sha123"),
        pytest.raises(WorkspaceCreationError) as exc_info,
    ):
        manager.create(repo_url="https://github.com/example/repo.git")

    assert isinstance(exc_info.value, WorkspaceError)
    assert "Failed to create workspace container" in str(exc_info.value)


def test_docker_daemon_unavailable_raises_docker_daemon_error() -> None:
    """Verifies DockerDaemonError is raised when docker daemon is unreachable."""
    with (
        patch(
            "docker.from_env",
            side_effect=docker.errors.DockerException("Daemon down"),
        ),
        pytest.raises(DockerDaemonError) as exc_info,
    ):
        WorkspaceManager(docker_client=None)

    assert isinstance(exc_info.value, WorkspaceError)
    assert "Docker daemon is unavailable" in str(exc_info.value)


def test_git_clone_failure_wrapped_as_git_clone_error(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Verifies Git failures raise typed GitCloneError (subclass of WorkspaceError)."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with (
        patch(
            "git.Repo.clone_from",
            side_effect=git.exc.GitCommandError("clone", 128, b"", b"Repo not found"),
        ),
        pytest.raises(GitCloneError) as exc_info,
    ):
        manager.create(repo_url="https://github.com/invalid/non-existent.git")

    assert isinstance(exc_info.value, WorkspaceError)
    assert exc_info.value.repo_url == "https://github.com/invalid/non-existent.git"


def test_destroy_error_handling(tmp_path: Path, mock_docker_client: MagicMock) -> None:
    """Verifies destroy handles docker API errors cleanly with WorkspaceDestroyError."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    mock_container = mock_docker_client.containers.get.return_value
    mock_container.stop.side_effect = docker.errors.APIError("Cannot stop container")

    with pytest.raises(WorkspaceDestroyError) as exc_info:
        manager.destroy(ws.workspace_id)

    assert isinstance(exc_info.value, WorkspaceError)
    assert exc_info.value.workspace_id == ws.workspace_id


def test_destroy_all_cleans_multiple_workspaces(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Verifies destroy_all tears down all registered workspaces."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        manager.create("https://github.com/example/repo1.git")
        manager.create("https://github.com/example/repo2.git")

    assert len(manager._workspaces) == 2
    manager.destroy_all()
    assert len(manager._workspaces) == 0


def test_run_command_success_and_failure(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Verifies run_command executes inside container and wraps execution failures."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    mock_container = mock_docker_client.containers.get.return_value

    mock_exec_success = MagicMock()
    mock_exec_success.exit_code = 0
    mock_exec_success.output = b"hello from container\n"
    mock_container.exec_run.return_value = mock_exec_success

    res = manager.run_command(ws.workspace_id, "echo 'hello'")
    assert res["exit_code"] == 0
    assert res["stdout"] == "hello from container"

    mock_exec_failure = MagicMock()
    mock_exec_failure.exit_code = 1
    mock_exec_failure.output = b"command error"
    mock_container.exec_run.return_value = mock_exec_failure

    with pytest.raises(WorkspaceExecutionError) as exc_info:
        manager.run_command(ws.workspace_id, "false")

    assert exc_info.value.exit_code == 1
    assert "failed with exit code 1" in str(exc_info.value)
