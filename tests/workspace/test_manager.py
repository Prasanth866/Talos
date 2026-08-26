from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import docker.errors
import git.exc
import pytest

from src.workspace import (
    CommandOutputLine,
    ContainerSecurityConfig,
    DockerDaemonError,
    GitCloneError,
    SentinelType,
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


@pytest.mark.asyncio
async def test_execute_command_streams_output(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: command output is streamed as CommandOutputLine structs."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    mock_docker_client.api.exec_create.return_value = {"Id": "exec_abc123"}
    mock_docker_client.api.exec_start.return_value = iter(
        [
            (b"first line\nsecond line\n", None),
            (None, b"stderr error line\n"),
            (b"third line\n", None),
        ]
    )

    lines: list[CommandOutputLine] = []
    async for item in manager.execute_command(ws.workspace_id, "echo test"):
        lines.append(item)

    assert len(lines) == 4
    assert lines[0] == CommandOutputLine(
        line="first line",
        stream="stdout",
        is_sentinel=False,
        sentinel_type=None,
    )
    assert lines[1] == CommandOutputLine(
        line="second line",
        stream="stdout",
        is_sentinel=False,
        sentinel_type=None,
    )
    assert lines[2] == CommandOutputLine(
        line="stderr error line",
        stream="stderr",
        is_sentinel=False,
        sentinel_type=None,
    )
    assert lines[3] == CommandOutputLine(
        line="third line",
        stream="stdout",
        is_sentinel=False,
        sentinel_type=None,
    )
    assert all(not line.is_sentinel for line in lines)


@pytest.mark.asyncio
async def test_execute_command_output_capping_triggers_truncated_sentinel(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: output exceeding max_output_bytes triggers TRUNCATED sentinel."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    mock_docker_client.api.exec_start.return_value = iter(
        [
            (b"A" * 60 + b"\n", None),
            (b"B" * 60 + b"\n", None),
            (b"C" * 60 + b"\n", None),
        ]
    )

    lines: list[CommandOutputLine] = []
    async for item in manager.execute_command(
        ws.workspace_id, "generate_large_output", max_output_bytes=100
    ):
        lines.append(item)

    assert len(lines) >= 1
    last_line = lines[-1]
    assert last_line.is_sentinel is True
    assert last_line.sentinel_type == SentinelType.TRUNCATED
    assert last_line.line == "[TRUNCATED]"


@pytest.mark.asyncio
async def test_execute_command_timeout_triggers_timeout_sentinel(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: command exceeding timeout triggers TIMEOUT sentinel."""
    import time

    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    mock_docker_client.api.exec_create.return_value = {"Id": "exec_abc123"}

    def slow_stream() -> Any:
        yield (b"initial output\n", None)
        for _ in range(25):
            time.sleep(0.01)
        yield (b"should not arrive\n", None)

    mock_docker_client.api.exec_start.return_value = slow_stream()

    lines: list[CommandOutputLine] = []
    async for item in manager.execute_command(
        ws.workspace_id, "sleep 10", timeout_s=0.08
    ):
        lines.append(item)

    assert len(lines) == 2
    assert lines[0].line == "initial output"
    assert lines[1].is_sentinel is True
    assert lines[1].sentinel_type == SentinelType.TIMEOUT
    assert lines[1].line == "[TIMEOUT]"


def test_container_hardening_default_security_config_applied(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: verifies default hardening parameters are passed to containers.run."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["mem_limit"] == "512m"
    assert call_kwargs["cpu_quota"] == 100000
    assert call_kwargs["cpu_period"] == 100000
    assert call_kwargs["pids_limit"] == 256
    assert call_kwargs["read_only"] is True
    assert call_kwargs["tmpfs"] == {
        "/tmp": "rw,noexec,nosuid,size=64m",  # noqa: S108
    }

    assert call_kwargs["network_mode"] == "none"
    assert call_kwargs["user"] == "1000:1000"

    assert ws.security_config.mem_limit == "512m"
    assert ws.security_config.pids_limit == 256
    assert ws.security_config.read_only is True
    assert ws.security_config.network_mode == "none"
    assert ws.security_config.user == "1000:1000"


def test_container_hardening_custom_security_config_override(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: verifies custom SecurityConfig override on create()."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    custom_cfg = ContainerSecurityConfig(
        mem_limit="1024m",
        cpu_quota=200000,
        cpu_period=100000,
        pids_limit=512,
        read_only=False,
        network_mode="bridge",
        user="2000:2000",
        tmpfs={"/tmp": "size=128m"},  # noqa: S108
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create(
            "https://github.com/example/repo.git",
            security_config=custom_cfg,
        )

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["mem_limit"] == "1024m"
    assert call_kwargs["cpu_quota"] == 200000
    assert call_kwargs["pids_limit"] == 512
    assert call_kwargs["read_only"] is False
    assert call_kwargs["network_mode"] == "bridge"
    assert call_kwargs["user"] == "2000:2000"
    assert call_kwargs["tmpfs"] == {"/tmp": "size=128m"}  # noqa: S108

    assert ws.security_config == custom_cfg


def test_container_hardening_mem_limit(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: container respects mem_limit configuration."""
    custom_cfg = ContainerSecurityConfig(mem_limit="256m")
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
        default_security_config=custom_cfg,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["mem_limit"] == "256m"
    assert ws.security_config.mem_limit == "256m"


def test_container_hardening_pids_limit(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: container respects pids_limit configuration."""
    custom_cfg = ContainerSecurityConfig(pids_limit=128)
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create(
            "https://github.com/example/repo.git",
            security_config=custom_cfg,
        )

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["pids_limit"] == 128
    assert ws.security_config.pids_limit == 128


def test_container_hardening_read_only_fs(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: filesystem is read-only outside /workspace."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["read_only"] is True
    assert call_kwargs["volumes"][str(ws.host_dir)]["bind"] == "/workspace"
    assert call_kwargs["volumes"][str(ws.host_dir)]["mode"] == "rw"
    assert call_kwargs["tmpfs"] == {
        "/tmp": "rw,noexec,nosuid,size=64m",  # noqa: S108
    }


def test_container_hardening_network_isolation(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: network egress is disabled via network_mode none."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["network_mode"] == "none"
    assert ws.security_config.network_mode == "none"


def test_container_hardening_non_root_user(
    tmp_path: Path, mock_docker_client: MagicMock
) -> None:
    """Unit test: non-root user execution configured."""
    manager = WorkspaceManager(
        docker_client=mock_docker_client,
        workspace_root=tmp_path,
    )

    with patch("src.workspace.manager.shallow_clone", return_value="sha123"):
        ws = manager.create("https://github.com/example/repo.git")

    call_kwargs = mock_docker_client.containers.run.call_args.kwargs
    assert call_kwargs["user"] == "1000:1000"
    assert ws.security_config.user == "1000:1000"
