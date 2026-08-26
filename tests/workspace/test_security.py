from __future__ import annotations

import time
from pathlib import Path

import docker
import git
import pytest

from src.agent.dispatcher import create_default_dispatcher
from src.agent.models import ToolCall
from src.tools.exceptions import PathTraversalError, ToolError
from src.tools.filesystem import FileSystemTool
from src.workspace import (
    ContainerSecurityConfig,
    WorkspaceManager,
)


@pytest.fixture
def hardened_workspace(tmp_path: Path) -> tuple[WorkspaceManager, str]:
    """Sets up a live hardened workspace for adversarial security testing."""
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception:
        pytest.skip("Docker daemon not available for live security tests.")

    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    (source_repo_dir / "README.md").write_text(
        "# Security Test Repo\n", encoding="utf-8"
    )
    repo = git.Repo.init(source_repo_dir)
    repo.index.add(["README.md"])
    repo.index.commit("Initial security commit")

    workspaces_root = tmp_path / "workspaces"
    workspaces_root.mkdir()

    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=workspaces_root,
        default_security_config=ContainerSecurityConfig(),
    )

    workspace = manager.create(repo_url=f"file://{source_repo_dir}")
    return manager, workspace.workspace_id


@pytest.mark.integration
def test_security_network_escape_fails(
    hardened_workspace: tuple[WorkspaceManager, str],
) -> None:
    """Security test: network escape (curl/socket to external IP) must fail."""
    manager, ws_id = hardened_workspace

    try:
        # 1. Test curl outbound egress to 8.8.8.8
        curl_cmd = "curl -m 2 https://8.8.8.8 2>&1 || true"
        curl_res = manager.run_command(ws_id, ["/bin/sh", "-c", curl_cmd])
        print(f"\n[SECURITY TEST] Curl output: {curl_res['stdout']}")

        # 2. Test raw Python socket connection to 8.8.8.8:53
        socket_cmd = (
            'python3 -c "import socket; s = socket.socket(); s.settimeout(2); '
            "s.connect(('8.8.8.8', 53))\" 2>&1 || true"
        )
        socket_res = manager.run_command(ws_id, ["/bin/sh", "-c", socket_cmd])
        print(f"[SECURITY TEST] Socket output: {socket_res['stdout']}")

        output_text = f"{curl_res['stdout']!s} {socket_res['stdout']!s}".lower()
        assert (
            "network is unreachable" in output_text
            or "errno 101" in output_text
            or "curl: (6)" in output_text
            or "curl: (7)" in output_text
            or "timed out" in output_text
            or "timeout" in output_text
        )

    finally:
        manager.destroy(ws_id)


@pytest.mark.integration
def test_security_filesystem_escape_fails(
    hardened_workspace: tuple[WorkspaceManager, str],
) -> None:
    """Security test: filesystem escape must fail with PermissionError/EROFS."""
    manager, ws_id = hardened_workspace

    try:
        # Attempt to write persistent backdoor into /etc/crontab
        fs_cmd = "python3 -c \"open('/etc/crontab', 'w').write('evil')\" 2>&1 || true"
        fs_res = manager.run_command(ws_id, ["/bin/sh", "-c", fs_cmd])
        print(f"\n[SECURITY TEST] /etc/crontab write output: {fs_res['stdout']}")

        # Attempt to modify root binary directory /bin/sh
        bin_cmd = "touch /bin/evil_binary 2>&1 || true"
        bin_res = manager.run_command(ws_id, ["/bin/sh", "-c", bin_cmd])
        print(f"[SECURITY TEST] /bin write output: {bin_res['stdout']}")

        output_text = str(fs_res["stdout"]) + " " + str(bin_res["stdout"])
        assert (
            "Read-only file system" in output_text
            or "PermissionError" in output_text
            or "Permission denied" in output_text
            or "Errno 30" in output_text
        )

    finally:
        manager.destroy(ws_id)


@pytest.mark.integration
def test_security_fork_bomb_killed_within_5_seconds(
    hardened_workspace: tuple[WorkspaceManager, str],
) -> None:
    """Security test: fork bomb must be killed/blocked by pids_limit within 5s."""
    manager, ws_id = hardened_workspace

    try:
        # Spawn 2^10 = 1024 processes inside container
        fork_cmd = (
            'python3 -c "import os, time; [os.fork() for _ in range(10)]" 2>&1 || true'
        )
        start_time = time.perf_counter()
        fork_res = manager.run_command(ws_id, ["/bin/sh", "-c", fork_cmd])
        elapsed_seconds = time.perf_counter() - start_time

        print(f"\n[SECURITY TEST] Fork bomb execution time: {elapsed_seconds:.4f}s")
        print(f"[SECURITY TEST] Fork bomb output: {str(fork_res['stdout'])[:150]}")

        # Must be blocked and return in well under 5 seconds (typically < 100ms)
        assert elapsed_seconds < 5.0
        assert (
            "BlockingIOError" in str(fork_res["stdout"])
            or "Resource temporarily unavailable" in str(fork_res["stdout"])
            or "Errno 11" in str(fork_res["stdout"])
        )

        # Ensure container and host are responsive and unharmed once reaper cleans up
        ping_res = None
        for _ in range(10):
            try:
                ping_res = manager.run_command(ws_id, "echo 'host_healthy'")
                if ping_res["stdout"] == "host_healthy":
                    break
            except Exception:
                time.sleep(0.5)

        assert ping_res is not None and ping_res["stdout"] == "host_healthy"

    finally:
        manager.destroy(ws_id)


@pytest.mark.integration
def test_security_memory_exhaustion_contained_to_container(
    hardened_workspace: tuple[WorkspaceManager, str],
) -> None:
    """Security test: memory exhaustion is contained and OOM-killed in container."""
    manager, ws_id = hardened_workspace

    try:
        # Attempt to allocate 1GB in a 512MB capped container
        mem_cmd = "python3 -c \"a = b'x' * (1024 * 1024 * 1024)\" 2>&1 || true"
        mem_res = manager.run_command(ws_id, ["/bin/sh", "-c", mem_cmd])
        print(f"\n[SECURITY TEST] 1GB Memory allocation output: {mem_res['stdout']}")

        assert (
            "Killed" in str(mem_res["stdout"])
            or "MemoryError" in str(mem_res["stdout"])
            or str(mem_res["stdout"]) == ""
        )

        # Verify container survives OOM-kill of child task and remains operational
        healthy_res = manager.run_command(ws_id, "echo 'container_alive'")
        assert healthy_res["stdout"] == "container_alive"

    finally:
        manager.destroy(ws_id)


@pytest.mark.asyncio
async def test_security_path_traversal_returns_tool_error(
    tmp_path: Path,
) -> None:
    """Security test: path traversal (../../etc/passwd) must return ToolError."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "valid_file.txt").write_text("safe content", encoding="utf-8")

    # 1. Direct FileSystemTool path traversal check
    fs_tool = FileSystemTool(sandbox_dir=sandbox_dir)
    with pytest.raises(PathTraversalError) as exc_info:
        fs_tool.read_file("../../etc/passwd")

    assert isinstance(exc_info.value, ToolError)
    assert "Access denied" in str(exc_info.value)
    assert exc_info.value.tool_name == "FileSystemTool"

    # Direct write path traversal check
    with pytest.raises(PathTraversalError):
        fs_tool.write_file("../../etc/shadow", "malicious_payload")

    # 2. Agent ToolDispatcher path traversal check
    dispatcher = create_default_dispatcher(sandbox_dir)
    traversal_call = ToolCall(
        tool_name="read_file",
        arguments={"path": "../../../etc/passwd"},
    )
    result = await dispatcher.execute_tool(traversal_call)

    print(f"\n[SECURITY TEST] Dispatcher traversal result: {result}")
    assert result.success is False
    assert result.error is not None
    assert "Access denied" in result.error or "PathTraversalError" in result.error
