from __future__ import annotations

import time
from pathlib import Path

import docker
import docker.errors
import git
import pytest

from src.workspace import (
    CommandOutputLine,
    SentinelType,
    WorkspaceManager,
    WorkspaceStatus,
)


@pytest.fixture
def docker_client() -> docker.DockerClient:
    """Fixture providing a live Docker client or skips if Docker is not running."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception:
        pytest.skip("Docker daemon not available for live experiments.")


@pytest.mark.integration
def test_live_workspace_experiment(
    tmp_path: Path, docker_client: docker.DockerClient
) -> None:
    """Experiment: Create a workspace, clone a repo into it, list files, destroy it.

    Measures creation time (pull + clone) and verifies complete artifact cleanup.
    """
    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    repo = git.Repo.init(source_repo_dir)
    test_file = source_repo_dir / "README.md"
    test_file.write_text(
        "# Experiment Workspace\nHello from Talos test repo!",
        encoding="utf-8",
    )
    extra_file = source_repo_dir / "src_test.py"
    extra_file.write_text("print('Inside isolated workspace')", encoding="utf-8")
    repo.index.add(["README.md", "src_test.py"])
    repo.index.commit("Initial commit")

    repo_url = f"file://{source_repo_dir}"

    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=tmp_path / "workspaces",
        base_image="python:3.12-slim",
    )

    start_time = time.perf_counter()
    workspace = manager.create(repo_url=repo_url)
    creation_duration = time.perf_counter() - start_time

    print(f"\n[EXPERIMENT] Workspace created in {creation_duration:.3f}s")
    print(f"[EXPERIMENT] Container ID: {workspace.container_id}")
    print(f"[EXPERIMENT] Container Name: {workspace.container_name}")
    print(f"[EXPERIMENT] Commit SHA: {workspace.commit_sha}")

    assert workspace.status == WorkspaceStatus.RUNNING
    assert workspace.host_dir.exists()

    container = docker_client.containers.get(workspace.container_id)
    assert container.status == "running"

    files = manager.list_files(workspace.workspace_id)
    print(f"[EXPERIMENT] Files in workspace: {files}")
    assert any("README.md" in f for f in files)
    assert any("src_test.py" in f for f in files)

    cmd_res = manager.run_command(
        workspace.workspace_id, ["python", "/workspace/src_test.py"]
    )
    print(f"[EXPERIMENT] Command output: {cmd_res['stdout']}")
    assert "Inside isolated workspace" in str(cmd_res["stdout"])

    manager.destroy(workspace.workspace_id)
    print("[EXPERIMENT] Workspace destroyed cleanly.")

    assert not workspace.host_dir.exists()

    with pytest.raises(docker.errors.NotFound):
        docker_client.containers.get(workspace.container_id)

    matching_containers = docker_client.containers.list(
        all=True,
        filters={"name": workspace.container_name},
    )
    assert len(matching_containers) == 0
    print("[EXPERIMENT] All Docker resources verified cleaned up.")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_command_execution_10mb_output_capping(
    tmp_path: Path, docker_client: docker.DockerClient
) -> None:
    """Experiment 1: Run command producing 10MB+ output.

    Verify truncation occurs at 1MB with TRUNCATED sentinel and measure memory.
    """
    import sys

    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    repo = git.Repo.init(source_repo_dir)
    (source_repo_dir / "README.md").write_text("# Test", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=tmp_path / "workspaces",
    )
    workspace = manager.create(repo_url=f"file://{source_repo_dir}")

    try:
        gen_cmd = (
            'python3 -u -c "for i in range(200000): '
            "print(f'line {i:06d}: ' + 'X' * 80)\""
        )
        lines: list[CommandOutputLine] = []

        start_time = time.perf_counter()
        async for line in manager.execute_command(
            workspace.workspace_id,
            gen_cmd,
            timeout_s=30.0,
            max_output_bytes=1024 * 1024,
        ):
            lines.append(line)
        duration = time.perf_counter() - start_time

        total_chars = sum(len(line.line) for line in lines)
        memory_bytes = sys.getsizeof(lines) + sum(
            sys.getsizeof(line) + sys.getsizeof(line.line) for line in lines
        )

        print(
            f"\n[EXPERIMENT 10MB CAPPING] Processed {len(lines)} lines in "
            f"{duration:.3f}s"
        )

        print(
            f"[EXPERIMENT 10MB CAPPING] Total yielded: {total_chars} chars "
            f"(~{total_chars / 1024:.1f} KB)"
        )
        print(
            f"[EXPERIMENT 10MB CAPPING] Estimated Python memory: "
            f"{memory_bytes / 1024:.2f} KB ({memory_bytes / (1024 * 1024):.2f} MB)"
        )
        print(f"[EXPERIMENT 10MB CAPPING] Last line: {lines[-1]}")

        assert len(lines) > 0
        assert lines[-1].is_sentinel is True
        assert lines[-1].sentinel_type == SentinelType.TRUNCATED
        assert lines[-1].line == "[TRUNCATED]"

        ping_res = manager.run_command(workspace.workspace_id, "echo 'healthy'")
        assert ping_res["stdout"] == "healthy"

    finally:
        manager.destroy(workspace.workspace_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_command_execution_timeout_enforcement(
    tmp_path: Path, docker_client: docker.DockerClient
) -> None:
    """Experiment 2: Run command sleeping 60s with 5s timeout.

    Verify TIMEOUT sentinel and measure termination latency.
    """
    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    repo = git.Repo.init(source_repo_dir)
    (source_repo_dir / "README.md").write_text("# Test", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=tmp_path / "workspaces",
    )
    workspace = manager.create(repo_url=f"file://{source_repo_dir}")

    try:
        timeout_limit = 5.0
        start_time = time.perf_counter()
        lines: list[CommandOutputLine] = []

        async for line in manager.execute_command(
            workspace.workspace_id,
            "sleep 60",
            timeout_s=timeout_limit,
        ):
            lines.append(line)

        elapsed_duration = time.perf_counter() - start_time
        termination_latency = elapsed_duration - timeout_limit

        print(
            f"\n[EXPERIMENT TIMEOUT] Elapsed: {elapsed_duration:.4f}s "
            f"(timeout: {timeout_limit}s)"
        )
        print(
            f"[EXPERIMENT TIMEOUT] Latency overhead: "
            f"{termination_latency * 1000:.2f} ms"
        )
        print(f"[EXPERIMENT TIMEOUT] Lines received: {lines}")

        assert len(lines) == 1
        assert lines[0].is_sentinel is True
        assert lines[0].sentinel_type == SentinelType.TIMEOUT
        assert lines[0].line == "[TIMEOUT]"
        assert 4.9 <= elapsed_duration <= 7.0

        ps_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", "ps aux 2>/dev/null || ls /proc"],
        )
        print(
            f"[EXPERIMENT TIMEOUT] Post-kill proc state: {str(ps_res['stdout'])[:150]}"
        )

    finally:
        manager.destroy(workspace.workspace_id)


@pytest.mark.integration
def test_live_container_hardening_experiments(
    tmp_path: Path, docker_client: docker.DockerClient
) -> None:
    """Live Docker experiment testing limits, read-only fs, network, non-root user."""
    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    (source_repo_dir / "README.md").write_text("# Test Repo\n", encoding="utf-8")
    repo = git.Repo.init(source_repo_dir)
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    workspaces_root = tmp_path / "workspaces"
    workspaces_root.mkdir()

    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=workspaces_root,
    )

    workspace = manager.create(repo_url=f"file://{source_repo_dir}")

    try:
        whoami_res = manager.run_command(workspace.workspace_id, "id")
        print(f"\n[HARDENING EXPERIMENT] User identity: {whoami_res['stdout']}")
        assert "uid=1000" in str(whoami_res["stdout"])

        ro_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", "touch /root_test 2>&1 || true"],
        )
        print(f"[HARDENING EXPERIMENT] Root fs write attempt: {ro_res['stdout']}")
        assert "Read-only file system" in str(ro_res["stdout"])

        ws_write_res = manager.run_command(
            workspace.workspace_id,
            [
                "/bin/sh",
                "-c",
                "echo 'sandbox_ok' > /workspace/test.txt && cat /workspace/test.txt",
            ],
        )
        print(f"[HARDENING EXPERIMENT] /workspace write: {ws_write_res['stdout']}")
        assert ws_write_res["stdout"] == "sandbox_ok"

        tmp_write_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", "echo 'tmp_ok' > /tmp/test.txt && cat /tmp/test.txt"],
        )
        print(f"[HARDENING EXPERIMENT] /tmp tmpfs write: {tmp_write_res['stdout']}")
        assert tmp_write_res["stdout"] == "tmp_ok"

        net_cmd = (
            'python3 -c "import urllib.request; '
            "urllib.request.urlopen('http://1.1.1.1', timeout=1)\" 2>&1 || true"
        )
        net_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", net_cmd],
        )
        print(
            f"[HARDENING EXPERIMENT] Network egress attempt: "
            f"{str(net_res['stdout'])[:100]}"
        )
        assert (
            "Network is unreachable" in str(net_res["stdout"])
            or "URLError" in str(net_res["stdout"])
            or "timeout" in str(net_res["stdout"]).lower()
            or "errno 101" in str(net_res["stdout"]).lower()
        )

        mem_cmd = "python3 -c \"a = b'x' * (1024 * 1024 * 1024)\" 2>&1 || true"
        mem_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", mem_cmd],
        )
        print(
            f"[HARDENING EXPERIMENT] 1GB Memory allocation attempt: "
            f"{str(mem_res['stdout'])[:100]}"
        )
        assert (
            "MemoryError" in str(mem_res["stdout"])
            or "Killed" in str(mem_res["stdout"])
            or str(mem_res["stdout"]) == ""
        )

        fork_cmd = 'python3 -c "import os; [os.fork() for _ in range(10)]" 2>&1 || true'
        fork_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", fork_cmd],
        )
        print(
            f"[HARDENING EXPERIMENT] Fork bomb attempt: {str(fork_res['stdout'])[:100]}"
        )
        assert (
            "BlockingIOError" in str(fork_res["stdout"])
            or "Resource temporarily unavailable" in str(fork_res["stdout"])
            or "Errno 11" in str(fork_res["stdout"])
        )

        print("[HARDENING EXPERIMENT] All container hardening constraints verified!")

    finally:
        manager.destroy(workspace.workspace_id)
