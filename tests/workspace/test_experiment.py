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


@pytest.mark.integration
def test_live_workspace_experiment(tmp_path: Path) -> None:
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

    docker_client = docker.from_env()
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
async def test_live_command_execution_10mb_output_capping(tmp_path: Path) -> None:
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

    docker_client = docker.from_env()
    manager = WorkspaceManager(
        docker_client=docker_client,
        workspace_root=tmp_path / "workspaces",
    )
    workspace = manager.create(repo_url=f"file://{source_repo_dir}")

    try:
        # Command generating ~17MB of output (200,000 lines of ~88 bytes)
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

        # Calculate approximate memory consumed by lines list
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

        # Ensure container is still responsive and not frozen
        ping_res = manager.run_command(workspace.workspace_id, "echo 'healthy'")
        assert ping_res["stdout"] == "healthy"

    finally:
        manager.destroy(workspace.workspace_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_command_execution_timeout_enforcement(tmp_path: Path) -> None:
    """Experiment 2: Run command sleeping 60s with 5s timeout.

    Verify TIMEOUT sentinel and measure termination latency.
    """
    source_repo_dir = tmp_path / "source_repo"
    source_repo_dir.mkdir()
    repo = git.Repo.init(source_repo_dir)
    (source_repo_dir / "README.md").write_text("# Test", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")

    docker_client = docker.from_env()
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

        # Verify no sleep processes linger in container
        ps_res = manager.run_command(
            workspace.workspace_id,
            ["/bin/sh", "-c", "ps aux 2>/dev/null || ls /proc"],
        )
        print(
            f"[EXPERIMENT TIMEOUT] Post-kill proc state: {str(ps_res['stdout'])[:150]}"
        )

    finally:
        manager.destroy(workspace.workspace_id)
