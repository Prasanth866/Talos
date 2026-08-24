from __future__ import annotations

import time
from pathlib import Path

import docker
import docker.errors
import git
import pytest

from src.workspace import WorkspaceManager, WorkspaceStatus


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
        filters={"label": "managed-by=talos"},
    )
    assert len(matching_containers) == 0
    print("[EXPERIMENT] All Docker resources verified cleaned up.")
