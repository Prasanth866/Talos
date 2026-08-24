from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import docker
import docker.errors
import structlog

from src.workspace.exceptions import (
    DockerDaemonError,
    WorkspaceCreationError,
    WorkspaceDestroyError,
    WorkspaceError,
    WorkspaceExecutionError,
    WorkspaceNotFoundError,
)
from src.workspace.git_utils import shallow_clone
from src.workspace.models import Workspace, WorkspaceStatus

logger = structlog.get_logger(__name__)

DEFAULT_BASE_IMAGE = "python:3.12-slim"
DEFAULT_WORKSPACE_LABELS = {"managed-by": "talos"}


class WorkspaceManager:
    """Manages Docker-based isolated task workspaces with Git repository checkout."""

    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        base_image: str = DEFAULT_BASE_IMAGE,
        workspace_root: Path | None = None,
        container_labels: dict[str, str] | None = None,
    ) -> None:
        self.base_image = base_image
        self.workspace_root = (
            workspace_root
            if workspace_root is not None
            else Path(tempfile.gettempdir()) / "talos_workspaces"
        ).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.container_labels = dict(container_labels or DEFAULT_WORKSPACE_LABELS)
        self._workspaces: dict[str, Workspace] = {}

        if docker_client is not None:
            self.client = docker_client
        else:
            try:
                self.client = docker.from_env()
                self.client.ping()
            except docker.errors.DockerException as exc:
                logger.error("docker_daemon_unavailable", error=str(exc))
                raise DockerDaemonError(
                    message=f"Docker daemon is unavailable or unreachable: {exc}",
                    details={"error": str(exc)},
                ) from exc

    def create(
        self,
        repo_url: str,
        commit_sha: str | None = None,
        image: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Workspace:
        """Creates an isolated container workspace and shallow-clones the repo."""
        workspace_id = uuid.uuid4().hex
        container_name = f"talos-ws-{workspace_id[:12]}"
        host_dir = self.workspace_root / workspace_id
        target_image = image or self.base_image

        logger.info(
            "workspace_create_started",
            workspace_id=workspace_id,
            repo_url=repo_url,
            commit_sha=commit_sha,
            image=target_image,
        )

        try:
            host_dir.mkdir(parents=True, exist_ok=True)
            resolved_sha = shallow_clone(
                repo_url=repo_url,
                target_dir=host_dir,
                commit_sha=commit_sha,
            )

            container = self.client.containers.run(
                image=target_image,
                name=container_name,
                command="tail -f /dev/null",
                detach=True,
                volumes={str(host_dir): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                environment=env or {},
                labels=self.container_labels,
                remove=False,
            )

            workspace = Workspace(
                workspace_id=workspace_id,
                container_id=str(container.id),
                container_name=container_name,
                repo_url=repo_url,
                commit_sha=resolved_sha,
                host_dir=host_dir,
                status=WorkspaceStatus.RUNNING,
                metadata={"image": target_image},
            )
            self._workspaces[workspace_id] = workspace

            logger.info(
                "workspace_create_completed",
                workspace_id=workspace_id,
                container_id=workspace.container_id,
                container_name=container_name,
            )
            return workspace

        except WorkspaceError:
            shutil.rmtree(host_dir, ignore_errors=True)
            raise
        except docker.errors.DockerException as exc:
            shutil.rmtree(host_dir, ignore_errors=True)
            logger.error(
                "workspace_create_docker_error",
                workspace_id=workspace_id,
                error=str(exc),
            )
            raise WorkspaceCreationError(
                message=f"Failed to create workspace container: {exc}",
                details={
                    "workspace_id": workspace_id,
                    "image": target_image,
                    "error": str(exc),
                },
            ) from exc
        except Exception as exc:
            shutil.rmtree(host_dir, ignore_errors=True)
            logger.error(
                "workspace_create_unexpected_error",
                workspace_id=workspace_id,
                error=str(exc),
            )
            raise WorkspaceCreationError(
                message=f"Unexpected failure during workspace creation: {exc}",
                details={"workspace_id": workspace_id, "error": str(exc)},
            ) from exc

    def get(self, workspace_id: str) -> Workspace:
        """Retrieves a workspace by its unique ID."""
        if workspace_id not in self._workspaces:
            raise WorkspaceNotFoundError(
                message=f"Workspace '{workspace_id}' was not found.",
                workspace_id=workspace_id,
            )
        return self._workspaces[workspace_id]

    def destroy(self, workspace_id: str, force: bool = True) -> None:
        """Stops and removes the container, deletes host files, and deregisters."""
        workspace = self.get(workspace_id)
        logger.info(
            "workspace_destroy_started",
            workspace_id=workspace_id,
            container_id=workspace.container_id,
        )

        try:
            try:
                container = self.client.containers.get(workspace.container_id)
                container.stop(timeout=5)
                container.remove(force=force)
            except docker.errors.NotFound:
                logger.debug(
                    "workspace_container_already_removed",
                    container_id=workspace.container_id,
                )
            except docker.errors.DockerException as exc:
                raise WorkspaceDestroyError(
                    message=(
                        f"Failed to destroy container for workspace "
                        f"'{workspace_id}': {exc}"
                    ),
                    workspace_id=workspace_id,
                    details={
                        "container_id": workspace.container_id,
                        "error": str(exc),
                    },
                ) from exc

            if workspace.host_dir.exists():
                shutil.rmtree(workspace.host_dir, ignore_errors=True)

            workspace.status = WorkspaceStatus.DESTROYED
            self._workspaces.pop(workspace_id, None)

            logger.info(
                "workspace_destroy_completed",
                workspace_id=workspace_id,
            )
        except WorkspaceError:
            raise
        except Exception as exc:
            raise WorkspaceDestroyError(
                message=(
                    f"Unexpected error destroying workspace '{workspace_id}': {exc}"
                ),
                workspace_id=workspace_id,
                details={"error": str(exc)},
            ) from exc

    def destroy_all(self, force: bool = True) -> None:
        """Best-effort cleanup of all registered workspaces."""
        logger.info(
            "workspace_destroy_all_started",
            active_count=len(self._workspaces),
        )
        workspace_ids = list(self._workspaces.keys())
        for wid in workspace_ids:
            try:
                self.destroy(wid, force=force)
            except Exception as exc:
                logger.warning(
                    "workspace_destroy_failed_during_destroy_all",
                    workspace_id=wid,
                    error=str(exc),
                )

    def run_command(
        self,
        workspace_id: str,
        command: str | list[str],
        workdir: str = "/workspace",
    ) -> dict[str, str | int]:
        """Executes a command inside the workspace container."""
        workspace = self.get(workspace_id)
        try:
            container = self.client.containers.get(workspace.container_id)
            exec_res = container.exec_run(cmd=command, workdir=workdir)
            exit_code: int = exec_res.exit_code if exec_res.exit_code is not None else 0
            raw_output = exec_res.output
            if isinstance(raw_output, tuple):
                raw_bytes = (raw_output[0] or b"") if raw_output else b""
            elif isinstance(raw_output, bytes):
                raw_bytes = raw_output
            elif raw_output is not None:
                raw_bytes = b"".join(
                    chunk for chunk in raw_output if isinstance(chunk, bytes)
                )
            else:
                raw_bytes = b""
            output_str = raw_bytes.decode("utf-8", errors="replace").strip()

            if exit_code != 0:
                logger.warning(
                    "workspace_command_nonzero_exit",
                    workspace_id=workspace_id,
                    command=command,
                    exit_code=exit_code,
                    output=output_str,
                )
                raise WorkspaceExecutionError(
                    message=f"Command '{command}' failed with exit code {exit_code}",
                    command=command,
                    exit_code=exit_code,
                    stderr=output_str,
                    stdout=output_str,
                )

            return {
                "exit_code": exit_code,
                "stdout": output_str,
                "stderr": "",
            }
        except WorkspaceError:
            raise
        except docker.errors.DockerException as exc:
            logger.error(
                "workspace_command_docker_error",
                workspace_id=workspace_id,
                command=command,
                error=str(exc),
            )
            raise WorkspaceExecutionError(
                message=f"Docker error executing command in workspace: {exc}",
                command=command,
                details={"error": str(exc)},
            ) from exc
        except Exception as exc:
            logger.error(
                "workspace_command_unexpected_error",
                workspace_id=workspace_id,
                command=command,
                error=str(exc),
            )
            raise WorkspaceExecutionError(
                message=f"Unexpected error executing command in workspace: {exc}",
                command=command,
                details={"error": str(exc)},
            ) from exc

    def list_files(self, workspace_id: str, path: str = ".") -> list[str]:
        """Lists files inside the workspace directory."""
        result = self.run_command(
            workspace_id=workspace_id,
            command=["find", path, "-maxdepth", "3", "-not", "-path", "*/.*"],
        )
        stdout = str(result.get("stdout", ""))
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        return lines
