from __future__ import annotations

from typing import Any


class WorkspaceError(Exception):
    """Base exception for all workspace operations."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Returns structured error representation."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DockerDaemonError(WorkspaceError):
    """Raised when the Docker daemon is unreachable or unavailable."""


class WorkspaceCreationError(WorkspaceError):
    """Raised when directory creation, git clone, or container startup fails."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a workspace with the given ID cannot be found."""

    def __init__(
        self,
        message: str,
        *,
        workspace_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"workspace_id": workspace_id, **(details or {})}
        super().__init__(message, details=merged_details)
        self.workspace_id = workspace_id


class WorkspaceDestroyError(WorkspaceError):
    """Raised when teardown or cleanup of workspace resources encounters failures."""

    def __init__(
        self,
        message: str,
        *,
        workspace_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if workspace_id:
            merged_details["workspace_id"] = workspace_id
        super().__init__(message, details=merged_details)
        self.workspace_id = workspace_id


class GitCloneError(WorkspaceError):
    """Raised when shallow cloning or repository checkout fails."""

    def __init__(
        self,
        message: str,
        *,
        repo_url: str,
        commit_sha: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            **(details or {}),
        }
        super().__init__(message, details=merged_details)
        self.repo_url = repo_url
        self.commit_sha = commit_sha


class WorkspaceExecutionError(WorkspaceError):
    """Raised when command execution inside a workspace container fails."""

    def __init__(
        self,
        message: str,
        *,
        command: str | list[str],
        exit_code: int | None = None,
        stderr: str | None = None,
        stdout: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "command": command,
            "exit_code": exit_code,
            "stderr": stderr,
            "stdout": stdout,
            **(details or {}),
        }
        super().__init__(message, details=merged_details)
        self.command = command
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
