"""Workspace package — Docker-based isolated environments with Git cloning."""

from src.workspace.exceptions import (
    DockerDaemonError,
    GitCloneError,
    WorkspaceCreationError,
    WorkspaceDestroyError,
    WorkspaceError,
    WorkspaceExecutionError,
    WorkspaceNotFoundError,
)
from src.workspace.git_utils import shallow_clone
from src.workspace.manager import WorkspaceManager
from src.workspace.models import Workspace, WorkspaceStatus

__all__ = [
    "DockerDaemonError",
    "GitCloneError",
    "Workspace",
    "WorkspaceCreationError",
    "WorkspaceDestroyError",
    "WorkspaceError",
    "WorkspaceExecutionError",
    "WorkspaceManager",
    "WorkspaceNotFoundError",
    "WorkspaceStatus",
    "shallow_clone",
]
