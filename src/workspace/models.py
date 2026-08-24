from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class WorkspaceStatus(StrEnum):
    CREATING = "creating"
    RUNNING = "running"
    DESTROYED = "destroyed"
    FAILED = "failed"


@dataclass
class Workspace:
    """Represents an isolated container workspace environment."""

    workspace_id: str
    container_id: str
    container_name: str
    repo_url: str
    commit_sha: str | None
    host_dir: Path
    status: WorkspaceStatus = WorkspaceStatus.RUNNING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
