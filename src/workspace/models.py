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


class SentinelType(StrEnum):
    TRUNCATED = "TRUNCATED"
    TIMEOUT = "TIMEOUT"


@dataclass
class CommandOutputLine:
    """Represents an output line or sentinel marker from sandbox command execution."""

    line: str
    stream: str = "stdout"
    is_sentinel: bool = False
    sentinel_type: SentinelType | str | None = None

    @property
    def content(self) -> str:
        """Alias for line content."""
        return self.line

    @classmethod
    def truncated(cls) -> CommandOutputLine:
        """Factory method for TRUNCATED output cap sentinel."""
        return cls(
            line="[TRUNCATED]",
            stream="system",
            is_sentinel=True,
            sentinel_type=SentinelType.TRUNCATED,
        )

    @classmethod
    def timeout(cls) -> CommandOutputLine:
        """Factory method for TIMEOUT sentinel."""
        return cls(
            line="[TIMEOUT]",
            stream="system",
            is_sentinel=True,
            sentinel_type=SentinelType.TIMEOUT,
        )


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
