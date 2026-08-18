import asyncio
from pathlib import Path
from typing import Literal, overload

import structlog

from src.tools.exceptions import (
    PathTraversalError,
    ToolError,
)

logger = structlog.get_logger(__name__)


class FileSystemTool:
    """Defensive file system wrapper restricted to a sandbox root directory."""

    def __init__(self, sandbox_dir: Path) -> None:
        self.sandbox_dir = sandbox_dir.resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str | Path) -> Path:
        """Resolves and verifies the target path is within the sandbox."""
        clean_path_str = str(relative_path).strip()
        if not clean_path_str:
            raise ToolError(
                message="File path cannot be empty.",
                tool_name="FileSystemTool",
                details={"requested_path": str(relative_path)},
            )

        # Normalize away leading slashes so Path / '/foo' doesn't escape to root
        sanitized_rel = clean_path_str.lstrip("/\\")
        target_path = (self.sandbox_dir / sanitized_rel).resolve()

        if not target_path.is_relative_to(self.sandbox_dir):
            logger.warning(
                "path_traversal_blocked",
                target=str(target_path),
                sandbox=str(self.sandbox_dir),
            )
            raise PathTraversalError(
                message=(
                    f"Access denied: path '{relative_path}' is outside sandbox root."
                ),
                tool_name="FileSystemTool",
                attempted_path=str(target_path),
            )
        return target_path

    def _get_validated_file_path(self, relative_path: str | Path) -> Path:
        """Resolves path safely and asserts that it exists and is a regular file."""
        safe_path = self._resolve_safe_path(relative_path)
        if not safe_path.exists():
            raise ToolError(
                message=f"File not found: '{relative_path}'",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            )
        if safe_path.is_dir():
            raise ToolError(
                message=f"Path '{relative_path}' is a directory, not a file.",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            )
        return safe_path

    @overload
    def read_file(
        self, relative_path: str | Path, binary: Literal[False] = False
    ) -> str: ...

    @overload
    def read_file(self, relative_path: str | Path, binary: Literal[True]) -> bytes: ...

    @overload
    def read_file(self, relative_path: str | Path, binary: bool) -> str | bytes: ...

    def read_file(self, relative_path: str | Path, binary: bool = False) -> str | bytes:
        """Reads file contents from the sandbox directory."""
        if binary:
            return self.read_bytes(relative_path)
        safe_path = self._get_validated_file_path(relative_path)
        try:
            return safe_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ToolError(
                message=f"Failed to read file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc

    def read_bytes(self, relative_path: str | Path) -> bytes:
        """Reads raw binary content from a file in the sandbox directory."""
        safe_path = self._get_validated_file_path(relative_path)
        try:
            return safe_path.read_bytes()
        except Exception as exc:
            raise ToolError(
                message=f"Failed to read file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc

    def write_file(
        self,
        relative_path: str | Path,
        content: str | bytes,
        binary: bool = False,
    ) -> None:
        """Writes content to a file inside the sandbox directory."""
        if binary or isinstance(content, bytes):
            if isinstance(content, str):
                content = content.encode("utf-8")
            self.write_bytes(relative_path, content)
            return

        safe_path = self._resolve_safe_path(relative_path)
        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            if not safe_path.parent.resolve().is_relative_to(self.sandbox_dir):
                raise PathTraversalError(
                    message=(
                        f"Access denied: directory for '{relative_path}'"
                        " points outside sandbox."
                    ),
                    tool_name="FileSystemTool",
                    attempted_path=str(safe_path),
                )
            safe_path.write_text(content, encoding="utf-8")
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                message=f"Failed to write file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc

    def write_bytes(self, relative_path: str | Path, content: bytes) -> None:
        """Writes raw binary content to a file inside the sandbox directory."""
        safe_path = self._resolve_safe_path(relative_path)
        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            if not safe_path.parent.resolve().is_relative_to(self.sandbox_dir):
                raise PathTraversalError(
                    message=(
                        f"Access denied: directory for '{relative_path}'"
                        " points outside sandbox."
                    ),
                    tool_name="FileSystemTool",
                    attempted_path=str(safe_path),
                )
            safe_path.write_bytes(content)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                message=f"Failed to write file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc

    @overload
    async def async_read_file(
        self, relative_path: str | Path, binary: Literal[False] = False
    ) -> str: ...

    @overload
    async def async_read_file(
        self, relative_path: str | Path, binary: Literal[True]
    ) -> bytes: ...

    @overload
    async def async_read_file(
        self, relative_path: str | Path, binary: bool
    ) -> str | bytes: ...

    async def async_read_file(
        self, relative_path: str | Path, binary: bool = False
    ) -> str | bytes:
        return await asyncio.to_thread(self.read_file, relative_path, binary)

    async def async_read_bytes(self, relative_path: str | Path) -> bytes:
        return await asyncio.to_thread(self.read_bytes, relative_path)

    async def async_write_file(
        self,
        relative_path: str | Path,
        content: str | bytes,
        binary: bool = False,
    ) -> None:
        await asyncio.to_thread(self.write_file, relative_path, content, binary)

    async def async_write_bytes(
        self, relative_path: str | Path, content: bytes
    ) -> None:
        await asyncio.to_thread(self.write_bytes, relative_path, content)


@overload
def read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: Literal[False] = False
) -> str: ...


@overload
def read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: Literal[True]
) -> bytes: ...


@overload
def read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: bool
) -> str | bytes: ...


def read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: bool = False
) -> str | bytes:
    """Convenience function to read a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    return tool.read_file(relative_path, binary=binary)


def read_bytes(sandbox_dir: Path, relative_path: str | Path) -> bytes:
    """Convenience function to read a binary file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    return tool.read_bytes(relative_path)


def write_file(
    sandbox_dir: Path,
    relative_path: str | Path,
    content: str | bytes,
    binary: bool = False,
) -> None:
    """Convenience function to write a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    tool.write_file(relative_path, content, binary=binary)


def write_bytes(sandbox_dir: Path, relative_path: str | Path, content: bytes) -> None:
    """Convenience function to write a binary file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    tool.write_bytes(relative_path, content)


@overload
async def async_read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: Literal[False] = False
) -> str: ...


@overload
async def async_read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: Literal[True]
) -> bytes: ...


@overload
async def async_read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: bool
) -> str | bytes: ...


async def async_read_file(
    sandbox_dir: Path, relative_path: str | Path, binary: bool = False
) -> str | bytes:
    """Convenience async function to read a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    return await tool.async_read_file(relative_path, binary=binary)


async def async_read_bytes(sandbox_dir: Path, relative_path: str | Path) -> bytes:
    """Convenience async function to read a binary file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    return await tool.async_read_bytes(relative_path)


async def async_write_file(
    sandbox_dir: Path,
    relative_path: str | Path,
    content: str | bytes,
    binary: bool = False,
) -> None:
    """Convenience async function to write a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    await tool.async_write_file(relative_path, content, binary=binary)


async def async_write_bytes(
    sandbox_dir: Path, relative_path: str | Path, content: bytes
) -> None:
    """Convenience async function to write a binary file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    await tool.async_write_bytes(relative_path, content)
