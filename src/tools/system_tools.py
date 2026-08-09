import asyncio
import shlex
from collections.abc import Sequence
from pathlib import Path

import structlog

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)

logger = structlog.get_logger(__name__)


class FileSystemTool:
    """Defensive file system wrapper restricted to a sandbox root directory."""

    def __init__(self, sandbox_dir: Path) -> None:
        self.sandbox_dir = sandbox_dir.resolve()
        if not self.sandbox_dir.exists():
            self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str | Path) -> Path:
        """Resolves target path and verifies it resides within the allowed sandbox root."""
        str_path = str(relative_path).strip()
        if not str_path:
            raise ToolError(
                message="File path cannot be empty.",
                tool_name="FileSystemTool",
                details={"requested_path": str(relative_path)},
            )

        target_path = (self.sandbox_dir / relative_path).resolve()

        if not target_path.is_relative_to(self.sandbox_dir):
            logger.warning(
                "path_traversal_blocked",
                target=str(target_path),
                sandbox=str(self.sandbox_dir),
            )
            raise PathTraversalError(
                message=f"Access denied: path '{relative_path}' is outside sandbox root.",
                tool_name="FileSystemTool",
                details={"requested_path": str(relative_path)},
            )
        return target_path

    def read_file(self, relative_path: str | Path) -> str:
        """Reads file contents from the sandbox directory."""
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
        try:
            return safe_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ToolError(
                message=f"Failed to read file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc

    def write_file(self, relative_path: str | Path, content: str) -> None:
        """Writes content to a file inside the sandbox directory."""
        safe_path = self._resolve_safe_path(relative_path)
        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding="utf-8")
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                message=f"Failed to write file: {exc}",
                tool_name="FileSystemTool",
                details={"path": str(relative_path)},
            ) from exc


class ShellTool:
    """Defensive shell command executor with timeouts and strict execution boundaries."""

    def __init__(self, working_dir: Path, timeout_seconds: float = 10.0) -> None:
        self.working_dir = working_dir.resolve()
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    async def run_shell(
        self,
        command_or_executable: str | Sequence[str],
        args: Sequence[str] | None = None,
    ) -> dict[str, str | int]:
        """Executes a command safely without shell=True.

        Accepts either a single command string, a list of command tokens,
        or an executable with an explicit list of arguments.
        """
        if isinstance(command_or_executable, str):
            command_str = command_or_executable.strip()
            if not command_str:
                raise ToolError(
                    message="Command string cannot be empty.",
                    tool_name="ShellTool",
                    details={"command": command_or_executable},
                )
            if args is not None:
                executable = command_str
                cmd_args = list(args)
            else:
                parts = shlex.split(command_str)
                if not parts:
                    raise ToolError(
                        message="Command string parsed to empty arguments.",
                        tool_name="ShellTool",
                        details={"command": command_or_executable},
                    )
                executable = parts[0]
                cmd_args = parts[1:]
        else:
            if not command_or_executable:
                raise ToolError(
                    message="Command sequence cannot be empty.",
                    tool_name="ShellTool",
                    details={"command": command_or_executable},
                )
            executable = command_or_executable[0]
            cmd_args = list(command_or_executable[1:]) + list(args or [])

        if not executable.strip():
            raise ToolError(
                message="Executable name cannot be empty.",
                tool_name="ShellTool",
                details={"executable": executable},
            )

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )

        except TimeoutError as exc:
            raise ExecutionTimeoutError(
                message=f"Command '{executable}' timed out after {self.timeout_seconds}s.",
                tool_name="ShellTool",
                details={"executable": executable, "args": cmd_args},
            ) from exc
        except FileNotFoundError as exc:
            raise ToolError(
                message=f"Executable '{executable}' not found on system PATH.",
                tool_name="ShellTool",
                details={"executable": executable},
            ) from exc
        except Exception as exc:
            raise ToolError(
                message=f"Failed to execute command '{executable}': {exc}",
                tool_name="ShellTool",
                details={"executable": executable, "args": cmd_args},
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            raise CommandExecutionError(
                message=f"Command failed with exit code {process.returncode}",
                tool_name="ShellTool",
                details={
                    "exit_code": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        return {"exit_code": 0, "stdout": stdout, "stderr": stderr}

    async def execute(
        self, executable: str, args: Sequence[str]
    ) -> dict[str, str | int]:
        """Backward-compatible alias for run_shell."""
        return await self.run_shell(executable, args)


def read_file(sandbox_dir: Path, relative_path: str | Path) -> str:
    """Convenience function to read a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    return tool.read_file(relative_path)


def write_file(sandbox_dir: Path, relative_path: str | Path, content: str) -> None:
    """Convenience function to write a file within a sandbox directory."""
    tool = FileSystemTool(sandbox_dir=sandbox_dir)
    tool.write_file(relative_path, content)


async def run_shell(
    working_dir: Path,
    command_or_executable: str | Sequence[str],
    args: Sequence[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, str | int]:
    """Convenience function to execute a shell command safely."""
    tool = ShellTool(working_dir=working_dir, timeout_seconds=timeout_seconds)
    return await tool.run_shell(command_or_executable, args)
