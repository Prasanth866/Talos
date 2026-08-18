import asyncio
import shlex
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    ToolError,
)

MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class ShellTool:
    """Defensive shell command executor with timeouts."""

    def __init__(self, working_dir: Path, timeout_seconds: float = 10.0) -> None:
        self.working_dir = working_dir.resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    async def run_shell(
        self,
        command_or_executable: str | Sequence[str],
        args: Sequence[str] | None = None,
    ) -> dict[str, str | int]:
        """Executes a command safely without shell=True."""
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

        process: asyncio.subprocess.Process | None = None
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
            if process is not None:
                with suppress(ProcessLookupError, OSError):
                    process.kill()
                    await process.wait()

            raise ExecutionTimeoutError(
                message=(
                    f"Command '{executable}' timed out after {self.timeout_seconds}s."
                ),
                tool_name="ShellTool",
                timeout_seconds=self.timeout_seconds,
                details={"executable": executable, "args": cmd_args},
            ) from exc
        except FileNotFoundError as exc:
            raise ToolError(
                message=f"Executable '{executable}' not found on system PATH.",
                tool_name="ShellTool",
                details={"executable": executable},
            ) from exc
        except PermissionError as exc:
            raise ToolError(
                message=f"Permission denied when executing '{executable}'.",
                tool_name="ShellTool",
                details={"executable": executable},
            ) from exc
        except Exception as exc:
            raise ToolError(
                message=f"Failed to execute command '{executable}': {exc}",
                tool_name="ShellTool",
                details={"executable": executable, "args": cmd_args},
            ) from exc

        stdout = _truncate(stdout_bytes.decode("utf-8", errors="replace").strip())
        stderr = _truncate(stderr_bytes.decode("utf-8", errors="replace").strip())

        return_code = process.returncode
        if return_code is None:
            raise ToolError(
                message=f"Process '{executable}' exited without a return code.",
                tool_name="ShellTool",
                details={"executable": executable, "args": cmd_args},
            )

        if return_code != 0:
            raise CommandExecutionError(
                message=f"Command failed with exit code {return_code}",
                tool_name="ShellTool",
                exit_code=return_code,
                stderr=stderr,
                details={"stdout": stdout},
            )

        return {"exit_code": return_code, "stdout": stdout, "stderr": stderr}

    async def execute(
        self, executable: str, args: Sequence[str]
    ) -> dict[str, str | int]:
        """Backward-compatible alias for run_shell."""
        return await self.run_shell(executable, args)


async def run_shell(
    working_dir: Path,
    command_or_executable: str | Sequence[str],
    args: Sequence[str] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, str | int]:
    """Convenience function to execute a shell command safely."""
    tool = ShellTool(working_dir=working_dir, timeout_seconds=timeout_seconds)
    return await tool.run_shell(command_or_executable, args)
