import asyncio
import os
import re
import shlex
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    ToolError,
)

MAX_OUTPUT_CHARS = 4000

# Commands that are too dangerous for an autonomous agent to execute.
# This is a heuristic defense layer — production deployments should also
# enforce OS-level isolation (containers, bubblewrap, nsjail).
_DENIED_EXECUTABLES = frozenset(
    {
        "sudo",
        "su",
        "chown",
        "chmod",
        "chgrp",
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "rsync",
        "ftp",
        "sftp",
        "docker",
        "podman",
        "kubectl",
        "mount",
        "umount",
        "mkfs",
        "fdisk",
        "shutdown",
        "reboot",
        "halt",
        "init",
        "dd",
        "shred",
        "nohup",
    }
)

# Shell patterns that indicate dangerous intent regardless of executable.
_DENIED_PATTERNS = [
    re.compile(r"rm\s+.*-.*r.*f", re.IGNORECASE),  # rm -rf / rm -fr
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),  # rm -rf /
    re.compile(r">{1,2}\s*/dev/"),  # redirect to /dev/
    re.compile(r"\|\s*(bash|sh|zsh|dash)"),  # pipe to shell
    re.compile(r"mkfifo"),  # named pipes
    re.compile(r"eval\s"),  # eval injection
]


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class ShellTool:
    """Defensive shell command executor with timeouts and command safety checks."""

    def __init__(self, working_dir: Path, timeout_seconds: float = 10.0) -> None:
        self.working_dir = working_dir.resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _check_command_safety(command_str: str, executable: str) -> None:
        """Raises ToolError if the command matches known dangerous patterns."""
        exe_basename = Path(executable).name.lower()
        if exe_basename in _DENIED_EXECUTABLES:
            raise ToolError(
                message=f"Command '{exe_basename}' is not permitted.",
                tool_name="ShellTool",
                details={"executable": executable, "reason": "denied_executable"},
            )
        for pattern in _DENIED_PATTERNS:
            if pattern.search(command_str):
                raise ToolError(
                    message="Command matches a denied safety pattern.",
                    tool_name="ShellTool",
                    details={
                        "command": command_str,
                        "pattern": pattern.pattern,
                        "reason": "denied_pattern",
                    },
                )

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

        # Validate command against denylist before execution
        cmd_string = f"{executable} {' '.join(cmd_args)}"
        self._check_command_safety(cmd_string, executable)

        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "")
        if venv_bin not in current_path.split(os.pathsep):
            env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

        process: asyncio.subprocess.Process | None = None
        try:
            created_process = await asyncio.create_subprocess_exec(
                executable,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
            )
            process = created_process
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                created_process.communicate(), timeout=self.timeout_seconds
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

        if process is None:
            raise ToolError(
                message="Process was not created.",
                tool_name="ShellTool",
                details={"executable": executable},
            )

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
