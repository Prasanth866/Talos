from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.agent.models import ToolCall, ToolResult
from src.agent.prompts import format_tool_doc
from src.tools.exceptions import ToolError
from src.tools.filesystem import FileSystemTool
from src.tools.shell import ShellTool

logger = structlog.get_logger(__name__)


@dataclass
class ToolDefinition:
    """Metadata and execution callable for a registered tool."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters_schema: dict[str, Any]
    is_async: bool


class ToolDispatcher:
    """Registry and execution dispatcher for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        """Registers a function as an agent tool."""
        schema = parameters_schema or {}
        is_async = inspect.iscoroutinefunction(handler)
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            handler=handler,
            parameters_schema=schema,
            is_async=is_async,
        )
        logger.debug("tool_registered", tool_name=name, is_async=is_async)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tools_documentation(self) -> str:
        """Generates formatted documentation string for all registered tools."""
        docs = []
        for tool in self._tools.values():
            docs.append(
                format_tool_doc(
                    name=tool.name,
                    description=tool.description,
                    parameters_schema=tool.parameters_schema,
                )
            )
        return "\n".join(docs)

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatches a tool call safely, returning a structured ToolResult."""
        start_time = time.perf_counter()
        tool_name = tool_call.tool_name

        if tool_name not in self._tools:
            duration = time.perf_counter() - start_time
            available = ", ".join(self.get_tool_names())
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=(
                    f"Unknown tool '{tool_name}'. Available tools are: [{available}]"
                ),
                success=False,
                duration_seconds=duration,
            )

        tool_def = self._tools[tool_name]
        try:
            kwargs = tool_call.arguments
            if tool_def.is_async:
                raw_result = await tool_def.handler(**kwargs)
            else:
                raw_result = tool_def.handler(**kwargs)

            if isinstance(raw_result, dict):
                if "exit_code" in raw_result:
                    exit_code = raw_result.get("exit_code")
                    stdout = raw_result.get("stdout", "")
                    stderr = raw_result.get("stderr", "")
                    output = f"Exit code: {exit_code}\nSTDOUT:\n{stdout}"
                    if stderr:
                        output += f"\nSTDERR:\n{stderr}"
                else:
                    output = str(raw_result)
            elif isinstance(raw_result, bytes):
                output = raw_result.decode("utf-8", errors="replace")
            elif raw_result is None:
                output = "Operation completed successfully."
            else:
                output = str(raw_result)

            duration = time.perf_counter() - start_time
            return ToolResult(
                tool_name=tool_name,
                output=output,
                error=None,
                success=True,
                duration_seconds=duration,
            )

        except ToolError as exc:
            duration = time.perf_counter() - start_time
            logger.warning(
                "tool_execution_failed",
                tool_name=tool_name,
                error=str(exc),
                duration_s=round(duration, 3),
            )
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=f"{type(exc).__name__}: {exc.message}",
                success=False,
                duration_seconds=duration,
            )
        except TypeError as exc:
            duration = time.perf_counter() - start_time
            logger.warning(
                "tool_arguments_invalid",
                tool_name=tool_name,
                error=str(exc),
                arguments=tool_call.arguments,
            )
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=f"Invalid arguments for tool '{tool_name}': {exc}",
                success=False,
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(
                "tool_unexpected_error",
                tool_name=tool_name,
                error=str(exc),
                exc_info=True,
            )
            err_msg = (
                f"Unexpected error executing '{tool_name}': {type(exc).__name__}: {exc}"
            )
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=err_msg,
                success=False,
                duration_seconds=duration,
            )


def create_default_dispatcher(sandbox_dir: Path) -> ToolDispatcher:
    """Factory creating a standard tool dispatcher with FileSystem and Shell tools."""
    sandbox_dir = sandbox_dir.resolve()
    fs = FileSystemTool(sandbox_dir=sandbox_dir)
    shell = ShellTool(working_dir=sandbox_dir)

    dispatcher = ToolDispatcher()

    def _read_file(path: str) -> str:
        return fs.read_file(path)

    dispatcher.register_tool(
        name="read_file",
        description="Reads the text content of a file within the sandbox.",
        handler=_read_file,
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to read.",
                }
            },
            "required": ["path"],
        },
    )

    def _write_file(path: str, content: str) -> str:
        fs.write_file(path, content)
        return f"File '{path}' written successfully."

    dispatcher.register_tool(
        name="write_file",
        description="Writes content to a file located within the sandbox directory.",
        handler=_write_file,
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Content string to write.",
                },
            },
            "required": ["path", "content"],
        },
    )

    def _list_dir(path: str = ".") -> str:
        target = (sandbox_dir / path).resolve()
        if not str(target).startswith(str(sandbox_dir)):
            return f"Error: Path '{path}' is outside the sandbox."
        if not target.exists():
            return f"Error: Directory '{path}' does not exist."
        if not target.is_dir():
            return f"Error: Path '{path}' is not a directory."

        entries = []
        for entry in sorted(target.iterdir()):
            rel = entry.relative_to(sandbox_dir)
            type_str = "DIR " if entry.is_dir() else "FILE"
            entries.append(f"[{type_str}] {rel}")
        return "\n".join(entries) if entries else f"Directory '{path}' is empty."

    dispatcher.register_tool(
        name="list_dir",
        description="Lists files and directories in a sandbox directory.",
        handler=_list_dir,
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory (default: '.')",
                    "default": ".",
                }
            },
            "required": [],
        },
    )

    async def _run_shell(command: str) -> dict[str, str | int]:
        return await shell.run_shell(command)

    dispatcher.register_tool(
        name="run_shell",
        description="Executes a shell command safely inside the sandbox directory.",
        handler=_run_shell,
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command string (e.g. 'pytest tests/').",
                }
            },
            "required": ["command"],
        },
    )

    return dispatcher
