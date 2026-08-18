"""Tools package — re-exports the public API from filesystem and shell modules."""

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)
from src.tools.filesystem import (
    FileSystemTool,
    async_read_bytes,
    async_read_file,
    async_write_bytes,
    async_write_file,
    read_bytes,
    read_file,
    write_bytes,
    write_file,
)
from src.tools.shell import (
    MAX_OUTPUT_CHARS,
    ShellTool,
    _truncate,
    run_shell,
)

__all__ = [
    "MAX_OUTPUT_CHARS",
    "CommandExecutionError",
    "ExecutionTimeoutError",
    "FileSystemTool",
    "PathTraversalError",
    "ShellTool",
    "ToolError",
    "_truncate",
    "async_read_bytes",
    "async_read_file",
    "async_write_bytes",
    "async_write_file",
    "read_bytes",
    "read_file",
    "run_shell",
    "write_bytes",
    "write_file",
]
