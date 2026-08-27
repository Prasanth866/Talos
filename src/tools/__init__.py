"""Tools package — re-exports the public API from filesystem and shell modules."""

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PatchError,
    PathTraversalError,
    ToolError,
    ToolValidationError,
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
from src.tools.patch import (
    FilePatch,
    Hunk,
    PatchTool,
    parse_unified_diff,
)
from src.tools.shell import (
    MAX_OUTPUT_CHARS,
    ShellTool,
    run_shell,
)

__all__ = [
    "MAX_OUTPUT_CHARS",
    "CommandExecutionError",
    "ExecutionTimeoutError",
    "FilePatch",
    "FileSystemTool",
    "Hunk",
    "PatchError",
    "PatchTool",
    "PathTraversalError",
    "ShellTool",
    "ToolError",
    "ToolValidationError",
    "async_read_bytes",
    "async_read_file",
    "async_write_bytes",
    "async_write_file",
    "parse_unified_diff",
    "read_bytes",
    "read_file",
    "run_shell",
    "write_bytes",
    "write_file",
]
