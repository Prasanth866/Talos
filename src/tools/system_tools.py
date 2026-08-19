"""Backward-compatible re-export shim.

Import from src.tools.filesystem or src.tools.shell instead.
"""

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
    run_shell,
)

__all__ = [
    "MAX_OUTPUT_CHARS",
    "FileSystemTool",
    "ShellTool",
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
