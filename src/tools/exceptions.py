from typing import Any


class ToolError(Exception):
    """Base exception for all tool execution failures."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Returns structured error representation."""
        return {
            "error": self.__class__.__name__,
            "tool_name": self.tool_name,
            "message": self.message,
            "details": self.details,
        }


class PathTraversalError(ToolError):
    """Raised when an operation attempts to escape the sandbox directory."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        attempted_path: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"attempted_path": attempted_path, **(details or {})}
        super().__init__(message, tool_name=tool_name, details=merged_details)
        self.attempted_path = attempted_path


class ExecutionTimeoutError(ToolError):
    """Raised when a subprocess execution times out."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        timeout_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"timeout_seconds": timeout_seconds, **(details or {})}
        super().__init__(message, tool_name=tool_name, details=merged_details)
        self.timeout_seconds = timeout_seconds


class CommandExecutionError(ToolError):
    """Raised when a shell command fails or returns a non-zero exit code."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        exit_code: int,
        stderr: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"exit_code": exit_code, "stderr": stderr, **(details or {})}
        super().__init__(message, tool_name=tool_name, details=merged_details)
        self.exit_code = exit_code
        self.stderr = stderr
