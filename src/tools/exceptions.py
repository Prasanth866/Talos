from typing import Any


class ToolError(Exception):
    """Base exception for all tool execution failures."""

    def __init__(
        self,
        message: str,
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
    """Raised when an operation attempts to break outside the allowed sandbox directory."""


class ExecutionTimeoutError(ToolError):
    """Raised when a subprocess execution times out."""


class CommandExecutionError(ToolError):
    """Raised when a shell command fails or returns a non-zero exit code."""
