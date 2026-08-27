from typing import Any


class ToolError(Exception):
    """Base exception for all tool execution failures."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        code: str = "TOOL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Returns structured error representation."""
        return {
            "error": self.__class__.__name__,
            "tool_name": self.tool_name,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ToolValidationError(ToolError):
    """Raised when tool arguments fail schema validation before execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        validation_error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"validation_error": validation_error, **(details or {})}
        super().__init__(
            message,
            tool_name=tool_name,
            code="SCHEMA_VALIDATION_ERROR",
            details=merged_details,
        )


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
        super().__init__(
            message,
            tool_name=tool_name,
            code="PATH_TRAVERSAL",
            details=merged_details,
        )
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
        super().__init__(
            message,
            tool_name=tool_name,
            code="TIMEOUT",
            details=merged_details,
        )
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
        super().__init__(
            message,
            tool_name=tool_name,
            code="COMMAND_FAILED",
            details=merged_details,
        )
        self.exit_code = exit_code
        self.stderr = stderr


class CircuitOpenError(ToolError):
    """Raised when the agent circuit breaker trips due to consecutive failures."""

    def __init__(
        self,
        message: str = "Circuit breaker is OPEN due to consecutive failures",
        *,
        tool_name: str = "circuit_breaker",
        consecutive_failures: int = 3,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "consecutive_failures": consecutive_failures,
            **(details or {}),
        }
        super().__init__(
            message,
            tool_name=tool_name,
            code="CIRCUIT_OPEN",
            details=merged_details,
        )
        self.consecutive_failures = consecutive_failures


class PatchError(ToolError):
    """Raised when a unified diff fails to validate or apply cleanly."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "apply_patch",
        reason: str = "PATCH_FAILED",
        context: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {"reason": reason, "context": context, **(details or {})}
        super().__init__(
            message,
            tool_name=tool_name,
            code="PATCH_ERROR",
            details=merged_details,
        )
        self.reason = reason
        self.context = context


class BudgetExceededError(ToolError):
    """Raised when a task exceeds its configured token or cost budget."""

    def __init__(
        self,
        message: str = "Task budget exceeded",
        *,
        tool_name: str = "token_tracker",
        budget_type: str = "tokens",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "budget_type": budget_type,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            **(details or {}),
        }
        super().__init__(
            message,
            tool_name=tool_name,
            code="BUDGET_EXCEEDED",
            details=merged_details,
        )
        self.budget_type = budget_type
        self.tokens_used = tokens_used
        self.cost_usd = cost_usd


class DangerousCommandError(ToolError):
    """Raised when a command matches dangerous shell execution safety patterns."""

    def __init__(
        self,
        message: str = "Command matches a denied safety pattern.",
        *,
        tool_name: str = "ShellTool",
        command: str = "",
        blocked_pattern: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "command": command,
            "blocked_pattern": blocked_pattern,
            **(details or {}),
        }
        super().__init__(
            message,
            tool_name=tool_name,
            code="DANGEROUS_COMMAND",
            details=merged_details,
        )
        self.command = command
        self.blocked_pattern = blocked_pattern


class SecurityViolationError(ToolError):
    """Raised when an operation violates security policy or contains threats."""

    def __init__(
        self,
        message: str = "Security policy violation detected.",
        *,
        tool_name: str = "SecurityGuard",
        threat_type: str = "general_threat",
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = {
            "threat_type": threat_type,
            **(details or {}),
        }
        super().__init__(
            message,
            tool_name=tool_name,
            code="SECURITY_VIOLATION",
            details=merged_details,
        )
        self.threat_type = threat_type
