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
from src.indexer import (
    CodeIndexer,
    HybridSearchEngine,
    create_default_embedding_client,
)
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

    def get_openai_tools_schema(self) -> list[dict[str, Any]]:
        """Generates OpenAI-compatible function calling schemas for registered tools."""
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": (
                            tool.parameters_schema
                            or {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return tools

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatches a tool call with schema validation."""
        start_time = time.perf_counter()
        tool_name = tool_call.tool_name

        if tool_name not in self._tools:
            duration = time.perf_counter() - start_time
            available = ", ".join(self.get_tool_names())
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=f"Unknown tool '{tool_name}'. Available tools are: [{available}]",
                error_code="UNKNOWN_TOOL",
                error_details={"available_tools": self.get_tool_names()},
                success=False,
                duration_seconds=duration,
            )

        tool_def = self._tools[tool_name]

        if tool_def.parameters_schema:
            try:
                import jsonschema

                jsonschema.validate(
                    instance=tool_call.arguments,
                    schema=tool_def.parameters_schema,
                )
            except jsonschema.ValidationError as val_err:
                duration = time.perf_counter() - start_time
                err_msg = (
                    f"Schema validation error for tool '{tool_name}': {val_err.message}"
                )
                logger.warning(
                    "tool_schema_validation_failed",
                    tool_name=tool_name,
                    error=val_err.message,
                    path=list(val_err.path),
                )
                return ToolResult(
                    tool_name=tool_name,
                    output="",
                    error=err_msg,
                    error_code="SCHEMA_VALIDATION_ERROR",
                    error_details={
                        "validation_error": val_err.message,
                        "path": list(val_err.path),
                        "validator": val_err.validator,
                    },
                    success=False,
                    duration_seconds=duration,
                )

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
                error_code=None,
                error_details={},
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
                error_code=getattr(exc, "code", "TOOL_ERROR"),
                error_details=getattr(exc, "details", {}),
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
                error_code="INVALID_ARGUMENTS",
                error_details={"error": str(exc)},
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
                f"Unexpected error executing tool '{tool_name}': "
                f"{type(exc).__name__}: {exc}"
            )
            return ToolResult(
                tool_name=tool_name,
                output="",
                error=err_msg,
                error_code="EXECUTION_ERROR",
                error_details={"error_type": type(exc).__name__, "message": str(exc)},
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

    indexer = CodeIndexer()

    def _get_symbol_definition(symbol_name: str, file_path: str | None = None) -> str:
        target_path = (sandbox_dir / file_path).resolve() if file_path else None
        if target_path and not str(target_path).startswith(str(sandbox_dir)):
            return f"Error: Path '{file_path}' is outside sandbox root."

        symbols = indexer.get_symbol_definition(symbol_name, file_path=target_path)
        if not symbols:
            indexer.index_directory(sandbox_dir)
            symbols = indexer.get_symbol_definition(symbol_name, file_path=target_path)

        if not symbols:
            return f"Symbol '{symbol_name}' not found."

        output_blocks = []
        for s in symbols:
            loc = (
                s.file_path.relative_to(sandbox_dir)
                if sandbox_dir in s.file_path.parents or s.file_path == sandbox_dir
                else s.file_path
            )
            block = (
                f"Symbol: {s.name} ({s.kind.value}) in {loc}:"
                f"{s.line_span.start_line}-{s.line_span.end_line}\n"
                f"Signature: {s.signature}"
            )
            if s.docstring:
                block += f"\nDocstring: {s.docstring}"
            output_blocks.append(block)
        return "\n\n".join(output_blocks)

    dispatcher.register_tool(
        name="get_symbol_definition",
        description="Extracts the AST definition and signature of a Python symbol.",
        handler=_get_symbol_definition,
        parameters_schema={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Symbol name to lookup (e.g. 'WorkspaceManager').",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional relative path to Python file.",
                },
            },
            "required": ["symbol_name"],
        },
    )

    def _list_file_structure(path: str) -> str:
        target = (sandbox_dir / path).resolve()
        if not str(target).startswith(str(sandbox_dir)):
            return f"Error: Path '{path}' is outside sandbox root."
        if not target.exists() or not target.is_file():
            return f"Error: File '{path}' does not exist or is not a file."

        structure = indexer.list_file_structure(target)
        lines = [f"File: {path}"]
        if structure.imports:
            lines.append(f"Imports ({len(structure.imports)}):")
            for imp in structure.imports:
                lines.append(f"  - {imp.statement} (L{imp.line_span.start_line})")
        if structure.classes:
            lines.append(f"Classes ({len(structure.classes)}):")
            for cls in structure.classes:
                span_str = f"L{cls.line_span.start_line}-L{cls.line_span.end_line}"
                lines.append(f"  - {cls.signature} ({span_str})")
                for m in cls.methods:
                    lines.append(f"      * {m.signature} (L{m.line_span.start_line})")
        if structure.functions:
            lines.append(f"Functions ({len(structure.functions)}):")
            for fn in structure.functions:
                span_str = f"L{fn.line_span.start_line}-L{fn.line_span.end_line}"
                lines.append(f"  - {fn.signature} ({span_str})")
        return "\n".join(lines)

    dispatcher.register_tool(
        name="list_file_structure",
        description="Returns structural overview of a Python file.",
        handler=_list_file_structure,
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the Python source file.",
                }
            },
            "required": ["path"],
        },
    )

    search_engine = HybridSearchEngine(
        indexer=indexer,
        embedding_client=create_default_embedding_client(),
    )

    async def _semantic_search(query: str, top_k: int = 5) -> str:
        await search_engine.index_directory(sandbox_dir)
        results = await search_engine.search_semantic(query, top_k=top_k)
        if not results:
            return f"No semantic matches found for '{query}'."
        blocks = []
        for r in results:
            loc = (
                r.chunk.file_path.relative_to(sandbox_dir)
                if (
                    sandbox_dir in r.chunk.file_path.parents
                    or r.chunk.file_path == sandbox_dir
                )
                else r.chunk.file_path
            )
            span_str = f"{r.chunk.line_span.start_line}-{r.chunk.line_span.end_line}"
            block = (
                f"[{r.match_type.value.upper()} | Score: {r.score:.3f}] "
                f"{r.chunk.symbol_name} ({r.chunk.kind.value}) in {loc}:{span_str}\n"
                f"Signature: {r.chunk.signature}"
            )
            if r.chunk.docstring:
                block += f"\nDocstring: {r.chunk.docstring}"
            blocks.append(block)
        return "\n\n".join(blocks)

    dispatcher.register_tool(
        name="semantic_search",
        description="Performs semantic vector search over Python code chunks.",
        handler=_semantic_search,
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing desired code.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    )

    async def _hybrid_search(query: str, top_k: int = 5) -> str:
        await search_engine.index_directory(sandbox_dir)
        results = await search_engine.search_hybrid(query, top_k=top_k)
        if not results:
            return f"No matches found for '{query}'."
        blocks = []
        for r in results:
            loc = (
                r.chunk.file_path.relative_to(sandbox_dir)
                if (
                    sandbox_dir in r.chunk.file_path.parents
                    or r.chunk.file_path == sandbox_dir
                )
                else r.chunk.file_path
            )
            span_str = f"{r.chunk.line_span.start_line}-{r.chunk.line_span.end_line}"
            block = (
                f"[{r.match_type.value.upper()} | Score: {r.score:.3f}] "
                f"{r.chunk.symbol_name} ({r.chunk.kind.value}) in {loc}:{span_str}\n"
                f"Signature: {r.chunk.signature}"
            )
            if r.chunk.docstring:
                block += f"\nDocstring: {r.chunk.docstring}"
            blocks.append(block)
        return "\n\n".join(blocks)

    dispatcher.register_tool(
        name="hybrid_search",
        description="Performs hybrid exact symbol & semantic code search.",
        handler=_hybrid_search,
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name or natural language description.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    )

    return dispatcher
