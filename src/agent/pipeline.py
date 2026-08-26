from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from src.agent.dispatcher import ToolDispatcher
from src.agent.llm_client import BaseLLMClient
from src.agent.loop import ReasoningLoop
from src.agent.models import ReasoningTrajectory
from src.api.schemas.events import AgentEvent
from src.indexer.chunker import ASTChunker
from src.indexer.embeddings import EmbeddingClient, create_default_embedding_client
from src.indexer.indexer import CodeIndexer
from src.indexer.search import HybridSearchEngine
from src.indexer.vector_store import InMemoryVectorStore, PGVectorStore
from src.tools.exceptions import ToolError
from src.tools.filesystem import FileSystemTool
from src.workspace.manager import WorkspaceManager

logger = structlog.get_logger(__name__)


def create_workspace_dispatcher(
    workspace_manager: WorkspaceManager,
    workspace_id: str,
    workspace_path: Path,
    embedding_client: EmbeddingClient | None = None,
    database_session_factory: Any = None,
) -> tuple[ToolDispatcher, HybridSearchEngine]:
    """Creates a ToolDispatcher wired to a Docker sandbox and code indexer."""
    dispatcher = ToolDispatcher()
    fs_tool = FileSystemTool(workspace_path)

    # 1. Register Filesystem Tools (constrained to workspace)
    async def _read_file(path: str) -> str:
        return str(fs_tool.read_file(path))

    dispatcher.register_tool(
        name="read_file",
        handler=_read_file,
        description="Reads the complete text content of a file within the workspace.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace",
                }
            },
            "required": ["path"],
        },
    )

    async def _write_file(path: str, content: str) -> str:
        fs_tool.write_file(path, content)
        return f"Successfully wrote to '{path}'."

    dispatcher.register_tool(
        name="write_file",
        handler=_write_file,
        description="Writes content to a file within the workspace.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace",
                },
                "content": {
                    "type": "string",
                    "description": "The exact content to write",
                },
            },
            "required": ["path", "content"],
        },
    )

    async def _list_dir(path: str = ".") -> str:
        safe_dir = fs_tool._resolve_safe_path(path)
        if not safe_dir.exists() or not safe_dir.is_dir():
            return "(empty or non-existent directory)"
        entries = [p.name for p in sorted(safe_dir.iterdir())]
        return "\n".join(entries) if entries else "(empty directory)"

    dispatcher.register_tool(
        name="list_dir",
        handler=_list_dir,
        description="Lists all files and subdirectories within a directory path.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path (defaults to '.')",
                }
            },
        },
    )

    # 2. Register Docker Sandbox Shell Tool
    async def _run_shell(command: str, timeout_s: float = 30.0) -> str:
        output_lines: list[str] = []
        is_timeout = False
        is_truncated = False
        async for line in workspace_manager.execute_command(
            workspace_id=workspace_id,
            cmd=command,
            timeout_s=timeout_s,
        ):
            if line.is_sentinel:
                if str(line.sentinel_type).upper() == "TIMEOUT":
                    is_timeout = True
                elif str(line.sentinel_type).upper() == "TRUNCATED":
                    is_truncated = True
            else:
                output_lines.append(line.line)

        full_output = "\n".join(output_lines)
        if is_timeout:
            return (
                f"[Command timed out after {timeout_s}s]\n{full_output.strip()}"
            ).strip()
        if is_truncated:
            return f"[Output truncated]\n{full_output.strip()}".strip()
        return full_output.strip() or "(command completed with no output)"

    dispatcher.register_tool(
        name="run_shell",
        handler=_run_shell,
        description="Executes a shell command inside the Docker sandbox.",
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute in sandbox container",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Maximum execution time in seconds (default: 30.0)",
                },
            },
            "required": ["command"],
        },
    )

    # 3. Setup AST Indexer & Vector Store for Workspace
    indexer = CodeIndexer()
    client = embedding_client or create_default_embedding_client()

    vector_store = (
        PGVectorStore(database_session_factory=database_session_factory)
        if database_session_factory is not None
        else InMemoryVectorStore()
    )
    chunker = ASTChunker()
    search_engine = HybridSearchEngine(
        indexer=indexer,
        embedding_client=client,
        vector_store=vector_store,
        chunker=chunker,
    )

    # 4. Register Structural & Semantic Search Tools
    async def _get_symbol_definition(name: str) -> str:
        indexer.index_directory(workspace_path)
        symbols = indexer.get_symbol_definition(name)
        if not symbols:
            raise ToolError(
                message=f"Symbol '{name}' not found in workspace.",
                tool_name="get_symbol_definition",
            )
        blocks = []
        for s in symbols:
            rel = (
                s.file_path.relative_to(workspace_path)
                if workspace_path in s.file_path.parents
                else s.file_path
            )
            header = (
                f"--- Symbol: {s.name} ({s.kind.value}) in "
                f"{rel}:{s.line_span.start_line} ---"
            )
            blocks.append(
                f"{header}\n"
                f"Signature: {s.signature}\n"
                f"Lines: {s.line_span.start_line}-{s.line_span.end_line}"
            )

        return "\n\n".join(blocks).strip()

    dispatcher.register_tool(
        name="get_symbol_definition",
        handler=_get_symbol_definition,
        description="Looks up definition and signature by symbol name.",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact class or function name to locate",
                }
            },
            "required": ["name"],
        },
    )

    async def _list_file_structure(path: str) -> str:
        target_path = (workspace_path / path).resolve()
        if not target_path.exists():
            raise ToolError(
                message=f"File '{path}' does not exist.",
                tool_name="list_file_structure",
            )
        structure = indexer.list_file_structure(target_path)
        if structure is None:
            raise ToolError(
                message=f"Failed to parse AST structure for '{path}'.",
                tool_name="list_file_structure",
            )
        imports_str = ", ".join(i.statement for i in structure.imports) or "None"
        symbols_str = (
            ", ".join(
                f"{s.name} ({s.kind.value}, L{s.line_span.start_line})"
                for s in structure.symbols.values()
            )
            or "None"
        )
        return (
            f"File: {path}\n"
            f"Module Imports: {imports_str}\n"
            f"Defined Symbols: {symbols_str}"
        )

    dispatcher.register_tool(
        name="list_file_structure",
        handler=_list_file_structure,
        description="Returns an outline of imports, functions, and classes.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the Python source file",
                }
            },
            "required": ["path"],
        },
    )

    async def _semantic_search(query: str, top_k: int = 5) -> str:
        await search_engine.index_directory(workspace_path)
        results = await search_engine.search_semantic(query, top_k=top_k)
        if not results:
            return f"No semantic matches found for '{query}'."
        blocks = []
        for r in results:
            loc = (
                r.chunk.file_path.relative_to(workspace_path)
                if workspace_path in r.chunk.file_path.parents
                else r.chunk.file_path
            )
            span_str = f"{r.chunk.line_span.start_line}-{r.chunk.line_span.end_line}"
            block = (
                f"[{r.match_type.value.upper()} | Score: {r.score:.3f}] "
                f"{loc}:{span_str} ({r.chunk.symbol_name})\n"
                f"Signature: {r.chunk.signature}\n"
                f"Code:\n{r.chunk.code_content}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    dispatcher.register_tool(
        name="semantic_search",
        handler=_semantic_search,
        description="Performs natural language vector search over code.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing functionality",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum matching chunks to return (default: 5)",
                },
            },
            "required": ["query"],
        },
    )

    async def _hybrid_search(query: str, top_k: int = 5) -> str:
        await search_engine.index_directory(workspace_path)
        results = await search_engine.search_hybrid(query, top_k=top_k)
        if not results:
            return f"No hybrid matches found for '{query}'."
        blocks = []
        for r in results:
            loc = (
                r.chunk.file_path.relative_to(workspace_path)
                if workspace_path in r.chunk.file_path.parents
                else r.chunk.file_path
            )
            span_str = f"{r.chunk.line_span.start_line}-{r.chunk.line_span.end_line}"
            block = (
                f"[{r.match_type.value.upper()} | Score: {r.score:.3f}] "
                f"{loc}:{span_str} ({r.chunk.symbol_name})\n"
                f"Signature: {r.chunk.signature}\n"
                f"Code:\n{r.chunk.code_content}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    dispatcher.register_tool(
        name="hybrid_search",
        handler=_hybrid_search,
        description="Hybrid search: exact symbol match with semantic fallback.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name or natural language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                },
            },
            "required": ["query"],
        },
    )

    return dispatcher, search_engine


async def execute_workspace_task(
    task: str,
    repo_url: str,
    workspace_manager: WorkspaceManager,
    llm_client: BaseLLMClient,
    max_steps: int = 25,
    on_event: Callable[[AgentEvent], Any] | None = None,
    task_id: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    database_session_factory: Any = None,
) -> tuple[ReasoningTrajectory, str]:
    """Clones repo into sandbox, indexes codebase, and runs agent loop."""

    workspace = workspace_manager.create(repo_url)
    workspace_id = workspace.workspace_id
    workspace_path = workspace.host_dir

    logger.info(
        "workspace_task_started",
        workspace_id=workspace_id,
        repo_url=repo_url,
        task=task,
    )

    try:
        dispatcher, search_engine = create_workspace_dispatcher(
            workspace_manager=workspace_manager,
            workspace_id=workspace_id,
            workspace_path=workspace_path,
            embedding_client=embedding_client,
            database_session_factory=database_session_factory,
        )

        # Index codebase upfront
        await search_engine.index_directory(workspace_path)

        loop = ReasoningLoop(
            llm_client=llm_client,
            dispatcher=dispatcher,
            max_steps=max_steps,
        )

        trajectory = await loop.run(
            task=task,
            metadata={"workspace_id": workspace_id, "repo_url": repo_url},
            on_event=on_event,
            task_id=task_id,
        )

        return trajectory, workspace_id
    finally:
        logger.info(
            "workspace_task_destroying_sandbox",
            workspace_id=workspace_id,
        )
        workspace_manager.destroy(workspace_id)
