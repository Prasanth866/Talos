# Talos

Autonomous software engineering agent framework featuring a tool-use reasoning loop, bounded async worker queues with structured concurrency, graceful shutdown, persistent task state with crash recovery, Docker-based isolated task workspaces, defensive sandboxing, real-time WebSocket event streaming, and a zero-dependency aesthetic web UI.

---

## Key Features

- **Autonomous Reasoning Loop**: Step-by-step ReAct-style agent execution loop (`ReasoningLoop`) with dynamic tool dispatching, compact observation limits, adaptive rate-limit (429) backoff, and token/cost tracking.
- **`tree-sitter` AST Code Indexing**: Incremental, fault-tolerant Python AST indexer (`PythonASTParser`, `CodeIndexer`) extracting function/class signatures, docstrings, typed arguments, inheritance hierarchies, and imports with sub-millisecond per-file latency, exposed directly to agent reasoning via `get_symbol_definition` and `list_file_structure` tools.
- **Docker Workspace Hardening & Isolation**: Task-isolated container environments (`WorkspaceManager`) with shallow Git repository cloning (`--depth 1`), custom volume mounting, memory caps (`512MB`), CPU quotas, process limits (`pids_limit=256`), read-only root filesystems, air-gapped network isolation (`network_mode="none"`), and non-root execution (`user="1000:1000"`).

- **Async Streaming Command Execution**: Async generator command runner (`execute_command`) yielding line streams in real-time, enforcing a 1MB output cap (`[TRUNCATED]` sentinel), timeout enforcement (`[TIMEOUT]` sentinel), and background process tree termination.
- **Adversarial Security Test Suite**: Comprehensive security test suite (`tests/workspace/test_security.py`) actively attempting network escape, filesystem escape, fork bombs, memory exhaustion, and path traversal to prove container isolation.
- **Async Worker Queue & Pool**: Structured concurrency managed via `asyncio.TaskGroup` over a bounded `asyncio.Queue` with configurable worker concurrency.
- **Persistent Task Store**: Async database persistence using SQLAlchemy and Alembic, tracking complete task lifecycles (`PENDING -> RUNNING -> COMPLETED / FAILED`), token counts, USD cost, and final answers.
- **Crash Recovery**: Automatic startup recovery identifying any interrupted `RUNNING` tasks from crashes or abrupt server kills and marking them as `FAILED`.
- **Task Query & Filtering**: REST endpoints (`POST /tasks`, `GET /tasks/{id}`, `GET /tasks?status=...`, `DELETE /tasks/{id}`, `DELETE /tasks`) with pagination and status filtering.
- **Resilient Retries with Full Jitter**: Tenacity-powered async and sync retry utilities with true randomized exponential backoff (`wait_random_exponential`), parameter validation, defensive HTTP status parsing, and structured `structlog` before-sleep logging with correlation ID preservation.
- **Backpressure & Load Shedding**: Explicit HTTP `503 Service Unavailable` with `Retry-After` headers when the task queue reaches maximum capacity or during shutdown.
- **Graceful Shutdown**: Intercepts `SIGTERM` / application lifespan exit, safely stops accepting new submissions, and drains in-flight tasks before termination.
- **Correlation IDs**: Full UUID `task_id` propagation threaded through every structured log line via `structlog` contextvars and all WebSocket stream events.
- **WebSocket Event Streaming**: Real-time subscriber endpoint (`/ws?task_id=<uuid>`) streaming versioned events (`thought`, `tool_call`, `tool_output`, `task_complete`, `error`) with event replay support.
- **Defensive Tool Sandboxing**: Secure `FileSystemTool` (path traversal prevention) and `ShellTool` (command execution with timeouts, output limits, and process cleanup).
- **Lovable & Replit AI Aesthetic Web UI**: Pure HTML, CSS, and Vanilla JavaScript single-page application with zero Node.js/npm dependencies, served directly by FastAPI. Features real-time streaming event cards, floating composer dock, live terminal drawer, and history management.

---

## Quick Start

### Prerequisites
- Python >= 3.14
- [`uv`](https://github.com/astral-sh/uv)
- Docker Desktop or Docker Engine (for workspace container isolation)

### Installation
```bash
# Clone the repository
git clone https://github.com/Prasanth866/Talos.git
cd Talos

# Install dependencies and create virtual environment
uv sync
```

### Database Migrations
```bash
# Apply migrations to initialize or upgrade the database
uv run alembic upgrade head
```

### Environment Configuration
Copy `.env.example` to `.env` and configure settings as needed:
```bash
cp .env.example .env
```

Key environment variables:
- `DATABASE_URL`: Database connection URL (default: `sqlite+aiosqlite:///talos.db`).
- `LLM_API_KEY`: API key for LLM provider (e.g. Groq or OpenAI-compatible endpoint; uses Mock LLM if empty).
- `LLM_BASE_URL`: Base URL for OpenAI-compatible LLM endpoint (default: `https://api.groq.com/openai/v1`).
- `LLM_MODEL`: Model name (default: `openai/gpt-oss-120b`).
- `WORKER_CONCURRENCY`: Number of concurrent async workers in the pool (default: `4`).
- `TASK_QUEUE_MAX_SIZE`: Maximum bounded task queue size before backpressure (default: `100`).
- `SHUTDOWN_DRAIN_TIMEOUT_SECONDS`: Maximum seconds to wait for in-flight tasks during graceful shutdown (default: `30.0`).
- `WORKSPACE_MEM_LIMIT`: Memory limit for sandbox containers (default: `512m`).
- `WORKSPACE_PIDS_LIMIT`: Maximum concurrent process/thread count per container (default: `256`).

### Running Locally

```bash
uv run fastapi dev src/main.py
```
- **Web UI & Live Agent Workspace**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

*(Note: The static frontend can also be viewed via VS Code Live Server at `http://127.0.0.1:5500/` or opened directly as a file; API and WebSocket requests automatically route to the backend on port 8000).*

---

## Docker Workspace Hardening & Streaming Execution

Talos provides isolated, hardened task workspaces where each task runs inside its own Docker container with a shallow-cloned Git repository and layered defense-in-depth:

```python
from src.workspace import ContainerSecurityConfig, WorkspaceManager

# Initialize manager with default or customized hardening
manager = WorkspaceManager(
    default_security_config=ContainerSecurityConfig(
        mem_limit="512m",
        pids_limit=256,
        read_only=True,
        network_mode="none",
        user="1000:1000",
    )
)

# Create an isolated workspace with a shallow clone
workspace = manager.create(
    repo_url="https://github.com/octocat/Hello-World.git",
    commit_sha=None,  # or specific commit SHA
    image="python:3.12-slim",
)

# 1. Synchronous command execution
result = manager.run_command(
    workspace.workspace_id, ["python3", "-c", "print('Hello from hardened container')"]
)
print(result["stdout"])

# 2. Asynchronous streaming command execution with output capping & timeout
async for output_line in manager.execute_command(
    workspace.workspace_id,
    "python3 -c 'for i in range(100): print(f\"step {i}\")'",
    timeout_s=10.0,
    max_output_bytes=1024 * 1024,  # 1MB cap
):
    print(f"[{output_line.stream}] {output_line.line}")
    if output_line.is_sentinel:
        print(f"Sentinel triggered: {output_line.sentinel_type}")

# List files inside workspace
files = manager.list_files(workspace.workspace_id)

# Teardown container and delete host directory cleanly
manager.destroy(workspace.workspace_id)
```

### Container Hardening Controls
- **Memory Ceiling (`mem_limit="512m"`)**: Cgroups terminate memory-hungry runaway processes with `SIGKILL` (exit code 137).
- **Process Ceiling (`pids_limit=256`)**: Cgroups block fork bomb attacks instantly with `BlockingIOError: [Errno 11] Resource temporarily unavailable`.
- **Read-Only Root Filesystem (`read_only=True`)**: Prevents modification of system binaries or persistence injection, with writes restricted to `/workspace` and a bounded `tmpfs` at `/tmp`.
- **Network Isolation (`network_mode="none"`)**: Air-gaps the container namespace, preventing exfiltration or unauthorized outbound traffic.
- **Unprivileged Execution (`user="1000:1000"`)**: Containers execute under a non-root UID/GID.

### Typed Error Hierarchy
All Docker and Git operations are strictly wrapped so no low-level raw exceptions leak to caller code:
- `WorkspaceError`: Base class providing structured `.to_dict()` outputs.
- `DockerDaemonError`: Unreachable or stopped Docker daemon.
- `WorkspaceCreationError`: Container initialization or volume mount failure.
- `WorkspaceNotFoundError`: Workspace ID not registered.
- `WorkspaceDestroyError`: Container teardown or cleanup failure.
- `GitCloneError`: Git clone or commit checkout failure.
- `WorkspaceExecutionError`: Command execution failure inside the container.


---

## `tree-sitter` AST Code Indexing

Talos includes a structural code indexer built on `tree-sitter` and `tree-sitter-python`. It parses source code into resilient concrete syntax trees, extracting detailed symbol definitions, signatures, line spans, docstrings, class inheritance hierarchies, and imports.

```python
from pathlib import Path
from src.indexer import CodeIndexer, SymbolKind

# Initialize in-memory indexer
indexer = CodeIndexer()

# 1. Index an entire codebase recursively
indexed_files = indexer.index_directory(Path("src"), recursive=True)
print(f"Indexed {indexed_files} Python source files")

# 2. Lookup symbol definitions across the repository or in a specific file
symbols = indexer.get_symbol_definition("WorkspaceManager")
for sym in symbols:
    print(f"Found: {sym.signature} (L{sym.line_span.start_line}-L{sym.line_span.end_line})")
    print(f"Docstring: {sym.docstring}")

# 3. Retrieve high-level file structure (imports, classes, methods, functions)
structure = indexer.list_file_structure("src/workspace/manager.py")
print(f"Imports: {len(structure.imports)}")
print(f"Classes: {[c.name for c in structure.classes]}")
print(f"Functions: {[f.name for f in structure.functions]}")
print(f"Syntax Errors Detected: {structure.has_syntax_errors}")

# 4. Search symbols by substring or fuzzy match
matches = indexer.search_symbols("execute_command")
for m in matches:
    print(f"Match: {m.name} -> {m.signature}")
```

### Agent Tool Dispatcher Integration
The reasoning loop has built-in access to the indexer via agent tools:
- `get_symbol_definition(symbol_name, file_path=None)`: Retrieves AST signature, line spans, and docstring of any class, method, or function.
- `list_file_structure(path)`: Returns a high-level architectural outline of any Python file (imports, classes, member methods, standalone functions).

### Resilient Error Recovery
Unlike Python's standard `ast.parse()` which raises an unrecoverable `SyntaxError` on partial code edits, `tree-sitter` creates localized `ERROR` nodes and continues indexing all valid sibling definitions before and after the syntax error.

---

## Web UI Overview

The web client is built with zero runtime or build dependencies (pure HTML5, CSS3, and modern JavaScript) and features:

1. **Lovable / Replit AI Dark Aesthetic**:
   - Deep obsidian background (`#09090b`), amber/orange accents (`#f97316`), and glassmorphism borders (`rgba(255, 255, 255, 0.08)`).
2. **Floating Composer Dock**:
   - Compact centered prompt composer with auto-expanding textarea, shortcut dispatch (`Cmd+Enter` / `Ctrl+Enter`), and glowing submit button.
3. **Agent Reasoning & Observation Stream**:
   - Real-time step-by-step accordion cards displaying agent thoughts, tool action badges, syntax-highlighted outputs, and final accomplishments.
4. **Live WebSocket Terminal**:
   - Color-coded log stream with filter pills (`All`, `Thoughts`, `Tools`, `Outputs`), live search, copy, and clear controls.
5. **Task History & Metrics HUD**:
   - Left sidebar with task search, status badges, deletion, and database clear action.
   - Top HUD displaying real-time **Steps**, **Tokens**, **USD Cost**, **Duration**, and View Mode switcher (**Split** / **Agent** / **Terminal**).

---

## API Workflow

### 1. Submit a Task
```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze code and fix failing tests", "metadata": {"priority": "high"}}'
```
**Response (202 Accepted)**:
```json
{
  "task_id": "c6a1b2c3-d4e5-4f6a-8b9c-0d1e2f3a4b5c",
  "status": "PENDING",
  "ws_url": "/ws?task_id=c6a1b2c3-d4e5-4f6a-8b9c-0d1e2f3a4b5c"
}
```
*(If the queue is full or server is shutting down, returns `503 Service Unavailable` with `Retry-After: 5`)*

### 2. Stream Real-Time Events over WebSocket
Connect to `/ws?task_id=<task_id>`:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws?task_id=c6a1b2c3-d4e5-4f6a-8b9c-0d1e2f3a4b5c");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.event_type}]`, data);
};
```

### 3. Query Task Status and Results
```bash
# Fetch single task details
curl http://localhost:8000/tasks/c6a1b2c3-d4e5-4f6a-8b9c-0d1e2f3a4b5c

# Filter completed tasks with pagination
curl "http://localhost:8000/tasks?status=COMPLETED&limit=10"

# Clear task history
curl -X DELETE http://localhost:8000/tasks
```

---

## Running Tests & Quality Checks

```bash
# Run all unit tests across the codebase
uv run python tests/run_tests.py

# Run tree-sitter AST parser & indexer unit tests
uv run pytest tests/indexer/test_parser.py tests/indexer/test_indexer.py -v

# Run adversarial security test suite (network, fs, fork, OOM, traversal)
uv run pytest tests/workspace/test_security.py -v -s

# Run live Docker workspace experiments
uv run pytest tests/workspace/test_experiment.py -v -s

# Run strict type checking
uv run mypy src tests migrations

# Run linter & code formatter checks
uv run ruff check .
uv run ruff format --check .
```

---

## Project Structure

```
Talos/
├── .env.example                  # Environment variable template
├── .pre-commit-config.yaml       # Pre-commit hooks (ruff, mypy, pytest)
├── alembic.ini                   # Alembic database migration configuration
├── migrations/                   # Asynchronous database migration versions
├── pyproject.toml                # Project metadata & tool configurations
├── static/                       # Zero-dependency frontend
│   ├── app.js                    # Vanilla JS WebSocket streaming & API client
│   ├── index.html                # Single-page application shell
│   └── style.css                 # Lovable & Replit AI dark theme stylesheet
├── src/
│   ├── main.py                   # FastAPI app, database lifespan & crash recovery
│   ├── agent/
│   │   ├── dispatcher.py         # Tool registry and execution dispatcher (with AST tools)
│   │   ├── llm_client.py         # HTTP (OpenAI-compatible) and Mock LLM clients
│   │   ├── loop.py               # ReAct reasoning loop coordinator
│   │   ├── models.py             # Trajectory, Step, TokenUsage & Message models
│   │   ├── prompts.py            # System prompts & tool documentation formatting
│   │   ├── retry.py              # Exponential backoff, jitter & Tenacity retry utilities
│   │   └── token_tracker.py      # Token counting & dollar cost tracking
│   ├── api/
│   │   ├── exception_handlers.py # Standardized JSON error response handlers
│   │   ├── routes/
│   │   │   ├── health.py         # /health (liveness) & /readiness endpoints
│   │   │   ├── tasks.py          # POST /tasks, GET /tasks/{id}, GET /tasks, DELETE /tasks
│   │   │   └── websocket.py      # /ws subscriber streaming endpoint
│   │   └── schemas/
│   │       └── events.py         # Streaming events & TaskDetailResponse schemas
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment validation
│   │   ├── database.py           # Async SQLAlchemy engine & session factory
│   │   ├── logging.py            # Structlog configuration with contextvars
│   │   ├── middleware.py         # Correlation ID & request duration middleware
│   │   └── worker.py             # TaskManager, TaskGroup worker pool & graceful drain
│   ├── db/
│   │   ├── models.py             # Task SQLAlchemy ORM model & TaskStatus enum
│   │   └── repository.py         # Pure-function asynchronous data access layer
│   ├── indexer/
│   │   ├── indexer.py            # CodeIndexer multi-file index & symbol query API
│   │   ├── models.py             # Symbol, FunctionDefinition, ClassDefinition dataclasses
│   │   └── parser.py             # PythonASTParser using tree-sitter Python grammars
│   ├── tools/
│   │   ├── exceptions.py         # Custom ToolError exception hierarchy
│   │   ├── filesystem.py         # Sandboxed FileSystemTool & path safety
│   │   ├── shell.py              # Defensive ShellTool with timeout & process cleanup
│   │   └── system_tools.py       # Backward-compatible re-exports
│   └── workspace/
│       ├── exceptions.py         # Typed WorkspaceError exception hierarchy
│       ├── git_utils.py          # Shallow clone & commit resolution with GitPython
│       ├── manager.py            # WorkspaceManager container lifecycle controller
│       └── models.py             # Workspace & ContainerSecurityConfig dataclasses
└── tests/
    ├── conftest.py               # Shared test fixtures & client lifecycle
    ├── run_tests.py              # Isolated test runner executing all module suites
    ├── agent/                    # Agent loop, dispatcher, LLM & retry tests
    ├── api/                      # API endpoint, WebSocket, task queries & schema tests
    ├── core/                     # Config, logging, middleware & worker pool tests
    ├── db/                       # Repository and crash recovery test suites
    ├── fixtures/sample_repo/     # Fixture repository for AST parser and indexer testing
    ├── indexer/                  # AST parser, indexer lookup & experiment tests
    ├── tools/                    # File system & shell tool test suites
    └── workspace/                # Manager unit tests, live experiments & security tests
```

---

## License

MIT

