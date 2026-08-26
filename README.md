# Talos

Autonomous software engineering agent framework featuring a tool-use reasoning loop, bounded async worker queues with structured concurrency, graceful shutdown, persistent task state with crash recovery, Docker-based isolated task workspaces, defensive sandboxing, native Python AST code indexing with syntax-error recovery, pgvector-powered semantic & hybrid code search, real-time WebSocket event streaming, and a zero-dependency aesthetic web UI.

---

## Key Features

- **Autonomous Reasoning Loop**: Step-by-step ReAct-style agent execution loop (`ReasoningLoop`) with dynamic tool dispatching, compact observation limits, adaptive rate-limit (429) backoff, and token/cost tracking.
- **End-to-End Sandbox + Indexer Pipeline**: Full lifecycle orchestration (`execute_workspace_task`) from task submission -> Git clone -> Docker container sandbox boot -> upfront AST & vector indexing -> agent code search -> in-container test execution -> guaranteed workspace teardown.
- **Native Python AST Code Indexing**: Incremental, fault-tolerant Python AST indexer (`PythonASTParser`, `CodeIndexer`) extracting function/class signatures, docstrings, typed arguments, inheritance hierarchies, and imports with sub-millisecond per-file latency and syntax-error recovery.
- **AST Semantic Chunking**: Chunking engine (`ASTChunker`) that slices code along syntax boundaries (complete functions, classes, and member methods) rather than arbitrary token cuts, preserving signatures, docstrings, decorators, and line spans.
- **Dense Code Embeddings (Gemini & OpenAI)**: Embeddings integration supporting Google Gemini (`gemini-embedding-001`, 3072 dimensions) and OpenAI (`text-embedding-3-small` / `large`), with batched requests and per-indexing token/cost tracking.
- **PostgreSQL `pgvector` & In-Memory Vector Store**: Vector database support (`PGVectorStore`) utilizing the `<=>` cosine distance operator in PostgreSQL, with resilient fallback to `InMemoryVectorStore` for local development and SQLite testing.
- **Hybrid Code Search Engine**: Multi-tier search (`HybridSearchEngine`) that boosts exact symbol and signature matches (`exact_boost=1.3`) for precise identifiers, seamlessly falling back to dense vector embeddings for natural language behavioral queries.
- **Docker Workspace Hardening & Isolation**: Task-isolated container environments (`WorkspaceManager`) with shallow Git repository cloning (`--depth 1`), custom volume mounting, memory caps (`512MB`), CPU quotas, process limits (`pids_limit=256`), read-only root filesystems, air-gapped network isolation (`network_mode="none"`), and non-root execution (`user="1000:1000"`).
- **Async Streaming Command Execution**: Async generator command runner (`execute_command`) yielding line streams in real-time, enforcing a 1MB output cap (`[TRUNCATED]` sentinel), timeout enforcement (`[TIMEOUT]` sentinel), and background process tree termination.
- **Adversarial Security Test Suite**: Comprehensive security test suite (`tests/workspace/test_security.py`) actively attempting network escape, filesystem escape, fork bombs, memory exhaustion, and path traversal to prove container isolation.
- **Async Worker Queue & Pool**: Structured concurrency managed via `asyncio.TaskGroup` over a bounded `asyncio.Queue` with configurable worker concurrency and automatic task routing.
- **Persistent Task Store**: Async database persistence using SQLAlchemy and Alembic, tracking complete task lifecycles (`PENDING -> RUNNING -> COMPLETED / FAILED`), token counts, USD cost, and final answers.
- **Crash Recovery**: Automatic startup recovery identifying any interrupted `RUNNING` tasks from crashes or abrupt server kills and marking them as `FAILED`.
- **Task & Code Search REST Endpoints**: Endpoints for task management (`/tasks`), semantic search (`POST /search/semantic`), and hybrid search (`POST /search/hybrid`).
- **Resilient Retries with Full Jitter**: Tenacity-powered async and sync retry utilities with true randomized exponential backoff (`wait_random_exponential`), parameter validation, and structured `structlog` context logging.
- **Backpressure & Load Shedding**: Explicit HTTP `503 Service Unavailable` with `Retry-After` headers when the task queue reaches maximum capacity or during shutdown.
- **Graceful Shutdown**: Intercepts `SIGTERM` / application lifespan exit, safely stops accepting new submissions, and drains in-flight tasks before termination.
- **WebSocket Event Streaming**: Real-time subscriber endpoint (`/ws?task_id=<uuid>`) streaming versioned events (`thought`, `tool_call`, `tool_output`, `task_complete`, `error`) with event replay support.
- **Defensive Tool Sandboxing**: Secure `FileSystemTool` (path traversal prevention) and `ShellTool` (command execution with timeouts, output limits, and process cleanup).
- **Lovable & Replit AI Aesthetic Web UI**: Pure HTML, CSS, and Vanilla JavaScript single-page application with zero Node.js/npm dependencies, served directly by FastAPI. Features real-time streaming event cards, floating composer dock, live terminal drawer, and history management.

---

## Quick Start

### Prerequisites
- Python >= 3.14
- [`uv`](https://github.com/astral-sh/uv)
- Docker Desktop or Docker Engine (for workspace container isolation)
- PostgreSQL with `pgvector` extension (optional; falls back to SQLite + In-Memory vectors)

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
Copy `.env.example` to `.env` and configure settings:
```bash
cp .env.example .env
```

Key environment variables:
- `DATABASE_URL`: PostgreSQL with asyncpg or SQLite (e.g. `sqlite+aiosqlite:///talos.db` or `postgresql+asyncpg://postgres:postgres@localhost:5432/talos_db`).
- `LLM_API_KEY`: API key for LLM provider (e.g. Groq or OpenAI-compatible endpoint; uses Mock LLM if empty).
- `LLM_BASE_URL`: Base URL for OpenAI-compatible LLM endpoint (default: `https://api.groq.com/openai/v1`).
- `LLM_MODEL`: Model name (default: `openai/gpt-oss-120b`).
- `GEMINI_API_KEY`: API key for Google Gemini embedding client (`gemini-embedding-001`; uses Mock Embeddings if empty).
- `EMBEDDING_MODEL`: Embedding model identifier (default: `gemini-embedding-001`).
- `EMBEDDING_DIMENSION`: Embedding vector dimensionality (default: `3072`).
- `WORKER_CONCURRENCY`: Number of concurrent async workers in the pool (default: `4`).
- `TASK_QUEUE_MAX_SIZE`: Maximum bounded task queue size before backpressure (default: `100`).
- `WORKSPACE_MEM_LIMIT`: Memory limit for sandbox containers (default: `512m`).
- `WORKSPACE_PIDS_LIMIT`: Maximum concurrent process/thread count per container (default: `256`).
- `WORKSPACE_READ_ONLY`: Sandbox read-only root filesystem (default: `true`).
- `WORKSPACE_NETWORK_MODE`: Sandbox network isolation (default: `none`).
- `WORKSPACE_USER`: Non-root execution UID/GID (default: `1000:1000`).

### Running Locally

```bash
uv run fastapi dev src/main.py
```
- **Web UI & Live Agent Workspace**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Full Agent Pipeline & Sandbox Integration

Talos wires together workspace creation, upfront AST & vector indexing, agent code navigation, and sandbox test execution into a seamless automated loop:

```mermaid
sequenceDiagram
    participant Worker as TaskManager / Worker
    participant Workspace as WorkspaceManager
    participant Sandbox as Docker Container
    participant Indexer as AST / Vector Indexer
    participant Agent as ReasoningLoop (LLM)

    Worker->>Workspace: create(repo_url)
    Workspace->>Workspace: Shallow clone git repo
    Workspace->>Sandbox: Boot isolated container (read-only, no-network, 512MB RAM)
    Worker->>Indexer: Upfront index_directory(workspace_path)
    Worker->>Agent: execute_workspace_task(task, tools=[hybrid_search, read/write_file, run_shell...])
    
    loop Agent Reasoning Steps
        Agent->>Indexer: hybrid_search(query="divide") / get_symbol_definition()
        Indexer-->>Agent: Exact match & vector code chunks
        Agent->>Sandbox: write_file("calculator.py", fixed_code)
        Agent->>Sandbox: run_shell("python3 test_calculator.py")
        Sandbox-->>Agent: Pytest output & exit code 0
    end

    Agent-->>Worker: Final answer with complete trajectory
    Worker->>Workspace: destroy(workspace_id)
    Workspace->>Sandbox: Stop and remove container
    Workspace->>Workspace: Clean host workspace directory
```

---

## Semantic & Hybrid Code Search

Talos pairs native Python AST syntax trees with dense vector embeddings stored in PostgreSQL via `pgvector`:

```python
from pathlib import Path
from src.indexer import (
    CodeIndexer,
    ASTChunker,
    GeminiEmbeddingClient,
    PGVectorStore,
    HybridSearchEngine,
)

# 1. Initialize Chunker and Embedding Client
chunker = ASTChunker()
embedding_client = GeminiEmbeddingClient(api_key="your-gemini-key")
vector_store = PGVectorStore(database_session_factory=session_factory)
indexer = CodeIndexer()

# 2. Build Hybrid Search Engine
search_engine = HybridSearchEngine(
    indexer=indexer,
    embedding_client=embedding_client,
    vector_store=vector_store,
    chunker=chunker,
)

# 3. Index Codebase upfront
await search_engine.index_directory(Path("src"))

# 4. Perform Hybrid Search
# Matches exact symbol definitions with top priority (1.300 boost),
# and smoothly falls back to dense vector similarity for descriptive queries.
results = await search_engine.search_hybrid(
    "execute command inside docker container and stream lines", top_k=5
)
for r in results:
    print(
        f"[{r.match_type.value.upper()} | Score: {r.score:.3f}] {r.chunk.symbol_name} ({r.chunk.file_path})"
    )
    print(r.chunk.signature)
```

### Search REST Endpoints
- **Semantic Vector Search**:
  ```bash
  curl -X POST http://localhost:8000/search/semantic     -H "Content-Type: application/json"     -d '{"query": "asynchronously stream docker command output", "top_k": 3}'
  ```
- **Hybrid Search**:
  ```bash
  curl -X POST http://localhost:8000/search/hybrid     -H "Content-Type: application/json"     -d '{"query": "WorkspaceManager", "top_k": 3}'
  ```

---

## Docker Workspace Hardening & Streaming Execution

Each task runs inside an isolated Docker container with layered defense-in-depth:

```python
from src.workspace import ContainerSecurityConfig, WorkspaceManager

manager = WorkspaceManager(
    default_security_config=ContainerSecurityConfig(
        mem_limit="512m",
        pids_limit=256,
        read_only=True,
        network_mode="none",
        user="1000:1000",
    )
)

workspace = manager.create(
    repo_url="https://github.com/octocat/Hello-World.git",
    image="python:3.12-slim",
)

# Streaming execution with output capping & timeout sentinels
async for output_line in manager.execute_command(
    workspace.workspace_id,
    "python3 -c \"print('Executing inside hardened container')\"",
    timeout_s=10.0,
    max_output_bytes=1024 * 1024,
):
    print(f"[{output_line.stream}] {output_line.line}")

# Teardown container and delete host directory cleanly
manager.destroy(workspace.workspace_id)
```

### Container Hardening Controls
- **Memory Ceiling (`mem_limit="512m"`)**: Cgroups terminate runaway memory allocations with `SIGKILL` (exit code 137).
- **Process Ceiling (`pids_limit=256"`)**: Cgroups block fork bomb attacks instantly with `BlockingIOError: [Errno 11] Resource temporarily unavailable`.
- **Read-Only Root Filesystem (`read_only=True`)**: Prevents modification of system binaries or persistence injection, with writes restricted to `/workspace` and a bounded `tmpfs` at `/tmp`.
- **Network Isolation (`network_mode="none"`)**: Air-gaps the container namespace, preventing exfiltration or unauthorized outbound traffic.
- **Unprivileged Execution (`user="1000:1000"`)**: Containers execute under a non-root UID/GID.

---

## Running Tests & Quality Checks

```bash
# Run all test suites across the entire codebase
uv run python tests/run_tests.py

# Run full agent pipeline integration test
uv run pytest tests/integration/test_full_agent_pipeline.py -v

# Run adversarial security test suite (network, fs, fork, OOM, traversal)
uv run pytest tests/workspace/test_security.py -v -s

# Run semantic & hybrid search tests
uv run pytest tests/indexer/test_search.py tests/api/test_search_api.py -v

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
├── static/                       # Zero-dependency frontend (HTML5, CSS3, Vanilla JS)
│   ├── index.html                # Aesthetic single-page application dashboard
│   ├── style.css                 # Dark-mode glassmorphic design system
│   └── app.js                    # Reactive event streaming, WebSocket & history state
├── src/
│   ├── main.py                   # FastAPI app, lifespan database & pgvector initialization
│   ├── agent/
│   │   ├── dispatcher.py         # Tool registry and execution dispatcher
│   │   ├── llm_client.py         # HTTP (OpenAI-compatible) and Mock LLM clients
│   │   ├── loop.py               # ReAct reasoning loop coordinator
│   │   ├── models.py             # Trajectory, Step, TokenUsage & Message models
│   │   ├── pipeline.py           # Full Workspace task runner & tool dispatcher wiring
│   │   ├── prompts.py            # System prompts & tool documentation formatting
│   │   ├── retry.py              # Exponential backoff, jitter & Tenacity retry utilities
│   │   └── token_tracker.py      # Token counting & dollar cost tracking
│   ├── api/
│   │   ├── exception_handlers.py # Standardized JSON error response handlers
│   │   ├── routes/
│   │   │   ├── health.py         # /health (liveness) & /readiness endpoints
│   │   │   ├── search.py         # POST /search/semantic, POST /search/hybrid
│   │   │   ├── tasks.py          # POST /tasks, GET /tasks/{id}, GET /tasks, DELETE /tasks
│   │   │   └── websocket.py      # /ws subscriber streaming endpoint
│   │   └── schemas/
│   │       ├── events.py         # Streaming events & TaskDetailResponse schemas
│   │       └── search.py         # Semantic & hybrid search request/response schemas
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment validation
│   │   ├── database.py           # Async SQLAlchemy engine & session factory
│   │   ├── logging.py            # Structlog configuration with contextvars
│   │   ├── middleware.py         # Correlation ID & request duration middleware
│   │   └── worker.py             # TaskManager, TaskGroup worker pool & graceful drain
│   ├── db/
│   │   ├── models.py             # Task, CodeChunkModel (Vector column) & TaskStatus
│   │   └── repository.py         # Pure-function asynchronous data access layer
│   ├── indexer/
│   │   ├── chunker.py            # ASTChunker semantic code partitioner
│   │   ├── embeddings.py         # GeminiEmbeddingClient, OpenAIEmbeddingClient & CostTracker
│   │   ├── indexer.py            # CodeIndexer multi-file index & symbol query API
│   │   ├── models.py             # CodeChunk, Symbol, FunctionDefinition dataclasses
│   │   ├── parser.py             # PythonASTParser using native ast module with error recovery
│   │   ├── search.py             # HybridSearchEngine (exact match + vector fallback)
│   │   └── vector_store.py       # PGVectorStore (pgvector <=>) & InMemoryVectorStore
│   ├── tools/
│   │   ├── exceptions.py         # Custom ToolError exception hierarchy
│   │   ├── filesystem.py         # Sandboxed FileSystemTool & path safety
│   │   ├── shell.py              # Defensive ShellTool with timeout & process cleanup
│   │   └── system_tools.py       # Backward-compatible tool export shim
│   └── workspace/
│       ├── exceptions.py         # Typed WorkspaceError exception hierarchy
│       ├── git_utils.py          # Shallow clone & commit resolution with GitPython
│       ├── manager.py            # WorkspaceManager container lifecycle controller
│       └── models.py             # Workspace & ContainerSecurityConfig dataclasses
└── tests/
    ├── conftest.py               # Shared test fixtures & client lifecycle
    ├── run_tests.py              # Isolated test runner executing all module suites
    ├── agent/                    # Agent loop, dispatcher, LLM & retry tests
    ├── api/                      # API endpoint, search routes & WebSocket tests
    ├── core/                     # Config, logging, middleware & worker pool tests
    ├── db/                       # Repository and crash recovery test suites
    ├── fixtures/                 # Sample repos and synthetic test fixtures
    ├── indexer/                  # AST parser, chunker, embeddings & hybrid search tests
    ├── integration/              # Full agent pipeline e2e integration tests
    ├── tools/                    # File system & shell tool test suites
    └── workspace/                # Manager unit tests, live experiments & security tests
```

---

## License

MIT
