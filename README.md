# Talos

Autonomous software engineering agent framework featuring a LangGraph state machine with SQLite checkpoint persistence, context window management, bounded retry loops with exponential backoff and circuit breakers, unified diff patch application with auto incremental re-indexing, token & cost budget hard stops, multi-layered security guardrails (path traversal, dangerous command detection, secrets scanning & redaction, prompt injection defense), Docker workspace sandboxing, native AST code indexing, pgvector semantic & hybrid code search, real-time WebSocket event streaming, and a zero-dependency aesthetic web UI.

---

## Key Features

### 1. LangGraph State Machine & Bounded Failure Modes
- **Typed Agent State**: Structured `AgentState` tracking `task_id`, `plan`, `current_step_index`, `tool_history`, `reflection_history`, `retry_count`, `circuit_failures`, `max_tokens`, `max_cost_usd`, `partial_result`, and execution `status`.
- **SQLite Checkpoint Persistence**: Resilient state persistence using `SqliteSaver` allowing seamless task inspection, state restoration, and crash resumption across execution steps.
- **Context Window Management**: Sliding window compaction (`ContextManager`) that compacts older tool history while preserving the initial user problem statement and recent detailed tool observations, preventing context window exhaustion on long-running tasks.
- **Planning Node with Structured Output**: Pydantic-validated JSON execution plan formulation with schema validation and retry-on-malformed-output.
- **Execution Node & Schema Validation**: Structured `ToolCall` dispatching with strict argument schema validation before execution; errors are captured as typed `ToolError` objects and never silently swallowed.
- **Reflection Node & Circuit Breaker**: Structured pytest diagnostic parser (`parse_pytest_output`) with ANSI escape code stripping, bounded retries with exponential backoff ($2^{\text{retry}}$ seconds), and a 3-consecutive API error circuit breaker (`CircuitOpenError`) to prevent cascading loops.

### 2. Code Editing, Patching & Auto Re-Indexing
- **Unified Diff Patch Tool (`apply_patch`)**: High-performance multi-hunk patch engine (`PatchTool`) with in-memory dry-run validation, atomic disk application, and mixed CRLF/LF line ending normalization.
- **Incremental Auto Re-Indexing**: Modifying files via `apply_patch` automatically triggers incremental AST re-parsing (`CodeIndexer`) and dense vector embedding updates (`HybridSearchEngine`) without full codebase rebuilds.
- **Actionable `PatchError`**: Rejection errors provide detailed context (mismatches, missing lines, deletion conflicts) enabling the agent to self-correct diffs.

### 3. Token & Cost Budget Enforcement
- **Configurable Task Budgets**: Enforces per-task limits on maximum tokens (`max_tokens`) and maximum cost in USD (`max_cost_usd`).
- **Pre-Call Hard Stops**: Verifies remaining budget before every LLM invocation; immediately halts execution and emits `BudgetExceededEvent` when exceeded.
- **Partial Result Preservation**: Compiles completed plan steps, thoughts, and tool actions into a human-readable partial report saved directly to the database upon budget exhaustion.
- **Task Status API Metrics**: REST API endpoints expose `tokens_used`, `cost_usd`, `max_tokens`, `max_cost_usd`, and `budget_remaining_pct`.

### 4. Multi-Layered Security Guardrails with Proof
- **Path Traversal Guard**: Strictly enforces that all file operations (`read_file`, `write_file`, `list_dir`, `apply_patch`) stay within the workspace sandbox root, blocking both relative (`../../etc/passwd`) and absolute (`/workspace/../etc/passwd`, `/etc/shadow`) traversal attempts with `PathTraversalError`.
- **Dangerous Command Guard**: Proactively scans shell commands upfront against blacklisted executables (`sudo`, `su`, `chmod`, `chown`, `dd`, `mkfs.*`, `fdisk.*`) and destructive patterns (`rm -rf /`, `curl ... | sh`, `wget ... | bash`, `mkfifo`, `eval`), raising `DangerousCommandError` without spawning subprocesses.
- **Secrets Scanner & Redactor**: Intercepts file writes and command outputs, scanning for AWS Access Keys (`AKIA...`), Provider API Keys (`sk-proj-...`, `sk-ant-...`, `ghp-...`), and PEM Private Keys (`-----BEGIN RSA PRIVATE KEY-----`), redacting them with `[REDACTED_SECRET: <TYPE>]` and emitting `SecurityEvent`.
- **Prompt Injection Delimiter Defense**: Encloses all external tool outputs inside `<untrusted_observation source="{tool}" step="{step}">` tags and equips system prompts with strict instruction isolation rules to neutralize prompt injection payloads.

### 5. Workspace Isolation, Code Indexing & Search
- **Docker Workspace Sandboxing**: Task-isolated container environments (`WorkspaceManager`) with shallow Git cloning (`--depth 1`), memory limits (`512MB`), CPU quotas, process limits (`pids_limit=256`), read-only root filesystems, air-gapped network isolation (`network_mode="none"`), and non-root execution (`user="1000:1000"`).
- **Native AST Code Indexing**: Incremental, fault-tolerant Python AST indexer extracting function/class signatures, docstrings, typed arguments, inheritance hierarchies, and imports with syntax-error recovery.
- **AST Semantic Chunking**: Slices code along syntax boundaries (complete functions, classes, methods) rather than arbitrary token cuts.
- **PostgreSQL `pgvector` & In-Memory Vector Store**: Dense vector similarity search with PostgreSQL `pgvector` (`<=>` cosine distance) and in-memory vector store fallback.
- **Hybrid Code Search Engine**: Multi-tier search (`HybridSearchEngine`) boosting exact symbol matches (`exact_boost=1.3`) while seamlessly falling back to dense vector embeddings for semantic queries.

### 6. Streaming API & Aesthetic Web UI
- **WebSocket Event Streaming**: Real-time subscriber endpoint (`/ws?task_id=<uuid>`) streaming versioned events (`thought`, `tool_call`, `tool_output`, `budget_exceeded`, `security_alert`, `task_complete`, `error`) with event replay support.
- **Zero-Dependency Web Dashboard**: Pure HTML5, CSS3, and Vanilla JavaScript UI served directly by FastAPI. Features real-time event streaming cards, floating composer dock, terminal drawer, and history management.

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

## Agent Architecture & LangGraph State Machine

```mermaid
stateDiagram-v2
    [*] --> Planning: Submit Task / Load Checkpoint
    
    Planning --> Execution: Plan Formulated
    Planning --> Failed: Malformed Plan (Max Retries)
    
    Execution --> Reflection: Tool Call Executed
    Execution --> Execution: Next Tool Step
    Execution --> BudgetExceeded: Budget Depleted
    
    Reflection --> Execution: Tests Failing (Retry with Backoff)
    Reflection --> Completed: All Tests Pass / Goal Met
    Reflection --> Failed: Max Retries / Circuit Open
    
    BudgetExceeded --> Failed: Write Partial Result
    Completed --> [*]
    Failed --> [*]
```

---

## Running Tests & Quality Checks

```bash
# Run all 216 unit and integration test suites
uv run pytest -v

# Run full test suite with test coverage report (>= 80% required)
uv run pytest --cov=src

# Run security proof tests (path traversal, dangerous commands, secrets, injection)
uv run pytest tests/agent/test_guardrails_proof.py -v

# Run real-world GitHub bug-fix evaluation benchmarks
uv run pytest tests/integration/test_real_world_bugfixes.py -v

# Run strict type checking
uv run mypy src tests

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
│   │   ├── context.py            # ContextManager sliding window & history compaction
│   │   ├── dispatcher.py         # Tool registry and execution dispatcher
│   │   ├── graph.py              # LangGraphAgent state machine (Plan, Execute, Reflect)
│   │   ├── llm_client.py         # HTTP (OpenAI-compatible) and Mock LLM clients
│   │   ├── loop.py               # ReAct reasoning loop coordinator
│   │   ├── models.py             # Trajectory, Step, TokenUsage & Message models
│   │   ├── pipeline.py           # Full Workspace task runner & tool dispatcher wiring
│   │   ├── prompts.py            # System prompts, injection defense & tool doc formatting
│   │   ├── reflection.py         # Pytest result parser, bounded retries & circuit breaker
│   │   ├── retry.py              # Exponential backoff, jitter & Tenacity retry utilities
│   │   ├── security.py           # Security re-exports and prompt isolation helpers
│   │   ├── state.py              # Typed AgentState schema & factory
│   │   └── token_tracker.py      # Token counting, USD cost estimation & budget hard stops
│   ├── api/
│   │   ├── exception_handlers.py # Standardized JSON error response handlers
│   │   ├── routes/
│   │   │   ├── health.py         # /health (liveness) & /readiness endpoints
│   │   │   ├── search.py         # POST /search/semantic, POST /search/hybrid
│   │   │   ├── tasks.py          # Task REST API (submit, detail, list, cancel, metrics)
│   │   │   └── websocket.py      # /ws subscriber streaming endpoint with event replay
│   │   └── schemas/
│   │       ├── events.py         # Streaming events (Thought, Tool, Budget, Security)
│   │       └── search.py         # Semantic & hybrid search request/response schemas
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment validation
│   │   ├── database.py           # Async SQLAlchemy engine & session factory
│   │   ├── logging.py            # Structlog configuration with contextvars
│   │   ├── middleware.py         # Correlation ID & request duration middleware
│   │   └── worker.py             # TaskManager, TaskGroup worker pool & graceful drain
│   ├── db/
│   │   ├── models.py             # Task (with token/cost budgets), CodeChunkModel & Status
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
│   │   ├── exceptions.py         # Typed ToolError hierarchy (PathTraversal, DangerousCommand, PatchError)
│   │   ├── filesystem.py         # Sandboxed FileSystemTool & path traversal defense
│   │   ├── patch.py              # PatchTool unified diff engine with dry-run validation
│   │   ├── security.py           # SecretsScanner (AWS, OpenAI, PEM private key redaction)
│   │   ├── shell.py              # Defensive ShellTool with command denylists & timeout
│   │   └── system_tools.py       # Backward-compatible tool export shim
│   └── workspace/
│       ├── exceptions.py         # Typed WorkspaceError exception hierarchy
│       ├── git_utils.py          # Shallow clone & commit resolution with GitPython
│       ├── manager.py            # WorkspaceManager container lifecycle controller
│       └── models.py             # Workspace & ContainerSecurityConfig dataclasses
└── tests/
    ├── conftest.py               # Shared test fixtures & client lifecycle
    ├── run_tests.py              # Isolated test runner executing all module suites
    ├── agent/                    # LangGraph, budget, guardrails, reflection, dispatcher tests
    ├── api/                      # API endpoint, search routes & WebSocket tests
    ├── core/                     # Config, logging, middleware & worker pool tests
    ├── db/                       # Repository and crash recovery test suites
    ├── fixtures/                 # Sample repos and synthetic test fixtures
    ├── indexer/                  # AST parser, chunker, embeddings & hybrid search tests
    ├── integration/              # Real-world GitHub bug-fix evaluations & regression tests
    ├── tools/                    # File system, patch, security & shell tool test suites
    └── workspace/                # Manager unit tests, live experiments & security tests
```

---

## License

MIT
