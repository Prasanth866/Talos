# Talos

Autonomous software engineering agent framework featuring a tool-use reasoning loop, bounded async worker queues with structured concurrency, graceful shutdown, persistent task state with crash recovery, defensive sandboxing, real-time WebSocket event streaming, and a zero-dependency Lovable / Replit AI aesthetic web UI.

---

## Key Features

- **Autonomous Reasoning Loop**: Step-by-step ReAct-style agent execution loop (`ReasoningLoop`) with dynamic tool dispatching, compact observation limits, adaptive rate-limit (429) backoff, and token/cost tracking.
- **Async Worker Queue & Pool**: Structured concurrency managed via `asyncio.TaskGroup` over a bounded `asyncio.Queue` with configurable worker concurrency.
- **Persistent Task Store**: Async database persistence using SQLAlchemy and Alembic, tracking complete task lifecycles (`PENDING -> RUNNING -> COMPLETED / FAILED`), token counts, USD cost, and final answers.
- **Crash Recovery**: Automatic startup recovery identifying any interrupted `RUNNING` tasks from crashes or abrupt server kills and marking them as `FAILED`.
- **Task Query & Filtering**: REST endpoints (`POST /tasks`, `GET /tasks/{id}`, `GET /tasks?status=...`, `DELETE /tasks/{id}`, `DELETE /tasks`) with pagination and status filtering.
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

### Running Locally

```bash
uv run fastapi dev src/main.py
```
- **Web UI & Live Agent Workspace**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

*(Note: The static frontend can also be viewed via VS Code Live Server at `http://127.0.0.1:5500/` or opened directly as a file; API and WebSocket requests automatically route to the backend on port 8000).*

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
# Run pytest with full coverage report
uv run pytest --cov=src --cov-report=term-missing

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
│   │   ├── dispatcher.py         # Tool registry and execution dispatcher
│   │   ├── llm_client.py         # HTTP (OpenAI-compatible) and Mock LLM clients
│   │   ├── loop.py               # ReAct reasoning loop coordinator
│   │   ├── models.py             # Trajectory, Step, TokenUsage & Message models
│   │   ├── prompts.py            # System prompts & tool documentation formatting
│   │   ├── retry.py              # Exponential backoff & retry mechanisms
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
│   └── tools/
│       ├── exceptions.py         # Custom ToolError exception hierarchy
│       ├── filesystem.py         # Sandboxed FileSystemTool & path safety
│       ├── shell.py              # Defensive ShellTool with timeout & process cleanup
│       └── system_tools.py       # Backward-compatible re-exports
└── tests/
    ├── conftest.py               # Shared test fixtures & client lifecycle
    ├── agent/                    # Agent loop, dispatcher, LLM & retry tests
    ├── api/                      # API endpoint, WebSocket, task queries & schema tests
    ├── core/                     # Config, logging, middleware & worker pool tests
    ├── db/                       # Repository and crash recovery test suites
    └── tools/                    # File system & shell tool test suites
```

---

## License

MIT
