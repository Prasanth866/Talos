# Talos

Autonomous software engineering agent framework featuring a tool-use reasoning loop, bounded async worker queues with structured concurrency, graceful shutdown, defensive sandboxing, and real-time WebSocket event streaming.

---

## Key Features

- **Autonomous Reasoning Loop**: Step-by-step ReAct-style agent execution loop (`ReasoningLoop`) with dynamic tool dispatching, observation truncation, and token/cost tracking.
- **Async Worker Queue & Worker Pool**: Structured concurrency managed via `asyncio.TaskGroup` over a bounded `asyncio.Queue` with configurable worker concurrency.
- **Backpressure & Load Shedding**: Explicit HTTP `503 Service Unavailable` with `Retry-After` headers when the task queue reaches maximum capacity or during shutdown.
- **Graceful Shutdown**: Intercepts `SIGTERM` / application lifespan exit, safely stops accepting new submissions, and drains in-flight tasks before termination.
- **Correlation IDs**: Full UUID `task_id` propagation threaded through every structured log line via `structlog` contextvars and all WebSocket stream events.
- **WebSocket Event Streaming**: Real-time subscriber endpoint (`/ws?task_id=<uuid>`) streaming versioned events (`thought`, `tool_call`, `tool_output`, `task_complete`, `error`) with event replay support.
- **Defensive Tool Sandboxing**: Secure `FileSystemTool` (path traversal prevention) and `ShellTool` (command execution with timeouts, output limits, and process cleanup).

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

### Environment Configuration
Copy `.env.example` to `.env` and configure settings as needed:
```bash
cp .env.example .env
```

Key environment variables:
- `LLM_API_KEY`: API key for LLM provider (optional in development; uses Mock LLM if empty).
- `LLM_BASE_URL`: Base URL for OpenAI-compatible LLM endpoint (default: `https://api.groq.com/openai/v1`).
- `LLM_MODEL`: Model name (default: `openai/gpt-oss-120b`).
- `WORKER_CONCURRENCY`: Number of concurrent async workers in the pool (default: `4`).
- `TASK_QUEUE_MAX_SIZE`: Maximum bounded task queue size before backpressure (default: `100`).
- `SHUTDOWN_DRAIN_TIMEOUT_SECONDS`: Maximum seconds to wait for in-flight tasks during graceful shutdown (default: `30.0`).

### Running Locally
```bash
uv run fastapi dev src/main.py
```
API Documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

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
  "status": "queued",
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

---

## Running Tests & Quality Checks

```bash
# Run pytest with coverage report
uv run pytest --cov=src --cov-report=term-missing

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
├── pyproject.toml                # Project metadata & tool configurations
├── src/
│   ├── main.py                   # FastAPI app entrypoint & lifespan management
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
│   │   │   ├── tasks.py          # POST /tasks submission endpoint
│   │   │   └── websocket.py      # /ws subscriber streaming endpoint
│   │   └── schemas/
│   │       └── events.py         # Versioned streaming event schemas with task_id
│   ├── core/
│   │   ├── config.py             # Pydantic Settings & environment validation
│   │   ├── logging.py            # Structlog configuration with contextvars
│   │   ├── middleware.py         # Correlation ID & request duration middleware
│   │   └── worker.py             # TaskManager, TaskGroup worker pool & graceful drain
│   └── tools/
│       ├── exceptions.py         # Custom ToolError exception hierarchy
│       ├── filesystem.py         # Sandboxed FileSystemTool & path safety
│       ├── shell.py              # Defensive ShellTool with timeout & process cleanup
│       └── system_tools.py       # Backward-compatible re-exports
└── tests/
    ├── conftest.py               # Shared test fixtures & client lifecycle
    ├── agent/                    # Agent loop, dispatcher, LLM & retry tests
    ├── api/                      # API endpoint, WebSocket & schema tests
    ├── core/                     # Config, logging, middleware & worker pool tests
    └── tools/                    # File system & shell tool test suites
```

---

## License

MIT
