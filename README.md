# Talos

Defensive Execution Framework & API with structured logging, sandbox safety, and async tooling.

## Quick Start

### Prerequisites
- Python >= 3.14
- [`uv`](https://github.com/astral-sh/uv)

### Installation
```bash
uv sync
```

### Running Locally
```bash
uv run fastapi dev src/main.py
```

### Running Tests & Quality Checks
```bash
# Run pytest with coverage report
uv run pytest --cov=src

# Run type checker (strict mode)
uv run mypy .

# Run linter & formatter
uv run ruff check .
uv run ruff format --check .
```

## Project Structure

```
Talos/
├── .env.example              # Environment variable template
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff, mypy, pytest)
├── pyproject.toml            # Project dependencies and tool configurations
├── src/
│   ├── main.py               # FastAPI entrypoint and lifespan management
│   ├── api/
│   │   ├── exception_handlers.py # Structured JSON exception handlers
│   │   └── routes/
│   │       └── health.py     # /health endpoint
│   ├── core/
│   │   ├── config.py         # Pydantic Settings & environment config
│   │   ├── logging.py        # Structlog configuration & processors
│   │   └── middleware.py     # Correlation ID & request logging middleware
│   └── tools/
│       ├── exceptions.py     # Custom ToolError exception hierarchy
│       ├── filesystem.py     # Sandboxed FileSystemTool & file helpers
│       ├── shell.py          # Defensive ShellTool with timeout & process cleanup
│       └── system_tools.py   # Backward-compatible re-export shim
└── tests/
    ├── conftest.py           # Shared test fixtures & helpers
    ├── core/
    │   └── test_core.py      # Core config, logging, middleware & exception tests
    └── tools/
        └── test_system_tools.py # Filesystem and Shell tool test suite
```
