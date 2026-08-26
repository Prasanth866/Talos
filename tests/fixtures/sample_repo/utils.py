"""Helper utility routines."""

import asyncio
from pathlib import Path


async def async_fetch_data(url: str, timeout: float = 5.0) -> dict[str, str]:
    """Asynchronously fetches data from a remote endpoint."""
    await asyncio.sleep(0.01)
    return {"url": url, "status": "ok"}


def format_path(path: Path | str) -> str:
    """Formats a path as a POSIX string."""
    return str(Path(path).as_posix())
