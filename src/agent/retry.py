from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class NonRetryableError(Exception):
    """Explicitly marks an error as unrecoverable to stop retries."""


def is_transient_error(exc: BaseException) -> bool:
    """Heuristic to determine if an error or HTTP status code is transient."""
    if isinstance(exc, NonRetryableError):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if status_code in (429, 500, 502, 503, 504, 520, 524):
            return True
        if 400 <= status_code < 500:
            return False

    transient_types = (
        TimeoutError,
        ConnectionError,
        asyncio.TimeoutError,
        OSError,
    )
    if isinstance(exc, transient_types):
        return True

    exc_repr = f"{type(exc).__name__}: {exc}".lower()
    transient_indicators = [
        "rate limit",
        "ratelimit",
        "429",
        "quota exceeded",
        "too many requests",
        "service unavailable",
        "503",
        "bad gateway",
        "502",
        "gateway timeout",
        "504",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "timed out",
        "timeout",
    ]
    return any(indicator in exc_repr for indicator in transient_indicators)


def compute_backoff_delay(
    attempt: int,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculates exponential backoff delay with randomized jitter."""
    calculated = initial_delay * (backoff_factor**attempt)
    delay = min(calculated, max_delay)
    if jitter and delay > 0:
        delay = random.uniform(0.5 * delay, delay)  # noqa: S311
    return delay


async def retry_async[R](
    func: Callable[..., Awaitable[R]],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_condition: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> R:
    """Executes an async callable with exponential backoff on transient errors."""
    should_retry = retry_condition or is_transient_error
    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except BaseException as exc:
            last_exception = exc

            if attempt >= max_retries or not should_retry(exc):
                logger.warning(
                    "retry_aborted",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

            delay = compute_backoff_delay(
                attempt=attempt,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                jitter=jitter,
            )

            logger.info(
                "retrying_after_transient_error",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=round(delay, 3),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await asyncio.sleep(delay)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected end of retry loop without return or exception.")


def retry_sync[R](
    func: Callable[..., R],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_condition: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> R:
    """Executes a synchronous callable with exponential backoff on transient errors."""
    should_retry = retry_condition or is_transient_error
    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except BaseException as exc:
            last_exception = exc

            if attempt >= max_retries or not should_retry(exc):
                logger.warning(
                    "retry_aborted_sync",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

            delay = compute_backoff_delay(
                attempt=attempt,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                jitter=jitter,
            )

            logger.info(
                "retrying_after_transient_error_sync",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay_seconds=round(delay, 3),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            time.sleep(delay)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected end of retry loop without return or exception.")
