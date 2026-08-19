from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = structlog.get_logger(__name__)

# tenacity uses the standard logging module for its before_sleep hook.
_std_logger = logging.getLogger(__name__)


class NonRetryableError(Exception):
    """Explicitly marks an error as unrecoverable to stop retries."""


def is_transient_error(exc: BaseException) -> bool:
    """Heuristic to determine if an error or HTTP status code is transient.

    Returns True for errors that are safe to retry (rate limits, server
    overloads, network glitches). Returns False for client-side errors
    that will not resolve by retrying (bad requests, auth failures, etc.).
    """
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
    """Calculates exponential backoff delay with optional randomized jitter.

    Kept for compatibility and direct use where callers need to compute a
    delay value without executing a retry loop.
    """
    calculated = initial_delay * (backoff_factor**attempt)
    delay = min(calculated, max_delay)
    if jitter and delay > 0:
        delay = random.uniform(0, delay)  # noqa: S311
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
    """Executes an async callable with exponential backoff on transient errors.

    Uses tenacity's AsyncRetrying internally for battle-tested retry semantics,
    composable stop/wait strategies, and structured before-sleep logging.

    Args:
        func: Async callable to execute.
        *args: Positional arguments forwarded to ``func``.
        max_retries: Maximum number of retry attempts (not counting the initial
            attempt). Total calls = max_retries + 1.
        initial_delay: Starting delay in seconds before the first retry.
        backoff_factor: Multiplier applied to the delay on each attempt.
        max_delay: Upper bound on the computed backoff delay (seconds).
        jitter: If True, applies full-jitter [0, delay] to desynchronise
            concurrent retrying callers.
        retry_condition: Optional predicate ``(exc) -> bool``. Defaults to
            :func:`is_transient_error`.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        The return value of ``func`` on a successful attempt.

    Raises:
        The last exception raised by ``func`` if all attempts are exhausted
        or the error is non-retryable.
    """
    should_retry = retry_condition or is_transient_error

    jitter_val = initial_delay if jitter else 0
    jitter_wait = wait_random(0, jitter_val) if jitter else wait_random(0, 0)
    wait_strategy = (
        wait_exponential(
            multiplier=initial_delay, exp_base=backoff_factor, max=max_delay
        )
        + jitter_wait
    )

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_strategy,
            retry=retry_if_exception(should_retry),
            before_sleep=before_sleep_log(_std_logger, logging.INFO),
            reraise=True,
        ):
            with attempt:
                return await func(*args, **kwargs)
    except RetryError as exc:
        raise exc.last_attempt.exception() from exc  # type: ignore[misc]

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
    """Executes a synchronous callable with exponential backoff on transient errors.

    Synchronous counterpart of :func:`retry_async`. Uses tenacity's
    ``Retrying`` context iterator internally.

    Args:
        func: Synchronous callable to execute.
        *args: Positional arguments forwarded to ``func``.
        max_retries: Maximum number of retry attempts after the initial call.
        initial_delay: Starting delay in seconds before the first retry.
        backoff_factor: Multiplier applied to the delay on each attempt.
        max_delay: Upper bound on the computed backoff delay (seconds).
        jitter: If True, applies full-jitter to spread retrying callers.
        retry_condition: Optional predicate ``(exc) -> bool``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        The return value of ``func`` on a successful attempt.

    Raises:
        The last exception raised by ``func`` if all attempts are exhausted
        or the error is non-retryable.
    """
    should_retry = retry_condition or is_transient_error
    jitter_val = initial_delay if jitter else 0
    jitter_wait = wait_random(0, jitter_val) if jitter else wait_random(0, 0)
    wait_strategy = (
        wait_exponential(
            multiplier=initial_delay, exp_base=backoff_factor, max=max_delay
        )
        + jitter_wait
    )

    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_strategy,
            retry=retry_if_exception(should_retry),
            before_sleep=before_sleep_log(_std_logger, logging.INFO),
            reraise=True,
        ):
            with attempt:
                return func(*args, **kwargs)
    except RetryError as exc:
        raise exc.last_attempt.exception() from exc  # type: ignore[misc]

    raise RuntimeError("Unexpected end of retry loop without return or exception.")
