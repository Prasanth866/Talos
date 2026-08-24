from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.retry import (
    NonRetryableError,
    compute_backoff_delay,
    is_transient_error,
    retry_async,
    retry_sync,
)


def test_is_transient_error() -> None:
    assert is_transient_error(TimeoutError("Operation timed out")) is True
    assert is_transient_error(ConnectionError("Connection refused")) is True
    assert is_transient_error(RuntimeError("Rate limit exceeded")) is True
    assert is_transient_error(RuntimeError("503 Service Unavailable")) is True

    # Non-retryable
    assert is_transient_error(NonRetryableError("Fatal")) is False
    assert is_transient_error(ValueError("Invalid argument")) is False


def test_compute_backoff_delay() -> None:
    # Without jitter: deterministic exponential growth
    d0 = compute_backoff_delay(0, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d0 == 1.0
    d1 = compute_backoff_delay(1, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d1 == 2.0
    d2 = compute_backoff_delay(2, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d2 == 4.0

    # With full-jitter: uniformly random in [0, delay]
    dj = compute_backoff_delay(1, initial_delay=1.0, backoff_factor=2.0, jitter=True)
    assert 0.0 <= dj <= 2.0

    # max_delay cap is respected
    d_capped = compute_backoff_delay(
        10, initial_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=False
    )
    assert d_capped == 5.0


@pytest.mark.asyncio
async def test_retry_async_succeeds_on_transient_error() -> None:
    """Retries on transient errors and returns the result of the first success."""
    mock_func = AsyncMock()
    # Fails twice with transient errors, succeeds on 3rd attempt
    mock_func.side_effect = [
        ConnectionError("Network glitch"),
        TimeoutError("Request timed out"),
        "success_result",
    ]

    # Patch tenacity's internal async sleep to avoid real delays in tests
    with patch("tenacity.nap.sleep", new_callable=AsyncMock):
        result = await retry_async(
            mock_func,
            max_retries=3,
            initial_delay=0.01,
            backoff_factor=2.0,
            jitter=False,
        )

    assert result == "success_result"
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_retry_async_aborts_on_non_transient_error() -> None:
    """Non-transient errors are re-raised immediately without any retry."""
    mock_func = AsyncMock()
    mock_func.side_effect = ValueError("Invalid prompt")

    with pytest.raises(ValueError, match="Invalid prompt"):
        await retry_async(mock_func, max_retries=3)

    # No retries — aborted after the first attempt
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_retry_async_exceeds_max_retries() -> None:
    """Exhausting all retries re-raises the last transient exception."""
    mock_func = AsyncMock()
    mock_func.side_effect = ConnectionError("Persistent outage")

    with (
        patch("tenacity.nap.sleep", new_callable=AsyncMock),
        pytest.raises(ConnectionError, match="Persistent outage"),
    ):
        await retry_async(mock_func, max_retries=2, initial_delay=0.01)

    assert mock_func.call_count == 3  # 1 initial + 2 retries


def test_retry_sync_success_and_failure() -> None:
    """Sync variant retries on transient error and returns success result."""
    mock_sync = MagicMock()
    mock_sync.side_effect = [TimeoutError("Timeout"), "done"]

    with patch("tenacity.nap.sleep"):
        res = retry_sync(mock_sync, max_retries=2, initial_delay=0.01, jitter=False)

    assert res == "done"
    assert mock_sync.call_count == 2


def test_retry_sync_aborts_on_non_transient() -> None:
    """Non-transient errors abort immediately without retry."""
    mock_sync = MagicMock()
    mock_sync.side_effect = ValueError("Invalid")

    with pytest.raises(ValueError, match="Invalid"):
        retry_sync(mock_sync, max_retries=2)

    assert mock_sync.call_count == 1


def test_retry_async_custom_retry_condition() -> None:
    """A custom retry_condition predicate is respected."""

    def only_value_errors(exc: BaseException) -> bool:
        return isinstance(exc, ValueError)

    mock_func = MagicMock(side_effect=RuntimeError("Not retried"))

    with pytest.raises(RuntimeError, match="Not retried"):
        retry_sync(mock_func, max_retries=3, retry_condition=only_value_errors)

    # RuntimeError does not satisfy the predicate — only one call
    assert mock_func.call_count == 1


def test_is_transient_error_http_status_codes() -> None:
    class DummyHTTPError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    assert is_transient_error(DummyHTTPError(429)) is True
    assert is_transient_error(DummyHTTPError(500)) is True
    assert is_transient_error(DummyHTTPError(503)) is True
    assert is_transient_error(DummyHTTPError(400)) is False
    assert is_transient_error(DummyHTTPError(401)) is False
    assert is_transient_error(DummyHTTPError(404)) is False


def test_retry_parameter_validation() -> None:
    """Verifies that invalid configuration parameters raise ValueError."""
    dummy = MagicMock()

    # retry_sync / retry_async validation
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        retry_sync(dummy, max_retries=-1)

    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        retry_sync(dummy, initial_delay=-0.5)

    with pytest.raises(ValueError, match="backoff_factor must be >= 1"):
        retry_sync(dummy, backoff_factor=0.5)

    with pytest.raises(ValueError, match="max_delay must be >= 0"):
        retry_sync(dummy, max_delay=-1.0)

    # compute_backoff_delay validation
    with pytest.raises(ValueError, match="attempt must be >= 0"):
        compute_backoff_delay(attempt=-1)

    with pytest.raises(ValueError, match="initial_delay must be >= 0"):
        compute_backoff_delay(attempt=0, initial_delay=-1.0)

    with pytest.raises(ValueError, match="backoff_factor must be >= 1"):
        compute_backoff_delay(attempt=0, backoff_factor=0.5)

    with pytest.raises(ValueError, match="max_delay must be >= 0"):
        compute_backoff_delay(attempt=0, max_delay=-5.0)
