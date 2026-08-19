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
    # Without jitter
    d0 = compute_backoff_delay(0, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d0 == 1.0
    d1 = compute_backoff_delay(1, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d1 == 2.0
    d2 = compute_backoff_delay(2, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert d2 == 4.0

    # With jitter: between 0.5 * delay and delay
    dj = compute_backoff_delay(1, initial_delay=1.0, backoff_factor=2.0, jitter=True)
    assert 1.0 <= dj <= 2.0


@pytest.mark.asyncio
async def test_retry_async_succeeds_on_transient_error() -> None:
    mock_func = AsyncMock()
    # Fails twice with transient error, then succeeds on 3rd attempt
    mock_func.side_effect = [
        ConnectionError("Network glitch"),
        TimeoutError("Request timed out"),
        "success_result",
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await retry_async(
            mock_func,
            max_retries=3,
            initial_delay=0.1,
            backoff_factor=2.0,
            jitter=False,
        )

    assert result == "success_result"
    assert mock_func.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(0.1)
    mock_sleep.assert_any_call(0.2)


@pytest.mark.asyncio
async def test_retry_async_aborts_on_non_transient_error() -> None:
    mock_func = AsyncMock()
    mock_func.side_effect = ValueError("Invalid prompt")

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(ValueError, match="Invalid prompt"),
    ):
        await retry_async(mock_func, max_retries=3)

    assert mock_func.call_count == 1
    assert mock_sleep.call_count == 0


@pytest.mark.asyncio
async def test_retry_async_exceeds_max_retries() -> None:
    mock_func = AsyncMock()
    mock_func.side_effect = ConnectionError("Persistent outage")

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(ConnectionError, match="Persistent outage"),
    ):
        await retry_async(mock_func, max_retries=2, initial_delay=0.01)

    assert mock_func.call_count == 3  # 1 initial + 2 retries


def test_retry_sync_success_and_failure() -> None:
    mock_sync = MagicMock()
    mock_sync.side_effect = [TimeoutError("Timeout"), "done"]

    with patch("time.sleep") as mock_sleep:
        res = retry_sync(mock_sync, max_retries=2, initial_delay=0.1, jitter=False)

    assert res == "done"
    assert mock_sync.call_count == 2
    mock_sleep.assert_called_once_with(0.1)


def test_retry_sync_aborts_on_non_transient() -> None:
    mock_sync = MagicMock()
    mock_sync.side_effect = ValueError("Invalid")

    with pytest.raises(ValueError, match="Invalid"):
        retry_sync(mock_sync, max_retries=2)

    assert mock_sync.call_count == 1


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
