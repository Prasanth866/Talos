from src.agent.models import CostRates
from src.agent.token_tracker import TokenTracker


def test_token_tracker_accumulates_counts_correctly() -> None:
    tracker = TokenTracker(model_name="gpt-4o")

    # First call
    u1 = tracker.record_usage(prompt_tokens=100, completion_tokens=50)
    assert u1.prompt_tokens == 100
    assert u1.completion_tokens == 50
    assert u1.total_tokens == 150
    assert tracker.cumulative_usage.prompt_tokens == 100
    assert tracker.cumulative_usage.completion_tokens == 50
    assert tracker.cumulative_usage.total_tokens == 150
    assert tracker.call_count == 1

    # Second call
    u2 = tracker.record_usage(prompt_tokens=200, completion_tokens=100)
    assert u2.prompt_tokens == 200
    assert u2.completion_tokens == 100
    assert u2.total_tokens == 300
    assert tracker.cumulative_usage.prompt_tokens == 300
    assert tracker.cumulative_usage.completion_tokens == 150
    assert tracker.cumulative_usage.total_tokens == 450
    assert tracker.call_count == 2


def test_token_tracker_cost_calculation() -> None:
    # Custom rates: $1.00 per 1M prompt, $2.00 per 1M completion
    custom = CostRates(prompt_cost_per_1m=1.00, completion_cost_per_1m=2.00)
    tracker = TokenTracker(custom_rates=custom)

    usage = tracker.record_usage(prompt_tokens=500_000, completion_tokens=250_000)
    # 500k * $1/1M = 0.50, 250k * $2/1M = 0.50 -> Total = 1.00
    assert usage.estimated_cost_usd == 1.00
    assert tracker.cumulative_usage.estimated_cost_usd == 1.00


def test_token_tracker_record_from_response() -> None:
    tracker = TokenTracker(model_name="gpt-4o")

    # OpenAI format
    raw_response_usage = {
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
    }
    usage = tracker.record_from_response(raw_response_usage)
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 40
    assert usage.total_tokens == 160

    # Anthropic format
    anthropic_usage = {"input_tokens": 80, "output_tokens": 30}
    usage2 = tracker.record_from_response(anthropic_usage)
    assert usage2.prompt_tokens == 80
    assert usage2.completion_tokens == 30

    # Empty usage
    usage3 = tracker.record_from_response(None)
    assert usage3.total_tokens == 0


def test_token_tracker_reset_and_summary() -> None:
    tracker = TokenTracker(model_name="gpt-4o-mini")
    tracker.record_usage(100, 50)
    summary = tracker.get_summary()

    assert summary["model"] == "gpt-4o-mini"
    assert summary["call_count"] == 1
    assert summary["total_tokens"] == 150

    tracker.reset()
    assert tracker.call_count == 0
    assert tracker.cumulative_usage.total_tokens == 0
