from __future__ import annotations

from typing import Any

import structlog

from src.agent.models import CostRates, TokenUsage

logger = structlog.get_logger(__name__)

# Default pricing rates in USD per 1,000,000 tokens
MODEL_PRICING: dict[str, CostRates] = {
    # OpenAI
    "gpt-4o": CostRates(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00),
    "gpt-4o-mini": CostRates(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60),
    "gpt-4-turbo": CostRates(prompt_cost_per_1m=10.00, completion_cost_per_1m=30.00),
    # Anthropic
    "claude-3-5-sonnet": CostRates(
        prompt_cost_per_1m=3.00, completion_cost_per_1m=15.00
    ),
    "claude-3-5-haiku": CostRates(prompt_cost_per_1m=0.80, completion_cost_per_1m=4.00),
    # Google
    "gemini-1.5-pro": CostRates(prompt_cost_per_1m=1.25, completion_cost_per_1m=5.00),
    "gemini-1.5-flash": CostRates(
        prompt_cost_per_1m=0.075, completion_cost_per_1m=0.30
    ),
    "gemini-2.0-flash": CostRates(prompt_cost_per_1m=0.10, completion_cost_per_1m=0.40),
    # DeepSeek
    "deepseek-chat": CostRates(prompt_cost_per_1m=0.14, completion_cost_per_1m=0.28),
}

DEFAULT_COST_RATES = CostRates(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00)


class TokenTracker:
    """Tracks token consumption and computes estimated API costs."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        custom_rates: CostRates | None = None,
    ) -> None:
        self.model_name = model_name
        self.rates = custom_rates or MODEL_PRICING.get(model_name, DEFAULT_COST_RATES)
        self.cumulative_usage = TokenUsage()
        self._history: list[TokenUsage] = []

    def compute_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """Calculates token usage and estimated cost for a single LLM invocation."""
        total = prompt_tokens + completion_tokens
        cost = self.rates.calculate_cost(prompt_tokens, completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
        )

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """Records an LLM call's token usage, updates cumulative metrics,
        and returns the call usage.
        """
        call_usage = self.compute_usage(prompt_tokens, completion_tokens)
        self.cumulative_usage = self.cumulative_usage + call_usage
        self._history.append(call_usage)

        logger.debug(
            "llm_token_usage",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            call_cost_usd=call_usage.estimated_cost_usd,
            total_tokens=self.cumulative_usage.total_tokens,
            total_cost_usd=self.cumulative_usage.estimated_cost_usd,
        )
        return call_usage

    def record_from_response(self, raw_usage: dict[str, Any] | None) -> TokenUsage:
        """Parses standard usage dictionaries (e.g. from OpenAI API responses)."""
        if not raw_usage:
            return self.record_usage(0, 0)

        prompt_tokens = int(
            raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0
        )
        completion_tokens = int(
            raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0
        )
        return self.record_usage(prompt_tokens, completion_tokens)

    def reset(self) -> None:
        """Resets the accumulated token metrics."""
        self.cumulative_usage = TokenUsage()
        self._history.clear()

    @property
    def call_count(self) -> int:
        return len(self._history)

    def get_summary(self) -> dict[str, Any]:
        """Provides summary dictionary of token consumption and cost."""
        return {
            "model": self.model_name,
            "call_count": self.call_count,
            "prompt_tokens": self.cumulative_usage.prompt_tokens,
            "completion_tokens": self.cumulative_usage.completion_tokens,
            "total_tokens": self.cumulative_usage.total_tokens,
            "total_cost_usd": round(self.cumulative_usage.estimated_cost_usd, 6),
        }
