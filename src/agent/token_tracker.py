from __future__ import annotations

from typing import Any

import structlog

from src.agent.models import CostRates, TokenUsage

logger = structlog.get_logger(__name__)

MODEL_PRICING: dict[str, CostRates] = {
    "llama-3.3-70b-versatile": CostRates(
        prompt_cost_per_1m=0.59, completion_cost_per_1m=0.79
    ),
    "llama-3.1-8b-instant": CostRates(
        prompt_cost_per_1m=0.05, completion_cost_per_1m=0.08
    ),
    "openai/gpt-oss-120b": CostRates(
        prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60
    ),
    "gpt-oss-120b": CostRates(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60),
    "gpt-4o": CostRates(prompt_cost_per_1m=2.50, completion_cost_per_1m=10.00),
    "gpt-4o-mini": CostRates(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60),
    "gpt-4-turbo": CostRates(prompt_cost_per_1m=10.00, completion_cost_per_1m=30.00),
    "claude-3-5-sonnet": CostRates(
        prompt_cost_per_1m=3.00, completion_cost_per_1m=15.00
    ),
    "claude-3-5-haiku": CostRates(prompt_cost_per_1m=0.80, completion_cost_per_1m=4.00),
    "gemini-1.5-pro": CostRates(prompt_cost_per_1m=1.25, completion_cost_per_1m=5.00),
    "gemini-1.5-flash": CostRates(
        prompt_cost_per_1m=0.075, completion_cost_per_1m=0.30
    ),
    "gemini-2.0-flash": CostRates(prompt_cost_per_1m=0.10, completion_cost_per_1m=0.40),
    "deepseek-chat": CostRates(prompt_cost_per_1m=0.14, completion_cost_per_1m=0.28),
}

DEFAULT_COST_RATES = CostRates(prompt_cost_per_1m=0.15, completion_cost_per_1m=0.60)


class TokenTracker:
    """Tracks token consumption and computes estimated API costs."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        custom_rates: CostRates | None = None,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.rates = custom_rates or MODEL_PRICING.get(model_name, DEFAULT_COST_RATES)
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.cumulative_usage = TokenUsage()
        self._history: list[TokenUsage] = []

    def is_budget_exceeded(
        self, projected_tokens: int = 0
    ) -> tuple[bool, str | None, str | None]:
        """Checks if current usage exceeds configured token or cost limits.

        Returns (is_exceeded, budget_type, reason_message).
        """
        current_tokens = self.cumulative_usage.total_tokens + projected_tokens
        if self.max_tokens is not None and current_tokens >= self.max_tokens:
            reason = (
                f"Token budget exceeded: {current_tokens}/{self.max_tokens} tokens used"
            )
            return True, "tokens", reason

        current_cost = self.cumulative_usage.estimated_cost_usd
        if self.max_cost_usd is not None and current_cost >= self.max_cost_usd:
            reason = (
                f"Cost budget exceeded: ${current_cost:.4f}/"
                f"${self.max_cost_usd:.4f} spent"
            )
            return True, "cost", reason

        return False, None, None

    def budget_remaining_pct(self) -> float | None:
        """Computes percentage of budget remaining (0.0 to 100.0)."""
        pcts: list[float] = []
        if self.max_tokens is not None and self.max_tokens > 0:
            used_pct = (self.cumulative_usage.total_tokens / self.max_tokens) * 100.0
            pcts.append(max(0.0, min(100.0, 100.0 - used_pct)))
        if self.max_cost_usd is not None and self.max_cost_usd > 0.0:
            used_pct = (
                self.cumulative_usage.estimated_cost_usd / self.max_cost_usd
            ) * 100.0
            pcts.append(max(0.0, min(100.0, 100.0 - used_pct)))

        if not pcts:
            return None
        return round(min(pcts), 2)

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
            "tokens_used": self.cumulative_usage.total_tokens,
            "total_cost_usd": round(self.cumulative_usage.estimated_cost_usd, 6),
            "cost_usd": round(self.cumulative_usage.estimated_cost_usd, 6),
            "max_tokens": self.max_tokens,
            "max_cost_usd": self.max_cost_usd,
            "budget_remaining_pct": self.budget_remaining_pct(),
        }


def format_partial_result(
    task: str,
    plan: Any | None = None,
    tool_history: list[Any] | None = None,
    last_thought: str | None = None,
    budget_reason: str | None = None,
) -> str:
    """Formats human-readable partial result string upon budget exhaustion."""
    sections = ["=== PARTIAL RESULT (BUDGET EXCEEDED) ==="]
    sections.append(f"Task: {task}")
    if budget_reason:
        sections.append(f"Reason: {budget_reason}")

    if last_thought:
        sections.append(f"\nLast Reasoning Thought:\n{last_thought}")

    if plan is not None and hasattr(plan, "steps") and plan.steps:
        completed = [s for s in plan.steps if getattr(s, "status", None) == "completed"]
        pending = [s for s in plan.steps if getattr(s, "status", None) != "completed"]
        sections.append(
            f"\nPlan Progress: {len(completed)}/{len(plan.steps)} steps completed."
        )
        for s in completed:
            desc = getattr(s, "description", "")
            sid = getattr(s, "step_id", "?")
            sections.append(f"  ✓ Step {sid}: {desc}")
        for s in pending:
            desc = getattr(s, "description", "")
            sid = getattr(s, "step_id", "?")
            sections.append(f"  ⏳ Step {sid}: {desc}")

    if tool_history:
        sections.append(f"\nTool Interactions Executed ({len(tool_history)}):")
        for rec in tool_history[-5:]:
            tool_name = getattr(rec, "tool_name", "tool")
            step = getattr(rec, "step", "?")
            success = getattr(rec, "success", True)
            status_str = "SUCCESS" if success else "FAILED"
            sections.append(f"  - Step {step} [{status_str}]: {tool_name}")

    return "\n".join(sections)
