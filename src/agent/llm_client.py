from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx
import structlog

from src.agent.models import LLMResponse, Message, TokenUsage, ToolCall
from src.agent.retry import retry_async
from src.agent.token_tracker import TokenTracker

logger = structlog.get_logger(__name__)


def extract_json_payload(text: str) -> dict[str, Any]:
    """Extracts and parses JSON object from an LLM response string."""
    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try markdown code block extraction ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try finding the first '{' and last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse valid JSON from LLM response: {text[:200]}")


def parse_llm_response_content(
    content: str,
    token_usage: TokenUsage,
    latency: float = 0.0,
) -> LLMResponse:
    """Parses raw text content into a structured LLMResponse."""
    try:
        payload = extract_json_payload(content)
    except ValueError as exc:
        logger.warning(
            "llm_response_unparseable_json",
            error=str(exc),
            content_prefix=content[:100],
        )
        return LLMResponse(
            thought=f"Failed to parse structured JSON: {content[:100]}",
            final_answer=None,
            tool_call=None,
            raw_content=content,
            token_usage=token_usage,
            latency_seconds=latency,
        )

    thought = str(payload.get("thought", "")).strip()
    final_answer = payload.get("final_answer")
    if final_answer is not None:
        final_answer = str(final_answer).strip()

    tool_call_dict = payload.get("tool_call") or payload.get("action")
    tool_call: ToolCall | None = None

    if isinstance(tool_call_dict, dict):
        tool_name = tool_call_dict.get("tool_name") or tool_call_dict.get("tool") or ""
        arguments = tool_call_dict.get("arguments") or tool_call_dict.get("args") or {}
        if tool_name:
            tool_call = ToolCall(tool_name=str(tool_name), arguments=dict(arguments))

    return LLMResponse(
        thought=thought,
        tool_call=tool_call,
        final_answer=final_answer,
        raw_content=content,
        token_usage=token_usage,
        latency_seconds=latency,
    )


class BaseLLMClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        """Sends messages to the LLM and returns a structured LLMResponse."""
        pass


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for deterministic testing, simulated errors, and unit tests."""

    def __init__(
        self,
        responses: Sequence[str | dict[str, Any] | Exception] | None = None,
        model_name: str = "mock-model",
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.responses: list[str | dict[str, Any] | Exception] = list(responses or [])
        self.call_history: list[list[Message]] = []
        self.token_tracker = token_tracker or TokenTracker(model_name=model_name)

    def add_response(self, response: str | dict[str, Any] | Exception) -> None:
        self.responses.append(response)

    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        self.call_history.append(messages)
        start_time = time.perf_counter()

        if not self.responses:
            raise RuntimeError("MockLLMClient has no configured responses left.")

        item = self.responses.pop(0)

        if isinstance(item, Exception):
            raise item

        # Default mock token estimation (~ 4 chars per token)
        prompt_len = sum(len(m.content) for m in messages)
        prompt_tokens = max(1, prompt_len // 4)

        raw_text = json.dumps(item) if isinstance(item, dict) else str(item)

        completion_tokens = max(1, len(raw_text) // 4)
        call_usage = self.token_tracker.record_usage(prompt_tokens, completion_tokens)
        latency = time.perf_counter() - start_time

        return parse_llm_response_content(
            content=raw_text,
            token_usage=call_usage,
            latency=latency,
        )


class HTTPLLMClient(BaseLLMClient):
    """OpenAI-compatible HTTP client with retry backoff and token tracking."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.token_tracker = token_tracker or TokenTracker(model_name=model)

    def _format_messages_payload(self, messages: list[Message]) -> list[dict[str, Any]]:
        formatted = []
        for msg in messages:
            item: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                item["name"] = msg.name
            formatted.append(item)
        return formatted

    async def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def generate_response(self, messages: list[Message]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._format_messages_payload(messages),
            "temperature": 0.1,
        }

        start_time = time.perf_counter()

        # Wrap API call in exponential backoff retry
        data = await retry_async(
            self._send_request,
            payload,
            max_retries=self.max_retries,
            initial_delay=self.initial_delay,
            backoff_factor=self.backoff_factor,
        )

        latency = time.perf_counter() - start_time
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from LLM API response.")

        content = choices[0].get("message", {}).get("content", "")
        raw_usage = data.get("usage")
        call_usage = self.token_tracker.record_from_response(raw_usage)

        return parse_llm_response_content(
            content=content,
            token_usage=call_usage,
            latency=latency,
        )
