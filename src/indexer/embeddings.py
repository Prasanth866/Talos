from __future__ import annotations

import hashlib
import math
import os
import re
import time
from typing import Any, Protocol, runtime_checkable

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = structlog.get_logger(__name__)

DEFAULT_EMBEDDING_COST_PER_MILLION_TOKENS = 0.02
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_GEMINI_MODEL = "text-embedding-004"


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding models generating dense vector representations."""

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generates embedding vectors for a batch of text chunks."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generates an embedding vector for a search query."""
        ...


class EmbeddingCostTracker:
    """Tracks token volume, latency, and estimated cost for embedding generation."""

    def __init__(
        self,
        cost_per_million_tokens: float = DEFAULT_EMBEDDING_COST_PER_MILLION_TOKENS,
    ) -> None:
        self.cost_per_million_tokens = cost_per_million_tokens
        self.total_tokens: int = 0
        self.total_chunks: int = 0
        self.total_cost_usd: float = 0.0
        self.start_time: float = time.perf_counter()

    def record(self, token_count: int, chunk_count: int = 1) -> None:
        """Records token consumption for chunks."""
        self.total_tokens += token_count
        self.total_chunks += chunk_count
        self.total_cost_usd = (
            self.total_tokens / 1_000_000.0
        ) * self.cost_per_million_tokens

    def log_summary(self, context: str = "indexing_run") -> dict[str, Any]:
        """Logs structured metrics and returns summary dictionary."""
        elapsed = time.perf_counter() - self.start_time
        summary = {
            "context": context,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "duration_seconds": round(elapsed, 4),
        }
        logger.info("embedding_cost_summary", **summary)
        return summary


class GeminiEmbeddingClient:
    """Google Gemini embedding client using text-embedding-004 via REST API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        batch_size: int = 50,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model.removeprefix("models/")
        self._dimension = dimension
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.cost_tracker = cost_tracker or EmbeddingCostTracker()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _is_transient_error(self, exc: BaseException) -> bool:
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
        return False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception(self._is_transient_error),
            wait=wait_random_exponential(min=1.0, max=10.0),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        async def _call() -> list[list[float]]:
            url = f"{self.base_url}/models/{self.model}:batchEmbedContents"
            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": text}]},
                    }
                    for text in batch
                ]
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

                raw_embeddings = data.get("embeddings", [])
                tokens_used = sum(max(1, len(t) // 4) for t in batch)
                self.cost_tracker.record(tokens_used, chunk_count=len(batch))

                return [item["values"] for item in raw_embeddings]

        return await _call()


class OpenAIEmbeddingClient:
    """Async OpenAI-compatible embedding client with batching and backoff."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 64,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.cost_tracker = cost_tracker or EmbeddingCostTracker()

    @property
    def dimension(self) -> int:
        return self._dimension

    def _is_transient_error(self, exc: BaseException) -> bool:
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 500, 502, 503, 504)
        return False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception(self._is_transient_error),
            wait=wait_random_exponential(min=1.0, max=10.0),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        async def _call() -> list[list[float]]:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": batch,
                "model": self.model,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", len(" ".join(batch)) // 4)
                self.cost_tracker.record(tokens_used, chunk_count=len(batch))

                embeddings_data = sorted(
                    data.get("data", []), key=lambda x: x.get("index", 0)
                )
                return [item["embedding"] for item in embeddings_data]

        return await _call()


class MockEmbeddingClient:
    """Deterministic, normalized embedding generator for offline testing."""

    def __init__(
        self,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> None:
        self._dimension = dimension
        self.cost_tracker = cost_tracker or EmbeddingCostTracker()

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            vector = self._generate_deterministic_vector(text)
            results.append(vector)
            tokens = max(1, len(text) // 4)
            self.cost_tracker.record(tokens, chunk_count=1)
        return results

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Produces a normalized vector reflecting token semantics."""
        vector = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower())

        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(8):
                dim = ((h[idx * 2] << 8) | h[idx * 2 + 1]) % self._dimension
                vector[dim] += 1.0

        squared_sum = sum(v * v for v in vector)
        norm = math.sqrt(squared_sum)
        if norm > 0:
            return [v / norm for v in vector]
        vector[0] = 1.0
        return vector


def create_default_embedding_client(
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    cost_tracker: EmbeddingCostTracker | None = None,
) -> EmbeddingClient:
    """Factory creating GeminiEmbeddingClient or MockEmbeddingClient fallback."""
    key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if key and key.strip():
        return GeminiEmbeddingClient(
            api_key=key.strip(),
            model=model,
            dimension=dimension,
            cost_tracker=cost_tracker,
        )
    return MockEmbeddingClient(dimension=dimension, cost_tracker=cost_tracker)
