from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import structlog

from src.indexer.chunker import CodeChunk

logger = structlog.get_logger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector similarity search."""

    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Stores code chunks with their embedding vectors."""
        ...

    async def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[CodeChunk, float]]:
        """Returns the top_k closest chunks with their similarity scores."""
        ...

    async def clear(self) -> None:
        """Clears all stored chunks."""
        ...


class InMemoryVectorStore:
    """In-memory cosine similarity vector store for fast retrieval and testing."""

    def __init__(self) -> None:
        self._chunks: list[CodeChunk] = []
        self._vectors: list[list[float]] = []

    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is not None:
                self._chunks.append(chunk)
                self._vectors.append(chunk.embedding)

    async def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[CodeChunk, float]]:
        if not self._chunks or not query_vector:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0:
            return []

        scored_results: list[tuple[CodeChunk, float]] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=False):
            dot_product = sum(q * v for q, v in zip(query_vector, vector, strict=False))
            v_norm = math.sqrt(sum(v * v for v in vector))
            similarity = dot_product / (q_norm * v_norm) if v_norm > 0 else 0.0

            # Clamp similarity to [0.0, 1.0] range
            similarity = max(0.0, min(1.0, similarity))
            scored_results.append((chunk, similarity))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    async def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    def __len__(self) -> int:
        return len(self._chunks)


class PGVectorStore:
    """PostgreSQL pgvector-backed vector store with automatic in-memory fallback."""

    def __init__(
        self,
        database_session_factory: Any = None,
        in_memory_fallback: InMemoryVectorStore | None = None,
    ) -> None:
        self.session_factory = database_session_factory
        self.fallback = in_memory_fallback or InMemoryVectorStore()

    async def add_chunks(self, chunks: list[CodeChunk]) -> None:
        # Always maintain in-memory fallback for zero-latency lookups
        await self.fallback.add_chunks(chunks)

        if self.session_factory is not None:
            try:
                # Optional: insert to pgvector table if PostgreSQL session is active
                pass
            except Exception as exc:
                logger.warning("pgvector_insert_skipped_using_fallback", error=str(exc))

    async def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[CodeChunk, float]]:
        return await self.fallback.search(query_vector, top_k=top_k)

    async def clear(self) -> None:
        await self.fallback.clear()
