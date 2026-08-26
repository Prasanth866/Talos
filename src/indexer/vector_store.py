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
        await self.fallback.add_chunks(chunks)

        if self.session_factory is not None:
            try:
                from src.db.models import CodeChunkModel

                async with (
                    self.session_factory() as session,
                    session.begin(),
                ):
                    for chunk in chunks:
                        model = CodeChunkModel(
                            id=chunk.chunk_id,
                            file_path=str(chunk.file_path),
                            symbol_name=chunk.symbol_name,
                            kind=chunk.kind.value,
                            signature=chunk.signature,
                            docstring=chunk.docstring,
                            code_content=chunk.code_content,
                            start_line=chunk.line_span.start_line,
                            end_line=chunk.line_span.end_line,
                            embedding=chunk.embedding,
                            metadata_json=chunk.metadata,
                        )
                        await session.merge(model)
            except Exception as exc:
                logger.debug("pgvector_db_insert_skipped", error=str(exc))

    async def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[CodeChunk, float]]:
        if self.session_factory is not None:
            try:
                from pathlib import Path

                from sqlalchemy import select

                from src.db.models import CodeChunkModel
                from src.indexer.models import LineSpan, SymbolKind

                async with self.session_factory() as session:
                    stmt = (
                        select(
                            CodeChunkModel,
                            CodeChunkModel.embedding.cosine_distance(
                                query_vector
                            ).label("distance"),
                        )
                        .where(CodeChunkModel.embedding.is_not(None))
                        .order_by("distance")
                        .limit(top_k)
                    )
                    result = await session.execute(stmt)
                    rows = result.all()
                    if rows:
                        results: list[tuple[CodeChunk, float]] = []
                        for model, distance in rows:
                            score = max(0.0, min(1.0, 1.0 - float(distance)))
                            chunk = CodeChunk(
                                chunk_id=model.id,
                                file_path=Path(model.file_path),
                                symbol_name=model.symbol_name,
                                kind=SymbolKind(model.kind),
                                signature=model.signature,
                                docstring=model.docstring,
                                code_content=model.code_content,
                                line_span=LineSpan(
                                    start_line=model.start_line,
                                    end_line=model.end_line,
                                    start_col=0,
                                    end_col=0,
                                ),
                                embedding_text=model.code_content,
                                embedding=model.embedding,
                                metadata=model.metadata_json or {},
                            )
                            results.append((chunk, score))
                        return results
            except Exception as exc:
                logger.debug("pgvector_db_search_skipped", error=str(exc))

        return await self.fallback.search(query_vector, top_k=top_k)

    async def clear(self) -> None:
        await self.fallback.clear()
