from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.indexer.chunker import ASTChunker, CodeChunk
from src.indexer.embeddings import EmbeddingClient, MockEmbeddingClient
from src.indexer.indexer import CodeIndexer
from src.indexer.vector_store import InMemoryVectorStore, VectorStore


class MatchType(StrEnum):
    """Classification of hybrid search match origin."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """A ranked search result with relevance score and contextual explanation."""

    chunk: CodeChunk
    score: float
    match_type: MatchType
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Converts search result to a dictionary."""
        return {
            "symbol_name": self.chunk.symbol_name,
            "file_path": str(self.chunk.file_path),
            "kind": self.chunk.kind.value,
            "signature": self.chunk.signature,
            "docstring": self.chunk.docstring,
            "line_span": self.chunk.line_span.to_dict(),
            "score": round(self.score, 4),
            "match_type": self.match_type.value,
            "explanation": self.explanation,
        }


class HybridSearchEngine:
    """Combines exact AST symbol matching with dense vector semantic search."""

    def __init__(
        self,
        chunker: ASTChunker | None = None,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        indexer: CodeIndexer | None = None,
    ) -> None:
        self.chunker = chunker or ASTChunker()
        self.embedding_client = embedding_client or MockEmbeddingClient()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.indexer = indexer or CodeIndexer()
        self._chunks_by_id: dict[str, CodeChunk] = {}
        self._chunks_list: list[CodeChunk] = []

    async def index_directory(
        self, directory: Path | str, recursive: bool = True
    ) -> int:
        """Chunks, embeds, and indexes all Python files in a directory."""
        self.indexer.index_directory(directory, recursive=recursive)
        chunks = self.chunker.chunk_directory(directory, recursive=recursive)

        if not chunks:
            return 0

        # Generate embeddings in batch
        texts_to_embed = [c.embedding_text for c in chunks]
        embeddings = await self.embedding_client.embed_texts(texts_to_embed)

        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb
            self._chunks_by_id[chunk.chunk_id] = chunk

        self._chunks_list = list(self._chunks_by_id.values())
        await self.vector_store.add_chunks(chunks)

        if hasattr(self.embedding_client, "cost_tracker"):
            self.embedding_client.cost_tracker.log_summary("index_directory")

        return len(chunks)

    async def index_file(
        self, file_path: Path | str, source_code: str | None = None
    ) -> int:
        """Chunks, embeds, and indexes a single Python file."""
        self.indexer.index_file(file_path)
        chunks = self.chunker.chunk_file(file_path, source_code)

        if not chunks:
            return 0

        texts = [c.embedding_text for c in chunks]
        embeddings = await self.embedding_client.embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk.embedding = emb
            self._chunks_by_id[chunk.chunk_id] = chunk

        self._chunks_list = list(self._chunks_by_id.values())
        await self.vector_store.add_chunks(chunks)
        return len(chunks)

    def search_exact(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Performs exact symbol and signature matching against indexed chunks."""
        q = query.strip().lower()
        if not q or not self._chunks_list:
            return []

        exact_results: list[SearchResult] = []
        for chunk in self._chunks_list:
            sym_lower = chunk.symbol_name.lower()
            sig_lower = chunk.signature.lower()
            unqualified = sym_lower.split(".")[-1]

            if q in (sym_lower, unqualified):
                exact_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=1.0,
                        match_type=MatchType.EXACT,
                        explanation=f"Exact symbol name match '{chunk.symbol_name}'",
                    )
                )
            elif q in sig_lower:
                exact_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=0.9,
                        match_type=MatchType.EXACT,
                        explanation=f"Exact signature match in '{chunk.signature}'",
                    )
                )
            elif q in sym_lower:
                exact_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=0.8,
                        match_type=MatchType.EXACT,
                        explanation=f"Partial symbol match in '{chunk.symbol_name}'",
                    )
                )

        exact_results.sort(key=lambda x: x.score, reverse=True)
        return exact_results[:top_k]

    async def search_semantic(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Performs dense vector similarity search for natural language queries."""
        q = query.strip()
        if not q:
            return []

        query_vector = await self.embedding_client.embed_query(q)
        scored_chunks = await self.vector_store.search(query_vector, top_k=top_k)

        results: list[SearchResult] = []
        for chunk, similarity in scored_chunks:
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=similarity,
                    match_type=MatchType.SEMANTIC,
                    explanation=f"Cosine similarity score: {similarity:.4f}",
                )
            )
        return results

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        exact_boost: float = 1.3,
    ) -> list[SearchResult]:
        """Hybrid search prioritizing exact matches while falling back to semantic."""
        q = query.strip()
        if not q:
            return []

        exact_matches = self.search_exact(q, top_k=top_k)
        semantic_matches = await self.search_semantic(q, top_k=top_k * 2)

        merged_by_id: dict[str, SearchResult] = {}

        for res in exact_matches:
            boosted_score = res.score
            if res.score == 1.0:
                boosted_score = 1.0 * exact_boost
            merged_by_id[res.chunk.chunk_id] = SearchResult(
                chunk=res.chunk,
                score=boosted_score,
                match_type=MatchType.EXACT,
                explanation=res.explanation,
            )

        for res in semantic_matches:
            cid = res.chunk.chunk_id
            if cid in merged_by_id:
                existing = merged_by_id[cid]
                combined_score = max(
                    existing.score,
                    existing.score * 0.7 + res.score * 0.3,
                )
                merged_by_id[cid] = SearchResult(
                    chunk=res.chunk,
                    score=combined_score,
                    match_type=MatchType.HYBRID,
                    explanation=(
                        f"Hybrid match: {existing.explanation} + "
                        f"semantic similarity {res.score:.4f}"
                    ),
                )
            else:
                merged_by_id[cid] = res

        ranked_results = list(merged_by_id.values())
        ranked_results.sort(key=lambda x: x.score, reverse=True)
        return ranked_results[:top_k]
