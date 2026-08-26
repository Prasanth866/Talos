from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.indexer.chunker import ASTChunker
from src.indexer.embeddings import MockEmbeddingClient
from src.indexer.search import HybridSearchEngine, MatchType
from src.indexer.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_semantic_search_returns_relevant_results() -> None:
    """Unit test: semantic_search returns relevant chunks."""
    engine = HybridSearchEngine(
        chunker=ASTChunker(),
        embedding_client=MockEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    fixture_dir = Path("tests/fixtures/sample_repo")
    await engine.index_directory(fixture_dir)

    # Search with natural language query
    results = await engine.search_semantic(
        "function that asynchronously fetches data from remote endpoint",
        top_k=3,
    )
    assert len(results) > 0
    top_result = results[0]
    assert top_result.match_type == MatchType.SEMANTIC
    assert "async_fetch_data" in top_result.chunk.symbol_name
    assert top_result.score > 0.0


@pytest.mark.asyncio
async def test_hybrid_search_prefers_exact_match_over_semantic() -> None:
    """Unit test: hybrid_search prioritizes exact symbol match."""
    engine = HybridSearchEngine(
        chunker=ASTChunker(),
        embedding_client=MockEmbeddingClient(),
        vector_store=InMemoryVectorStore(),
    )
    fixture_dir = Path("tests/fixtures/sample_repo")
    await engine.index_directory(fixture_dir)

    # 1. Exact query for 'Calculator'
    exact_results = await engine.search_hybrid("Calculator", top_k=5)
    assert len(exact_results) > 0
    assert exact_results[0].chunk.symbol_name == "Calculator"
    assert exact_results[0].score >= 1.0

    # 2. Exact query for 'add'
    add_results = await engine.search_hybrid("add", top_k=5)
    assert len(add_results) > 0
    assert add_results[0].chunk.symbol_name == "add"
    assert add_results[0].score >= 1.0

    # 3. Descriptive natural language query falls back to semantic
    desc_results = await engine.search_hybrid(
        "multiplies two floating point numbers", top_k=5
    )
    assert len(desc_results) > 0
    symbols = [r.chunk.symbol_name for r in desc_results]
    assert any("multiply" in s for s in symbols)


def test_live_project_hybrid_and_semantic_experiment() -> None:
    """Experiment: Index codebase, compare hybrid vs semantic search."""

    async def _run() -> None:
        engine = HybridSearchEngine(
            chunker=ASTChunker(),
            embedding_client=MockEmbeddingClient(),
            vector_store=InMemoryVectorStore(),
        )
        src_dir = Path("src")

        t0 = time.perf_counter()
        chunk_count = await engine.index_directory(src_dir, recursive=True)
        index_duration = time.perf_counter() - t0

        print(
            f"\n[SEMANTIC EXPERIMENT] Indexed {chunk_count} code chunks in "
            f"{index_duration:.4f}s"
        )

        # 1. Query: exact symbol
        t_start = time.perf_counter()
        exact_res = await engine.search_hybrid("WorkspaceManager", top_k=5)
        exact_latency_ms = (time.perf_counter() - t_start) * 1000.0

        assert len(exact_res) > 0
        assert exact_res[0].chunk.symbol_name == "WorkspaceManager"
        print(
            f"[SEMANTIC EXPERIMENT] Exact Query 'WorkspaceManager' took "
            f"{exact_latency_ms:.2f}ms -> Top: {exact_res[0].chunk.symbol_name}"
        )

        # 2. Query: natural language semantic description
        t_start = time.perf_counter()
        sem_res = await engine.search_hybrid(
            "asynchronously execute command inside docker container and stream lines",
            top_k=5,
        )
        sem_latency_ms = (time.perf_counter() - t_start) * 1000.0

        assert len(sem_res) > 0
        top = sem_res[0]
        print(
            f"[SEMANTIC EXPERIMENT] Semantic Query took {sem_latency_ms:.2f}ms "
            f"-> Top: {top.chunk.symbol_name} ({top.match_type.value}, "
            f"score: {top.score:.3f})"
        )
        assert any("execute_command" in r.chunk.symbol_name for r in sem_res)

    import asyncio

    asyncio.run(_run())
