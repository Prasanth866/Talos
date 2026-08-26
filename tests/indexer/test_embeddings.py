from __future__ import annotations

import math

import pytest

from src.indexer.embeddings import (
    EmbeddingCostTracker,
    MockEmbeddingClient,
)


@pytest.mark.asyncio
async def test_mock_embedding_generation_and_cosine_similarity() -> None:
    """Unit test: generates normalized vectors with expected similarity."""
    client = MockEmbeddingClient(dimension=1536)

    # 1. Dimension and unit norm
    vector = await client.embed_query("function that calculates sum and addition")
    assert len(vector) == 1536
    squared_sum = sum(x * x for x in vector)
    assert math.isclose(squared_sum, 1.0, rel_tol=1e-4)

    # 2. Semantic similarity: query with overlapping terms vs completely unrelated query
    calc_chunk_text = "def add(a: int, b: int) -> int: return a + b # calculates sum"
    unrelated_text = "class DatabaseConnection: host: str, port: int"

    calc_emb = await client.embed_query(calc_chunk_text)
    unrelated_emb = await client.embed_query(unrelated_text)
    query_emb = await client.embed_query("function that calculates sum")

    # Cosine similarity (dot product of normalized vectors)
    calc_sim = sum(q * c for q, c in zip(query_emb, calc_emb, strict=False))
    unrelated_sim = sum(q * u for q, u in zip(query_emb, unrelated_emb, strict=False))

    assert calc_sim > unrelated_sim
    assert calc_sim > 0.2


@pytest.mark.asyncio
async def test_embedding_cost_tracking_and_logging() -> None:
    """Unit test: EmbeddingCostTracker records tokens and calculates dollar costs."""
    tracker = EmbeddingCostTracker(cost_per_million_tokens=0.02)
    client = MockEmbeddingClient(cost_tracker=tracker)

    texts = [
        "short text snippet",
        (
            "another relatively longer function implementation with arguments "
            "and docstrings"
        ),
    ]
    embeddings = await client.embed_texts(texts)
    assert len(embeddings) == 2

    summary = tracker.log_summary(context="test_run")

    assert summary["total_chunks"] == 2
    assert summary["total_tokens"] > 0
    assert summary["total_cost_usd"] >= 0.0
    assert summary["duration_seconds"] >= 0.0


def test_gemini_embedding_client_properties() -> None:
    """Unit test: GeminiEmbeddingClient properties and configuration."""
    from src.indexer.embeddings import GeminiEmbeddingClient

    client = GeminiEmbeddingClient(
        api_key="test-api-key",
        model="text-embedding-004",
        dimension=768,
    )
    assert client.dimension == 768
    assert client.model == "text-embedding-004"
    assert client.api_key == "test-api-key"


def test_create_default_embedding_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit test: create_default_embedding_client picks Gemini when key is present."""
    from src.indexer.embeddings import (
        GeminiEmbeddingClient,
        MockEmbeddingClient,
        create_default_embedding_client,
    )

    # 1. No key -> returns MockEmbeddingClient
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client_mock = create_default_embedding_client()
    assert isinstance(client_mock, MockEmbeddingClient)

    # 2. Key provided -> returns GeminiEmbeddingClient
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    client_gemini = create_default_embedding_client()
    assert isinstance(client_gemini, GeminiEmbeddingClient)
    assert client_gemini.api_key == "test-gemini-key"
