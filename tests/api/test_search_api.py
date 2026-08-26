from __future__ import annotations

from starlette.testclient import TestClient


def test_semantic_search_api_endpoint(client: TestClient) -> None:
    """Verifies POST /search/semantic returns matches for a natural language query."""
    response = client.post(
        "/search/semantic",
        json={
            "query": "asynchronously execute command",
            "top_k": 3,
            "path": "tests/fixtures/sample_repo",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "asynchronously execute command"
    assert "results" in data
    assert isinstance(data["results"], list)


def test_hybrid_search_api_endpoint(client: TestClient) -> None:
    """Verifies POST /search/hybrid prioritizes exact symbol match."""
    response = client.post(
        "/search/hybrid",
        json={
            "query": "Calculator",
            "top_k": 3,
            "path": "tests/fixtures/sample_repo",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Calculator"
    assert data["count"] > 0
    assert data["results"][0]["symbol_name"] == "Calculator"
