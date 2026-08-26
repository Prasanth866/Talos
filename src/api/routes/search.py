from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.config import ROOT_DIR
from src.indexer.embeddings import create_default_embedding_client
from src.indexer.search import HybridSearchEngine

router = APIRouter(prefix="/search", tags=["Search"])

_search_engine = HybridSearchEngine(embedding_client=create_default_embedding_client())


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="Code or natural language search query"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")
    path: str = Field(
        default="src", description="Relative directory path to search within"
    )


class SearchItem(BaseModel):
    symbol_name: str
    file_path: str
    kind: str
    signature: str
    docstring: str | None
    line_span: dict[str, int]
    score: float
    match_type: str
    explanation: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchItem]


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search_endpoint(payload: SearchRequest) -> dict[str, Any]:
    """Performs dense vector similarity search for natural language queries."""
    target_dir = (ROOT_DIR / payload.path).resolve()
    if target_dir.exists() and target_dir.is_dir():
        await _search_engine.index_directory(target_dir)

    results = await _search_engine.search_semantic(payload.query, top_k=payload.top_k)
    items = [r.to_dict() for r in results]
    return {
        "query": payload.query,
        "count": len(items),
        "results": items,
    }


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search_endpoint(payload: SearchRequest) -> dict[str, Any]:
    """Performs hybrid exact symbol & semantic similarity search."""
    target_dir = (ROOT_DIR / payload.path).resolve()
    if target_dir.exists() and target_dir.is_dir():
        await _search_engine.index_directory(target_dir)

    results = await _search_engine.search_hybrid(payload.query, top_k=payload.top_k)
    items = [r.to_dict() for r in results]
    return {
        "query": payload.query,
        "count": len(items),
        "results": items,
    }
