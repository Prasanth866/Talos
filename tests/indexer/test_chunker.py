from __future__ import annotations

from pathlib import Path

from src.indexer.chunker import ASTChunker
from src.indexer.models import SymbolKind


def test_ast_chunking_produces_correct_units() -> None:
    """Unit test: extracts functions, classes, and methods as semantic chunks."""
    chunker = ASTChunker()

    fixture_dir = Path("tests/fixtures/sample_repo")
    calculator_path = fixture_dir / "calculator.py"

    chunks = chunker.chunk_file(calculator_path)
    assert len(chunks) >= 3

    # Check top-level add function chunk
    add_chunks = [c for c in chunks if c.symbol_name == "add"]
    assert len(add_chunks) == 1
    add_chunk = add_chunks[0]
    assert add_chunk.kind == SymbolKind.FUNCTION
    assert "def add(a: int, b: int = 0) -> int" in add_chunk.signature
    assert add_chunk.docstring == "Adds two integers together."
    assert "return a + b" in add_chunk.code_content
    assert "File: calculator.py" in add_chunk.embedding_text
    assert add_chunk.token_count > 0

    # Check Calculator class chunk
    class_chunks = [c for c in chunks if c.symbol_name == "Calculator"]
    assert len(class_chunks) == 1
    calc_chunk = class_chunks[0]
    assert calc_chunk.kind == SymbolKind.CLASS
    assert calc_chunk.docstring == "A standard arithmetic calculator class."

    # Check member method chunk
    method_chunks = [c for c in chunks if c.symbol_name == "Calculator.multiply"]
    assert len(method_chunks) == 1
    mult_chunk = method_chunks[0]
    assert mult_chunk.kind == SymbolKind.METHOD
    assert "Multiplies two floating point numbers." in mult_chunk.docstring  # type: ignore[operator]


def test_chunk_directory_recursive() -> None:
    """Unit test: chunk_directory processes all fixture files."""
    chunker = ASTChunker()
    fixture_dir = Path("tests/fixtures/sample_repo")

    chunks = chunker.chunk_directory(fixture_dir)
    assert len(chunks) >= 5

    symbol_names = [c.symbol_name for c in chunks]
    assert "add" in symbol_names
    assert "Calculator" in symbol_names
    assert "Calculator.multiply" in symbol_names
    assert "async_fetch_data" in symbol_names
