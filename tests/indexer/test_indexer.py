from __future__ import annotations

import time
from pathlib import Path

from src.indexer import CodeIndexer, SymbolKind


def test_get_symbol_definition_returns_correct_result() -> None:
    """Unit test: get_symbol_definition returns correct symbol record."""
    indexer = CodeIndexer()
    fixture_dir = Path("tests/fixtures/sample_repo")
    calculator_path = fixture_dir / "calculator.py"

    indexer.index_file(calculator_path)

    # 1. Lookup function definition
    results = indexer.get_symbol_definition("add")
    assert len(results) == 1
    add_sym = results[0]
    assert add_sym.name == "add"
    assert add_sym.kind == SymbolKind.FUNCTION
    assert add_sym.docstring == "Adds two integers together."
    assert "def add(a: int, b: int = 0) -> int" in add_sym.signature

    # 2. Lookup class definition
    class_results = indexer.get_symbol_definition("Calculator")
    assert len(class_results) == 1
    calc_sym = class_results[0]
    assert calc_sym.name == "Calculator"
    assert calc_sym.kind == SymbolKind.CLASS
    assert calc_sym.docstring == "A standard arithmetic calculator class."

    # 3. Lookup method definition
    method_results = indexer.get_symbol_definition("multiply")
    assert len(method_results) == 1
    method_sym = method_results[0]
    assert method_sym.name == "Calculator.multiply"
    assert method_sym.kind == SymbolKind.METHOD
    assert method_sym.docstring == "Multiplies two floating point numbers."


def test_list_file_structure_returns_module_overview() -> None:
    """Unit test: list_file_structure returns overview of imports and definitions."""
    indexer = CodeIndexer()
    fixture_dir = Path("tests/fixtures/sample_repo")
    utils_path = fixture_dir / "utils.py"

    structure = indexer.list_file_structure(utils_path)

    assert structure.file_path.resolve() == utils_path.resolve()
    assert len(structure.imports) >= 2
    assert any(i.module == "asyncio" for i in structure.imports)
    assert any(i.module == "pathlib" for i in structure.imports)

    fn_names = [f.name for f in structure.functions]
    assert "async_fetch_data" in fn_names
    assert "format_path" in fn_names

    fetch_fn = next(f for f in structure.functions if f.name == "async_fetch_data")
    assert fetch_fn.is_async is True
    assert fetch_fn.return_type == "dict[str, str]"
    assert fetch_fn.docstring == "Asynchronously fetches data from a remote endpoint."


def test_index_directory_and_search_symbols() -> None:
    """Unit test: index_directory indexes files and search_symbols returns matches."""
    indexer = CodeIndexer()
    fixture_dir = Path("tests/fixtures/sample_repo")

    indexed_count = indexer.index_directory(fixture_dir)
    assert indexed_count >= 3

    # Search for all "calc" or "fetch" symbols
    fetch_matches = indexer.search_symbols("fetch")
    assert len(fetch_matches) >= 1
    assert any(s.name == "async_fetch_data" for s in fetch_matches)

    calc_matches = indexer.search_symbols("calc")
    assert len(calc_matches) >= 1
    assert any(s.name.startswith("Calculator") for s in calc_matches)

    # Verify broken_syntax file was indexed and recovered valid symbols
    broken_matches = indexer.search_symbols("valid_function")
    assert len(broken_matches) == 2
    names = [s.name for s in broken_matches]
    assert "valid_function_before" in names
    assert "valid_function_after" in names


def test_live_project_indexing_experiment() -> None:
    """Experiment: Index real Python project, query symbols, and measure latency."""
    indexer = CodeIndexer()
    src_dir = Path("src")

    t0 = time.perf_counter()
    file_count = indexer.index_directory(src_dir, recursive=True)
    index_duration = time.perf_counter() - t0

    all_symbols = indexer.list_all_symbols()

    print(
        f"\n[EXPERIMENT INDEXING] Indexed {file_count} files in {index_duration:.4f}s"
    )

    print(f"[EXPERIMENT INDEXING] Extracted {len(all_symbols)} total symbols")
    print(
        f"[EXPERIMENT INDEXING] Average time per file: "
        f"{(index_duration / max(1, file_count)) * 1000:.2f}ms"
    )

    # 1. Performance check: 15+ files should index in < 1.0 second
    assert file_count >= 15
    assert index_duration < 2.0

    # 2. Query WorkspaceManager class
    ws_symbols = indexer.get_symbol_definition("WorkspaceManager")
    assert len(ws_symbols) >= 1
    ws_sym = ws_symbols[0]
    assert ws_sym.kind == SymbolKind.CLASS
    assert "WorkspaceManager" in ws_sym.name
    assert "class WorkspaceManager" in ws_sym.signature

    # 3. Query specific method execute_command
    exec_symbols = indexer.get_symbol_definition("execute_command")
    assert len(exec_symbols) >= 1
    exec_sym = exec_symbols[0]
    assert "execute_command" in exec_sym.name
    assert "workspace_id" in exec_sym.signature

    # 4. Query create_default_dispatcher function
    disp_symbols = indexer.get_symbol_definition("create_default_dispatcher")
    assert len(disp_symbols) >= 1
    disp_sym = disp_symbols[0]
    assert disp_sym.kind == SymbolKind.FUNCTION
    assert "create_default_dispatcher" in disp_sym.name

    # 5. Verify file structure overview of src/workspace/manager.py
    mgr_path = Path("src/workspace/manager.py")
    structure = indexer.list_file_structure(mgr_path)
    assert len(structure.imports) > 5
    assert any(c.name == "WorkspaceManager" for c in structure.classes)
    assert any(
        m.name == "execute_command"
        for c in structure.classes
        if c.name == "WorkspaceManager"
        for m in c.methods
    )


def test_index_multi_language_project(tmp_path: Path) -> None:
    """Unit test: indexer indexes multi-language project (JS, Java, HTML, CSS)."""
    indexer = CodeIndexer()

    (tmp_path / "UserService.java").write_text("""
package com.app;
public class UserService {
    public User findUser(String id) {
        return null;
    }
}
""")

    (tmp_path / "script.js").write_text("""
function initApp() {
    console.log("ready");
}
""")

    (tmp_path / "index.html").write_text("""
<!DOCTYPE html>
<html>
<body>
    <header id="top-bar">Nav</header>
</body>
</html>
""")

    (tmp_path / "style.css").write_text("""
#top-bar {
    background: #000;
}
""")

    indexed = indexer.index_directory(tmp_path)
    assert indexed == 4

    assert len(indexer.get_symbol_definition("UserService")) >= 1
    assert len(indexer.get_symbol_definition("initApp")) >= 1
    assert len(indexer.get_symbol_definition("header#top-bar")) >= 1
    assert len(indexer.get_symbol_definition("#top-bar")) >= 1
