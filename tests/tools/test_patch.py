from __future__ import annotations

from pathlib import Path

import pytest

from src.indexer.chunker import ASTChunker
from src.indexer.embeddings import MockEmbeddingClient
from src.indexer.indexer import CodeIndexer
from src.indexer.search import HybridSearchEngine
from src.indexer.vector_store import InMemoryVectorStore
from src.tools.exceptions import PatchError, PathTraversalError
from src.tools.patch import PatchTool, parse_unified_diff


def test_parse_unified_diff_basic() -> None:
    """Unit test: parse_unified_diff parses standard unified diff headers and hunks."""
    diff_text = """--- a/src/calc.py
+++ b/src/calc.py
@@ -1,3 +1,4 @@
 def add(a, b):
-    return a - b
+    # Fixed addition
+    return a + b
"""
    file_patches = parse_unified_diff(diff_text)
    assert len(file_patches) == 1
    fp = file_patches[0]
    assert fp.old_file == "src/calc.py"
    assert fp.new_file == "src/calc.py"
    assert len(fp.hunks) == 1
    hunk = fp.hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 3
    assert hunk.new_start == 1
    assert hunk.new_count == 4


async def test_valid_patch_applies_correctly(tmp_path: Path) -> None:
    """Unit test: Valid patch modifies file on disk correctly."""
    target_file = tmp_path / "hello.py"
    target_file.write_text("def hello():\n    print('old')\n", encoding="utf-8")

    patch_text = """--- a/hello.py
+++ b/hello.py
@@ -1,2 +1,2 @@
 def hello():
-    print('old')
+    print('new')
"""
    tool = PatchTool(sandbox_dir=tmp_path)
    result = await tool.apply_patch(patch_text, dry_run=False)

    assert result["success"]
    assert not result["dry_run"]
    assert "hello.py" in result["files_modified"]
    assert result["lines_added"] == 1
    assert result["lines_removed"] == 1

    content = target_file.read_text(encoding="utf-8")
    assert "print('new')" in content
    assert "print('old')" not in content


async def test_invalid_patch_returns_patch_error_without_modifying_files(
    tmp_path: Path,
) -> None:
    """Unit test: Conflicting patch raises PatchError and leaves disk file untouched."""
    target_file = tmp_path / "app.py"
    original_content = "def run():\n    return 'original'\n"
    target_file.write_text(original_content, encoding="utf-8")

    conflicting_patch = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 def run():
-    return 'mismatched_context'
+    return 'updated'
"""
    tool = PatchTool(sandbox_dir=tmp_path)

    with pytest.raises(PatchError) as exc_info:
        await tool.apply_patch(conflicting_patch, dry_run=False)

    assert exc_info.value.code == "PATCH_ERROR"
    assert exc_info.value.reason in ("CONTEXT_MISMATCH", "DELETION_MISMATCH")

    assert target_file.read_text(encoding="utf-8") == original_content


async def test_dry_run_validation_catches_conflicting_patches(
    tmp_path: Path,
) -> None:
    """Unit test: dry_run=True validates patches in-memory without writing to disk."""
    target_file = tmp_path / "server.py"
    original_content = "def start():\n    pass\n"
    target_file.write_text(original_content, encoding="utf-8")

    valid_patch = """--- a/server.py
+++ b/server.py
@@ -1,2 +1,3 @@
 def start():
+    print('Starting')
     pass
"""
    tool = PatchTool(sandbox_dir=tmp_path)
    dry_res = await tool.apply_patch(valid_patch, dry_run=True)
    assert dry_res["success"]
    assert dry_res["dry_run"]

    assert target_file.read_text(encoding="utf-8") == original_content

    conflict_patch = """--- a/server.py
+++ b/server.py
@@ -1,2 +1,2 @@
 def stop():
-    pass
+    print('stopped')
"""
    with pytest.raises(PatchError):
        await tool.apply_patch(conflict_patch, dry_run=True)


async def test_search_index_updated_after_patch_application(
    tmp_path: Path,
) -> None:
    """Unit test: AST symbol index and vector search engine reflect patched code."""
    code_file = tmp_path / "service.py"
    code_file.write_text("def old_calculate():\n    return 42\n", encoding="utf-8")

    indexer = CodeIndexer()
    indexer.index_directory(tmp_path)

    vector_store = InMemoryVectorStore()
    embedding_client = MockEmbeddingClient()
    search_engine = HybridSearchEngine(
        indexer=indexer,
        embedding_client=embedding_client,
        vector_store=vector_store,
        chunker=ASTChunker(),
    )
    await search_engine.index_directory(tmp_path)

    syms_before = indexer.get_symbol_definition("old_calculate")
    assert len(syms_before) == 1

    patch_text = """--- a/service.py
+++ b/service.py
@@ -1,2 +1,3 @@
-def old_calculate():
-    return 42
+def payment_processor(amount: float) -> bool:
+    \"\"\"Processes user payments securely.\"\"\"
+    return True
"""
    tool = PatchTool(
        sandbox_dir=tmp_path,
        indexer=indexer,
        search_engine=search_engine,
    )
    res = await tool.apply_patch(patch_text, dry_run=False)
    assert res["success"]
    assert "service.py" in res["reindexed_files"]

    syms_after = indexer.get_symbol_definition("payment_processor")
    assert len(syms_after) == 1
    assert syms_after[0].name == "payment_processor"

    old_syms = indexer.get_symbol_definition("old_calculate")
    assert len(old_syms) == 0

    search_results = await search_engine.search_hybrid("payment processor")
    assert len(search_results) > 0
    assert any("payment_processor" in r.chunk.symbol_name for r in search_results)


async def test_file_creation_and_deletion_via_patch(tmp_path: Path) -> None:
    """Unit test: Unified diff can create new files and delete existing files."""
    tool = PatchTool(sandbox_dir=tmp_path)

    create_patch = """--- /dev/null
+++ b/new_module.py
@@ -0,0 +1,2 @@
+def greet():
+    return "Hi"
"""
    create_res = await tool.apply_patch(create_patch, dry_run=False)
    assert create_res["success"]
    new_file = tmp_path / "new_module.py"
    assert new_file.exists()
    assert "def greet()" in new_file.read_text(encoding="utf-8")

    delete_patch = """--- a/new_module.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def greet():
-    return "Hi"
"""
    delete_res = await tool.apply_patch(delete_patch, dry_run=False)
    assert delete_res["success"]
    assert not new_file.exists()


async def test_path_traversal_in_patch_is_blocked(tmp_path: Path) -> None:
    """Unit test: Patch attempting to escape sandbox raises PathTraversalError."""
    traversal_patch = """--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -1,1 +1,1 @@
-root
+hacked
"""
    tool = PatchTool(sandbox_dir=tmp_path)
    with pytest.raises(PathTraversalError):
        await tool.apply_patch(traversal_patch, dry_run=False)
