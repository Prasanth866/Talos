from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.indexer.indexer import CodeIndexer
from src.indexer.search import HybridSearchEngine
from src.tools.exceptions import PatchError, PathTraversalError

logger = structlog.get_logger(__name__)


@dataclass
class Hunk:
    """Represents a single diff hunk within a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    """Represents changes to a single file parsed from a unified diff."""

    old_file: str
    new_file: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_delete: bool = False


def _clean_diff_path(raw_path: str) -> str:
    """Strips a/ or b/ prefixes and normalizes diff path."""
    cleaned = raw_path.strip()
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        cleaned = cleaned[2:]
    return cleaned


def parse_unified_diff(diff_str: str) -> list[FilePatch]:
    """Parses a unified diff string into structured FilePatch objects."""
    if not diff_str.strip():
        raise PatchError("Unified diff string is empty", reason="EMPTY_DIFF")

    file_patches: list[FilePatch] = []
    current_patch: FilePatch | None = None
    current_hunk: Hunk | None = None

    lines = diff_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match header lines: --- a/path/to/file or --- /dev/null
        if line.startswith("--- "):
            old_file_raw = line[4:].strip()
            # Expect next line to be +++ b/path/to/file
            if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
                new_file_raw = lines[i + 1][4:].strip()
                i += 1  # Skip +++ line

                is_new = "/dev/null" in old_file_raw
                is_delete = "/dev/null" in new_file_raw

                old_file = "/dev/null" if is_new else _clean_diff_path(old_file_raw)
                new_file = "/dev/null" if is_delete else _clean_diff_path(new_file_raw)

                if current_patch:
                    file_patches.append(current_patch)

                current_patch = FilePatch(
                    old_file=old_file,
                    new_file=new_file,
                    is_new=is_new,
                    is_delete=is_delete,
                )
                current_hunk = None
            i += 1
            continue

        # Match hunk header: @@ -old_start,old_count +new_start,new_count @@
        hunk_match = re.match(
            r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line
        )
        if hunk_match:
            if not current_patch:
                raise PatchError(
                    f"Hunk header found without file header at line {i + 1}: '{line}'",
                    reason="MALFORMED_HEADER",
                )
            old_start = int(hunk_match.group(1))
            old_count = (
                int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            )
            new_start = int(hunk_match.group(3))
            new_count = (
                int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1
            )

            current_hunk = Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
            current_patch.hunks.append(current_hunk)
            i += 1
            continue

        # Inside a hunk
        if current_hunk is not None:
            if line.startswith(("+", "-", " ")) or line == "":
                current_hunk.lines.append(line if line else " ")
            elif line.startswith("\\ No newline at end of file"):
                pass  # Skip EOF newline markers
            else:
                # Outside of hunk or garbage line
                current_hunk = None

        i += 1

    if current_patch:
        file_patches.append(current_patch)

    if not file_patches:
        raise PatchError(
            "No valid unified diff hunks or file headers recognized",
            reason="NO_VALID_HUNKS",
        )

    return file_patches


def apply_hunks_to_content(
    original_content: str, hunks: list[Hunk], file_path: str
) -> str:
    """Applies a list of hunks in-memory to original file content.

    Validates context lines and line numbers strictly.
    """
    orig_lines = original_content.splitlines(keepends=False) if original_content else []
    patched_lines: list[str] = []
    orig_idx = 0

    for hunk in hunks:
        # 1-based start line to 0-based index
        target_idx = max(0, hunk.old_start - 1)

        # Copy unchanged lines before the hunk
        while orig_idx < target_idx and orig_idx < len(orig_lines):
            patched_lines.append(orig_lines[orig_idx])
            orig_idx += 1

        # Process hunk lines
        for hunk_line in hunk.lines:
            if not hunk_line:
                continue
            marker = hunk_line[0]
            content = hunk_line[1:]

            if marker == " ":  # Context line: must match original
                if orig_idx >= len(orig_lines):
                    raise PatchError(
                        f"Hunk @@ -{hunk.old_start},{hunk.old_count} failed: "
                        f"unexpected EOF in '{file_path}' "
                        f"(expected context '{content}')",
                        reason="CONFLICT",
                        context=f"Expected: '{content}', Found EOF",
                    )
                if orig_lines[orig_idx] != content:
                    raise PatchError(
                        f"Hunk @@ -{hunk.old_start},{hunk.old_count} "
                        f"conflict in '{file_path}' at line {orig_idx + 1}",
                        reason="CONTEXT_MISMATCH",
                        context=(
                            f"Expected: '{content}'\nActual:   '{orig_lines[orig_idx]}'"
                        ),
                    )
                patched_lines.append(orig_lines[orig_idx])
                orig_idx += 1

            elif marker == "-":  # Removal line: must match original
                if orig_idx >= len(orig_lines):
                    raise PatchError(
                        f"Hunk @@ -{hunk.old_start},{hunk.old_count} failed in "
                        f"'{file_path}': unexpected EOF removing '{content}'",
                        reason="CONFLICT",
                        context=f"Cannot remove '{content}' at EOF",
                    )
                if orig_lines[orig_idx] != content:
                    raise PatchError(
                        f"Hunk deletion mismatch in '{file_path}' "
                        f"at line {orig_idx + 1}",
                        reason="DELETION_MISMATCH",
                        context=(
                            f"Expected to delete: '{content}'\n"
                            f"Actual content:     '{orig_lines[orig_idx]}'"
                        ),
                    )
                orig_idx += 1  # Skip removed line

            elif marker == "+":  # Addition line
                patched_lines.append(content)

    # Append remaining lines
    while orig_idx < len(orig_lines):
        patched_lines.append(orig_lines[orig_idx])
        orig_idx += 1

    return "\n".join(patched_lines) + ("\n" if patched_lines else "")


class PatchTool:
    """Applies unified diff patches with dry-run validation and auto re-indexing."""

    def __init__(
        self,
        sandbox_dir: Path | str,
        indexer: CodeIndexer | None = None,
        search_engine: HybridSearchEngine | None = None,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.indexer = indexer
        self.search_engine = search_engine

    def _resolve_sandbox_path(self, rel_path: str) -> Path:
        """Ensures target path resides strictly inside the sandbox directory."""
        clean = _clean_diff_path(rel_path)
        resolved = (self.sandbox_dir / clean).resolve()
        try:
            resolved.relative_to(self.sandbox_dir)
        except ValueError as exc:
            raise PathTraversalError(
                f"Patch path '{rel_path}' escapes sandbox '{self.sandbox_dir}'",
                tool_name="apply_patch",
                attempted_path=str(resolved),
            ) from exc
        return resolved

    async def apply_patch(self, patch: str, dry_run: bool = False) -> dict[str, Any]:
        """Validates and applies a unified diff patch to disk with auto re-indexing."""
        file_patches = parse_unified_diff(patch)

        planned_changes: list[dict[str, Any]] = []
        total_added = 0
        total_removed = 0

        # Phase 1: Dry-Run In-Memory Validation across all file patches
        for file_patch in file_patches:
            target_rel = (
                file_patch.new_file if not file_patch.is_delete else file_patch.old_file
            )
            target_abs = self._resolve_sandbox_path(target_rel)

            if file_patch.is_new:
                if target_abs.exists():
                    logger.warning(
                        "patch.new_file_exists_overwriting",
                        file=str(target_rel),
                    )
                orig_content = ""
            elif file_patch.is_delete:
                if not target_abs.exists():
                    raise PatchError(
                        f"Cannot delete non-existent file '{target_rel}'",
                        reason="FILE_NOT_FOUND",
                    )
                orig_content = target_abs.read_text(encoding="utf-8", errors="replace")
            else:
                if not target_abs.exists():
                    raise PatchError(
                        f"Target file to patch '{target_rel}' does not exist",
                        reason="FILE_NOT_FOUND",
                    )
                orig_content = target_abs.read_text(encoding="utf-8", errors="replace")

            # Apply hunks in memory
            new_content = apply_hunks_to_content(
                orig_content, file_patch.hunks, file_path=target_rel
            )

            # Count additions and deletions
            for h in file_patch.hunks:
                for line in h.lines:
                    if line.startswith("+"):
                        total_added += 1
                    elif line.startswith("-"):
                        total_removed += 1

            planned_changes.append(
                {
                    "file_patch": file_patch,
                    "target_abs": target_abs,
                    "target_rel": target_rel,
                    "new_content": new_content,
                }
            )

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": (
                    f"Dry run successful for {len(file_patches)} file(s). "
                    f"+{total_added}/-{total_removed} lines."
                ),
                "files_validated": [c["target_rel"] for c in planned_changes],
                "lines_added": total_added,
                "lines_removed": total_removed,
            }

        # Phase 2: Atomic Disk Application
        modified_files: list[str] = []
        for change in planned_changes:
            fp: FilePatch = change["file_patch"]
            path_abs: Path = change["target_abs"]
            path_rel: str = change["target_rel"]

            if fp.is_delete:
                if path_abs.exists():
                    path_abs.unlink()
            else:
                path_abs.parent.mkdir(parents=True, exist_ok=True)
                path_abs.write_text(change["new_content"], encoding="utf-8")

            modified_files.append(path_rel)

        # Phase 3: Incremental Auto Re-Indexing
        reindexed_files: list[str] = []
        for change in planned_changes:
            fp = change["file_patch"]
            path_abs = change["target_abs"]
            path_rel = change["target_rel"]

            if fp.is_delete:
                continue

            # 1. Update AST CodeIndexer
            if self.indexer is not None:
                try:
                    self.indexer.index_file(path_abs)
                    reindexed_files.append(path_rel)
                except Exception as exc:
                    logger.warning(
                        "indexer.reindex_failed",
                        file=str(path_abs),
                        error=str(exc),
                    )

            # 2. Update Vector Semantic HybridSearchEngine
            if self.search_engine is not None:
                try:
                    await self.search_engine.index_file(path_abs)
                except Exception as exc:
                    logger.warning(
                        "search_engine.reindex_failed",
                        file=str(path_abs),
                        error=str(exc),
                    )

        logger.info(
            "patch.applied_successfully",
            files=modified_files,
            lines_added=total_added,
            lines_removed=total_removed,
            reindexed=reindexed_files,
        )

        return {
            "success": True,
            "dry_run": False,
            "message": (
                f"Successfully applied patch to {len(modified_files)} file(s). "
                f"+{total_added}/-{total_removed} lines. "
                f"Incrementally re-indexed {len(reindexed_files)} file(s)."
            ),
            "files_modified": modified_files,
            "lines_added": total_added,
            "lines_removed": total_removed,
            "reindexed_files": reindexed_files,
        }
