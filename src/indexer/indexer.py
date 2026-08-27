from __future__ import annotations

import contextlib
from pathlib import Path

from src.indexer.models import (
    FileStructure,
    ImportDefinition,
    Symbol,
)
from src.indexer.parser import (
    EXTENSION_LANGUAGE_MAP,
    TreeSitterParser,
)


class CodeIndexer:
    """In-memory code indexer managing multi-language symbol indexes."""

    def __init__(self, parser: TreeSitterParser | None = None) -> None:
        self.parser = parser or TreeSitterParser()
        self._files: dict[Path, FileStructure] = {}
        self._symbols_by_name: dict[str, list[Symbol]] = {}

    def index_source(
        self, file_path: Path | str, source_code: str | bytes
    ) -> FileStructure:
        """Indexes source code directly for a given file path."""
        path_obj = Path(file_path).resolve()
        structure = self.parser.parse(path_obj, source_code)
        self._register_file_structure(path_obj, structure)
        return structure

    def index_file(self, file_path: Path | str) -> FileStructure:
        """Reads and indexes a single source code file from disk."""
        path_obj = Path(file_path).resolve()
        content = path_obj.read_bytes()
        return self.index_source(path_obj, content)

    def index_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> int:
        """Indexes all supported code files in the specified directory."""
        dir_obj = Path(directory).resolve()
        if not dir_obj.exists() or not dir_obj.is_dir():
            return 0

        ext_set = (
            {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in extensions
            }
            if extensions is not None
            else set(EXTENSION_LANGUAGE_MAP.keys())
        )

        indexed_count = 0
        files = dir_obj.rglob("*") if recursive else dir_obj.glob("*")
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in ext_set:
                with contextlib.suppress(Exception):
                    self.index_file(file_path)
                    indexed_count += 1
        return indexed_count

    def _register_file_structure(
        self, path_obj: Path, structure: FileStructure
    ) -> None:
        """Stores file structure and updates the global symbol lookup table."""
        if path_obj in self._files:
            old_structure = self._files[path_obj]
            for sym_name in old_structure.symbols:
                if sym_name in self._symbols_by_name:
                    self._symbols_by_name[sym_name] = [
                        s
                        for s in self._symbols_by_name[sym_name]
                        if s.file_path != path_obj
                    ]
                    if not self._symbols_by_name[sym_name]:
                        del self._symbols_by_name[sym_name]

        self._files[path_obj] = structure
        for sym_name, sym in structure.symbols.items():
            if sym_name not in self._symbols_by_name:
                self._symbols_by_name[sym_name] = []
            self._symbols_by_name[sym_name].append(sym)

    def get_symbol_definition(
        self, name: str, file_path: Path | str | None = None
    ) -> list[Symbol]:
        """Looks up symbol definitions across the index or in a specific file."""

        if file_path is not None:
            path_obj = Path(file_path).resolve()
            if path_obj not in self._files and path_obj.exists():
                self.index_file(path_obj)
            structure = self._files.get(path_obj)
            if not structure:
                return []
            if name in structure.symbols:
                return [structure.symbols[name]]
            matches = [
                s
                for s in structure.symbols.values()
                if s.name == name or s.name.endswith(f".{name}")
            ]
            return matches

        if name in self._symbols_by_name:
            return list(self._symbols_by_name[name])

        matches = []
        for sym_name, symbols in self._symbols_by_name.items():
            if sym_name.endswith(f".{name}") or sym_name.lower() == name.lower():
                matches.extend(symbols)
        return matches

    def list_file_structure(self, file_path: Path | str) -> FileStructure:
        """Returns the complete structural overview of a given file."""
        path_obj = Path(file_path).resolve()
        if path_obj not in self._files:
            if path_obj.exists():
                return self.index_file(path_obj)
            return FileStructure(file_path=path_obj)
        return self._files[path_obj]

    def list_all_symbols(self) -> list[Symbol]:
        """Returns all registered symbols across all indexed files."""
        all_symbols: list[Symbol] = []
        for symbols in self._symbols_by_name.values():
            all_symbols.extend(symbols)
        return all_symbols

    def search_symbols(self, query: str) -> list[Symbol]:
        """Searches symbols matching the query string (case-insensitive substring)."""
        query_lower = query.lower()
        results: list[Symbol] = []
        for sym_name, symbols in self._symbols_by_name.items():
            if query_lower in sym_name.lower():
                results.extend(symbols)
        return results

    def get_file_imports(self, file_path: Path | str) -> list[ImportDefinition]:
        """Returns all import definitions extracted from a file."""
        structure = self.list_file_structure(file_path)
        return structure.imports

    def clear(self) -> None:
        """Clears all indexed files and symbols."""
        self._files.clear()
        self._symbols_by_name.clear()
