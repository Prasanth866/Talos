from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.indexer.models import (
    ClassDefinition,
    FileStructure,
    FunctionDefinition,
    LineSpan,
    SymbolKind,
)
from src.indexer.parser import (
    EXTENSION_LANGUAGE_MAP,
    TreeSitterParser,
)


@dataclass
class CodeChunk:
    """A semantic chunk of code derived from AST definitions."""

    chunk_id: str
    file_path: Path
    symbol_name: str
    kind: SymbolKind
    signature: str
    docstring: str | None
    code_content: str
    line_span: LineSpan
    embedding_text: str
    token_count: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts chunk to a serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": str(self.file_path),
            "symbol_name": self.symbol_name,
            "kind": self.kind.value,
            "signature": self.signature,
            "docstring": self.docstring,
            "code_content": self.code_content,
            "line_span": self.line_span.to_dict(),
            "embedding_text": self.embedding_text,
            "token_count": self.token_count,
            "has_embedding": self.embedding is not None,
            "metadata": self.metadata,
        }


class ASTChunker:
    """Splits source code into contextual semantic chunks from AST units."""

    def __init__(self, parser: TreeSitterParser | None = None) -> None:
        self.parser = parser or TreeSitterParser()

    def chunk_file(
        self, file_path: Path | str, source_code: str | None = None
    ) -> list[CodeChunk]:
        """Parses and chunks a source file into AST semantic units."""
        path_obj = Path(file_path).resolve()
        if source_code is None:
            if not path_obj.exists():
                return []
            source_code = path_obj.read_text(encoding="utf-8", errors="replace")

        structure = self.parser.parse(path_obj, source_code)
        return self.chunk_structure(structure, source_code)

    def chunk_structure(
        self, structure: FileStructure, source_code: str
    ) -> list[CodeChunk]:
        """Extracts code chunks from an existing FileStructure and raw source."""
        lines = source_code.splitlines()
        chunks: list[CodeChunk] = []

        for fn in structure.functions:
            chunk = self._create_function_chunk(structure.file_path, fn, lines)
            chunks.append(chunk)

        for cls in structure.classes:
            class_chunk = self._create_class_chunk(structure.file_path, cls, lines)
            chunks.append(class_chunk)

            for method in cls.methods:
                method_chunk = self._create_function_chunk(
                    structure.file_path, method, lines, parent_class=cls.name
                )
                chunks.append(method_chunk)

        if structure.imports:
            module_chunk = self._create_module_imports_chunk(
                structure.file_path, structure, lines
            )
            if module_chunk is not None:
                chunks.append(module_chunk)

        handled_spans = {c.line_span for c in chunks}
        for _sym_key, sym in structure.symbols.items():
            if sym.kind in (
                SymbolKind.FUNCTION,
                SymbolKind.ASYNC_FUNCTION,
                SymbolKind.CLASS,
                SymbolKind.METHOD,
                SymbolKind.IMPORT,
            ):
                continue
            if sym.line_span in handled_spans:
                continue

            code_content = self._extract_source_lines(lines, sym.line_span)
            chunk_id = (
                f"{structure.file_path}:{sym.name}:{sym.line_span.start_line}:"
                f"{sym.line_span.end_line}"
            )
            embedding_text = (
                f"File: {structure.file_path.name}\n"
                f"Symbol: {sym.name} ({sym.kind.value})\n"
                f"Signature: {sym.signature}\n"
                f"Code:\n{code_content}"
            )
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=structure.file_path,
                    symbol_name=sym.name,
                    kind=sym.kind,
                    signature=sym.signature,
                    docstring=sym.docstring,
                    code_content=code_content,
                    line_span=sym.line_span,
                    embedding_text=embedding_text,
                    token_count=self._estimate_token_count(embedding_text),
                    metadata={"kind": sym.kind.value},
                )
            )
            handled_spans.add(sym.line_span)

        return chunks

    def chunk_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        extensions: list[str] | None = None,
    ) -> list[CodeChunk]:
        """Recursively parses and chunks all supported files in a directory."""
        dir_obj = Path(directory).resolve()
        if not dir_obj.exists() or not dir_obj.is_dir():
            return []

        ext_set = (
            {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in extensions
            }
            if extensions is not None
            else set(EXTENSION_LANGUAGE_MAP.keys())
        )

        all_chunks: list[CodeChunk] = []
        files = dir_obj.rglob("*") if recursive else dir_obj.glob("*")
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in ext_set:
                try:
                    file_chunks = self.chunk_file(file_path)
                    all_chunks.extend(file_chunks)
                except Exception:  # noqa: S110
                    pass
        return all_chunks

    def _extract_source_lines(self, lines: list[str], span: LineSpan) -> str:
        """Extracts lines bounded by line_span (1-indexed)."""
        start_idx = max(0, span.start_line - 1)
        end_idx = min(len(lines), span.end_line)
        return "\n".join(lines[start_idx:end_idx])

    def _estimate_token_count(self, text: str) -> int:
        """Estimates token count (~4 characters per token)."""
        return max(1, len(text) // 4)

    def _create_function_chunk(
        self,
        file_path: Path,
        fn: FunctionDefinition,
        lines: list[str],
        parent_class: str | None = None,
    ) -> CodeChunk:
        """Creates a CodeChunk for a function or method definition."""
        code_content = self._extract_source_lines(lines, fn.line_span)
        kind = (
            SymbolKind.METHOD
            if parent_class
            else (SymbolKind.ASYNC_FUNCTION if fn.is_async else SymbolKind.FUNCTION)
        )
        qualified_name = f"{parent_class}.{fn.name}" if parent_class else fn.name

        embedding_parts = [
            f"File: {file_path.name}",
            f"Symbol: {qualified_name} ({kind.value})",
            f"Signature: {fn.signature}",
        ]
        if fn.docstring:
            embedding_parts.append(f"Docstring: {fn.docstring}")
        if fn.decorators:
            embedding_parts.append(f"Decorators: {', '.join(fn.decorators)}")
        embedding_parts.append("Code:\n" + code_content)
        embedding_text = "\n".join(embedding_parts)

        chunk_id = (
            f"{file_path}:{qualified_name}:{fn.line_span.start_line}:"
            f"{fn.line_span.end_line}"
        )

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            symbol_name=qualified_name,
            kind=kind,
            signature=fn.signature,
            docstring=fn.docstring,
            code_content=code_content,
            line_span=fn.line_span,
            embedding_text=embedding_text,
            token_count=self._estimate_token_count(embedding_text),
            metadata={
                "is_async": fn.is_async,
                "decorators": fn.decorators,
                "parent_class": parent_class,
                "return_type": fn.return_type,
            },
        )

    def _create_class_chunk(
        self,
        file_path: Path,
        cls: ClassDefinition,
        lines: list[str],
    ) -> CodeChunk:
        """Creates a CodeChunk for a class overview and signature."""
        code_content = self._extract_source_lines(lines, cls.line_span)
        method_signatures = [m.signature for m in cls.methods]

        embedding_parts = [
            f"File: {file_path.name}",
            f"Symbol: {cls.name} (class)",
            f"Signature: {cls.signature}",
        ]
        if cls.docstring:
            embedding_parts.append(f"Docstring: {cls.docstring}")
        if cls.bases:
            embedding_parts.append(f"Inherits: {', '.join(cls.bases)}")
        if method_signatures:
            embedding_parts.append(
                f"Methods ({len(method_signatures)}): {', '.join(method_signatures)}"
            )
        embedding_parts.append("Code:\n" + code_content)
        embedding_text = "\n".join(embedding_parts)

        chunk_id = (
            f"{file_path}:{cls.name}:{cls.line_span.start_line}:"
            f"{cls.line_span.end_line}"
        )

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            symbol_name=cls.name,
            kind=SymbolKind.CLASS,
            signature=cls.signature,
            docstring=cls.docstring,
            code_content=code_content,
            line_span=cls.line_span,
            embedding_text=embedding_text,
            token_count=self._estimate_token_count(embedding_text),
            metadata={
                "bases": cls.bases,
                "decorators": cls.decorators,
                "method_count": len(cls.methods),
            },
        )

    def _create_module_imports_chunk(
        self,
        file_path: Path,
        structure: FileStructure,
        lines: list[str],
    ) -> CodeChunk | None:
        """Creates an overview CodeChunk for module-level imports."""
        if not structure.imports:
            return None

        start_line = min(i.line_span.start_line for i in structure.imports)
        end_line = max(i.line_span.end_line for i in structure.imports)
        span = LineSpan(
            start_line=start_line, end_line=end_line, start_col=0, end_col=0
        )
        code_content = self._extract_source_lines(lines, span)
        import_statements = [i.statement for i in structure.imports]

        embedding_text = (
            f"File: {file_path.name}\n"
            f"Module Imports ({len(import_statements)}):\n"
            + "\n".join(import_statements)
        )

        chunk_id = f"{file_path}:__imports__:{start_line}:{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            symbol_name=f"{file_path.stem}:__imports__",
            kind=SymbolKind.IMPORT,
            signature=f"imports ({len(import_statements)})",
            docstring=f"Module imports for {file_path.name}",
            code_content=code_content,
            line_span=span,
            embedding_text=embedding_text,
            token_count=self._estimate_token_count(embedding_text),
            metadata={"import_count": len(import_statements)},
        )


MultiLanguageChunker = ASTChunker
