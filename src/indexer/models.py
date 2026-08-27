from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SymbolKind(StrEnum):
    """Categorization of extracted code symbols."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    IMPORT = "import"
    VARIABLE = "variable"
    TAG = "tag"
    RULE = "rule"
    PROPERTY = "property"


@dataclass(frozen=True)
class LineSpan:
    """Zero-indexed line and column span for a code element."""

    start_line: int
    end_line: int
    start_col: int
    end_col: int

    def to_dict(self) -> dict[str, int]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_col": self.start_col,
            "end_col": self.end_col,
        }


@dataclass
class ArgumentDefinition:
    """Function or method parameter definition."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None

    def to_signature_part(self) -> str:
        part = self.name
        if self.type_annotation:
            part += f": {self.type_annotation}"
        if self.default_value is not None:
            part += f" = {self.default_value}"
        return part

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "default_value": self.default_value,
        }


@dataclass
class FunctionDefinition:
    """Extracted function or method metadata."""

    name: str
    args: list[ArgumentDefinition]
    return_type: str | None
    line_span: LineSpan
    docstring: str | None = None
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None

    @property
    def signature(self) -> str:
        prefix = "async def " if self.is_async else "def "
        args_str = ", ".join(arg.to_signature_part() for arg in self.args)
        ret_str = f" -> {self.return_type}" if self.return_type else ""
        return f"{prefix}{self.name}({args_str}){ret_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "args": [arg.to_dict() for arg in self.args],
            "return_type": self.return_type,
            "line_span": self.line_span.to_dict(),
            "docstring": self.docstring,
            "is_async": self.is_async,
            "decorators": self.decorators,
            "parent_class": self.parent_class,
        }


@dataclass
class ClassDefinition:
    """Extracted class definition metadata."""

    name: str
    bases: list[str]
    methods: list[FunctionDefinition]
    line_span: LineSpan
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)

    @property
    def signature(self) -> str:
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        return f"class {self.name}{bases_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "line_span": self.line_span.to_dict(),
            "docstring": self.docstring,
            "decorators": self.decorators,
        }


@dataclass
class ImportDefinition:
    """Extracted module or symbol import."""

    module: str
    names: list[str]
    alias: str | None
    is_from_import: bool
    line_span: LineSpan

    @property
    def statement(self) -> str:
        if self.is_from_import:
            names_str = ", ".join(self.names)
            return f"from {self.module} import {names_str}"
        if self.alias:
            return f"import {self.module} as {self.alias}"
        return f"import {self.module}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "is_from_import": self.is_from_import,
            "statement": self.statement,
            "line_span": self.line_span.to_dict(),
        }


@dataclass
class Symbol:
    """Canonical extracted symbol record."""

    name: str
    kind: SymbolKind
    file_path: Path
    line_span: LineSpan
    docstring: str | None
    signature: str
    definition: FunctionDefinition | ClassDefinition | ImportDefinition | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "file_path": str(self.file_path),
            "line_span": self.line_span.to_dict(),
            "docstring": self.docstring,
            "signature": self.signature,
            "definition": (
                self.definition.to_dict()
                if self.definition and hasattr(self.definition, "to_dict")
                else None
            ),
        }


@dataclass
class FileStructure:
    """Module-level overview of an indexed Python source file."""

    file_path: Path
    imports: list[ImportDefinition] = field(default_factory=list)
    classes: list[ClassDefinition] = field(default_factory=list)
    functions: list[FunctionDefinition] = field(default_factory=list)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    has_syntax_errors: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "imports": [imp.to_dict() for imp in self.imports],
            "classes": [cls.to_dict() for cls in self.classes],
            "functions": [fn.to_dict() for fn in self.functions],
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "has_syntax_errors": self.has_syntax_errors,
        }
