from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from src.indexer.models import (
    ArgumentDefinition,
    ClassDefinition,
    FileStructure,
    FunctionDefinition,
    ImportDefinition,
    LineSpan,
    Symbol,
    SymbolKind,
)


class PythonASTParser:
    """Extracts structured symbols, imports, and definitions from Python ASTs."""

    def __init__(self, parser: Any | None = None) -> None:
        self.parser = parser

    def parse(self, file_path: Path | str, source_code: str | bytes) -> FileStructure:
        """Parses Python source code into a structured FileStructure."""
        if isinstance(source_code, bytes):
            code_str = source_code.decode("utf-8", errors="replace")
        else:
            code_str = source_code

        file_path_obj = Path(file_path)

        try:
            tree = ast.parse(code_str, filename=str(file_path_obj))
            has_syntax_errors = False
        except SyntaxError:
            has_syntax_errors = True
            tree = self._recover_partial_tree(code_str, str(file_path_obj))

        file_structure = FileStructure(
            file_path=file_path_obj,
            has_syntax_errors=has_syntax_errors,
        )

        self._walk_root_body(tree.body, file_structure)
        return file_structure

    def _recover_partial_tree(self, code_str: str, filename: str) -> ast.Module:
        """Recovers valid top-level AST nodes from a module with syntax errors."""
        lines = code_str.splitlines(keepends=True)
        valid_nodes: list[ast.stmt] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith(
                ("def ", "async def ", "class ", "import ", "from ", "@")
            ) and not stripped.startswith("#"):
                best_node: ast.stmt | None = None
                best_j = i
                for j in range(i + 1, min(len(lines) + 1, i + 200)):
                    block = "".join(lines[i:j])
                    try:
                        parsed = ast.parse(block, filename=filename)
                        if parsed.body:
                            best_node = parsed.body[0]
                            ast.increment_lineno(best_node, i)
                            best_j = j
                    except SyntaxError:
                        continue
                if best_node:
                    valid_nodes.append(best_node)
                    i = best_j
                    continue
            i += 1

        return ast.Module(body=valid_nodes, type_ignores=[])

    def _walk_root_body(
        self, body: list[ast.stmt], file_structure: FileStructure
    ) -> None:
        """Walks top-level module statements extracting definitions and imports."""
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = self._extract_imports(node)
                file_structure.imports.extend(imports)
                for imp in imports:
                    mod_name = imp.module
                    first_name = imp.names[0] if imp.names else ""
                    sym_name = f"{mod_name}.{first_name}" if first_name else mod_name
                    sym = Symbol(
                        name=sym_name,
                        kind=SymbolKind.IMPORT,
                        file_path=file_structure.file_path,
                        line_span=imp.line_span,
                        docstring=None,
                        signature=imp.statement,
                        definition=imp,
                    )
                    file_structure.symbols[f"import:{sym.name}"] = sym

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = self._extract_function(node, parent_class=None)
                file_structure.functions.append(fn)
                kind = SymbolKind.ASYNC_FUNCTION if fn.is_async else SymbolKind.FUNCTION
                sym = Symbol(
                    name=fn.name,
                    kind=kind,
                    file_path=file_structure.file_path,
                    line_span=fn.line_span,
                    docstring=fn.docstring,
                    signature=fn.signature,
                    definition=fn,
                )
                file_structure.symbols[fn.name] = sym

            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node)
                file_structure.classes.append(cls)
                sym = Symbol(
                    name=cls.name,
                    kind=SymbolKind.CLASS,
                    file_path=file_structure.file_path,
                    line_span=cls.line_span,
                    docstring=cls.docstring,
                    signature=cls.signature,
                    definition=cls,
                )
                file_structure.symbols[cls.name] = sym
                for method in cls.methods:
                    method_sym = Symbol(
                        name=f"{cls.name}.{method.name}",
                        kind=SymbolKind.METHOD,
                        file_path=file_structure.file_path,
                        line_span=method.line_span,
                        docstring=method.docstring,
                        signature=method.signature,
                        definition=method,
                    )
                    file_structure.symbols[f"{cls.name}.{method.name}"] = method_sym

    def _extract_imports(
        self, node: ast.Import | ast.ImportFrom
    ) -> list[ImportDefinition]:
        """Extracts ImportDefinition list from ast.Import or ast.ImportFrom."""
        line_span = self._extract_line_span(node)
        imports: list[ImportDefinition] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportDefinition(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        is_from_import=False,
                        line_span=line_span,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module_name = "." * node.level + (node.module or "")
            names: list[str] = []
            for alias in node.names:
                if alias.asname:
                    names.append(f"{alias.name} as {alias.asname}")
                else:
                    names.append(alias.name)

            imports.append(
                ImportDefinition(
                    module=module_name,
                    names=names,
                    alias=None,
                    is_from_import=True,
                    line_span=line_span,
                )
            )

        return imports

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_class: str | None = None,
    ) -> FunctionDefinition:
        """Extracts FunctionDefinition from ast.FunctionDef or ast.AsyncFunctionDef."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        name = node.name
        docstring = ast.get_docstring(node)
        return_type = ast.unparse(node.returns) if node.returns else None
        decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        args = self._extract_arguments(node.args)
        line_span = self._extract_line_span(node)

        return FunctionDefinition(
            name=name,
            args=args,
            return_type=return_type,
            line_span=line_span,
            docstring=docstring,
            is_async=is_async,
            decorators=decorators,
            parent_class=parent_class,
        )

    def _extract_arguments(self, args_node: ast.arguments) -> list[ArgumentDefinition]:
        """Extracts ArgumentDefinition list from ast.arguments."""
        arguments: list[ArgumentDefinition] = []

        # Positional args & defaults
        pos_args = args_node.posonlyargs + args_node.args
        num_pos = len(pos_args)
        num_defaults = len(args_node.defaults)
        default_offset = num_pos - num_defaults

        for idx, arg in enumerate(pos_args):
            default_val = None
            if idx >= default_offset:
                def_idx = idx - default_offset
                default_val = ast.unparse(args_node.defaults[def_idx])

            type_ann = ast.unparse(arg.annotation) if arg.annotation else None
            arguments.append(
                ArgumentDefinition(
                    name=arg.arg,
                    type_annotation=type_ann,
                    default_value=default_val,
                )
            )

        # *vararg
        if args_node.vararg:
            arg = args_node.vararg
            type_ann = ast.unparse(arg.annotation) if arg.annotation else None
            arguments.append(
                ArgumentDefinition(
                    name=f"*{arg.arg}",
                    type_annotation=type_ann,
                    default_value=None,
                )
            )

        # Keyword-only args
        for idx, arg in enumerate(args_node.kwonlyargs):
            kw_def = (
                args_node.kw_defaults[idx] if idx < len(args_node.kw_defaults) else None
            )
            default_val = ast.unparse(kw_def) if kw_def is not None else None

            type_ann = ast.unparse(arg.annotation) if arg.annotation else None
            arguments.append(
                ArgumentDefinition(
                    name=arg.arg,
                    type_annotation=type_ann,
                    default_value=default_val,
                )
            )

        # **kwarg
        if args_node.kwarg:
            arg = args_node.kwarg
            type_ann = ast.unparse(arg.annotation) if arg.annotation else None
            arguments.append(
                ArgumentDefinition(
                    name=f"**{arg.arg}",
                    type_annotation=type_ann,
                    default_value=None,
                )
            )

        return arguments

    def _extract_class(self, node: ast.ClassDef) -> ClassDefinition:
        """Extracts ClassDefinition from ast.ClassDef."""
        name = node.name
        docstring = ast.get_docstring(node)
        bases = [ast.unparse(b) for b in node.bases]
        decorators = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        line_span = self._extract_line_span(node)

        methods: list[FunctionDefinition] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item, parent_class=name))

        return ClassDefinition(
            name=name,
            bases=bases,
            methods=methods,
            line_span=line_span,
            docstring=docstring,
            decorators=decorators,
        )

    def _extract_line_span(self, node: ast.AST) -> LineSpan:
        """Converts ast.AST line/col coordinates to 1-indexed LineSpan."""
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line) or start_line
        start_col = getattr(node, "col_offset", 0)
        end_col = getattr(node, "end_col_offset", start_col) or start_col

        return LineSpan(
            start_line=start_line,
            end_line=end_line,
            start_col=start_col,
            end_col=end_col,
        )
