from __future__ import annotations

import inspect
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

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

PY_LANGUAGE = Language(tspython.language())
_SHARED_PARSER = Parser(PY_LANGUAGE)


class PythonASTParser:
    """Extracts structured symbols, imports, and definitions from Python ASTs."""

    def __init__(self, parser: Parser | None = None) -> None:
        self.parser = parser or _SHARED_PARSER

    def parse(self, file_path: Path | str, source_code: str | bytes) -> FileStructure:
        """Parses Python source code into a structured FileStructure."""
        if isinstance(source_code, str):
            code_bytes = source_code.encode("utf-8")
        else:
            code_bytes = source_code

        file_path_obj = Path(file_path)
        tree = self.parser.parse(code_bytes)

        has_syntax_errors = tree.root_node.has_error

        file_structure = FileStructure(
            file_path=file_path_obj,
            has_syntax_errors=has_syntax_errors,
        )

        self._walk_root_node(tree.root_node, file_structure, code_bytes)
        return file_structure

    def _walk_root_node(
        self, root: Node, file_structure: FileStructure, code_bytes: bytes
    ) -> None:
        """Walks top-level module statements extracting definitions and imports."""
        for child in root.children:
            if child.type in ("import_statement", "import_from_statement"):
                imports = self._extract_imports(child, code_bytes)
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

            elif child.type == "function_definition":
                fn = self._extract_function(
                    child,
                    decorators=[],
                    parent_class=None,
                    code_bytes=code_bytes,
                )
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

            elif child.type == "class_definition":
                cls = self._extract_class(child, decorators=[], code_bytes=code_bytes)
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

            elif child.type == "decorated_definition":
                decorators = self._extract_decorators_from_wrapper(child, code_bytes)
                definition_node = child.child_by_field_name("definition")
                if not definition_node:
                    for sub in child.children:
                        if sub.type in (
                            "function_definition",
                            "class_definition",
                        ):
                            definition_node = sub
                            break

                if definition_node and definition_node.type == "function_definition":
                    fn = self._extract_function(
                        definition_node,
                        decorators=decorators,
                        parent_class=None,
                        code_bytes=code_bytes,
                    )
                    file_structure.functions.append(fn)
                    kind = (
                        SymbolKind.ASYNC_FUNCTION
                        if fn.is_async
                        else SymbolKind.FUNCTION
                    )
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

                elif definition_node and definition_node.type == "class_definition":
                    cls = self._extract_class(
                        definition_node,
                        decorators=decorators,
                        code_bytes=code_bytes,
                    )
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

    def _extract_line_span(self, node: Node) -> LineSpan:
        """Converts zero-indexed tree-sitter points to 1-indexed LineSpan."""
        return LineSpan(
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            start_col=node.start_point.column,
            end_col=node.end_point.column,
        )

    def _extract_docstring(
        self, body_node: Node | None, code_bytes: bytes
    ) -> str | None:
        """Extracts and normalizes docstring from a function/class body block."""
        if not body_node:
            return None

        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        raw_bytes = code_bytes[sub.start_byte : sub.end_byte]
                        raw_str = raw_bytes.decode("utf-8", errors="replace")
                        return self._clean_docstring(raw_str)
            elif child.type not in ("comment", ":"):
                break
        return None

    def _clean_docstring(self, raw_str: str) -> str:
        """Strips quotes and formats indentation using inspect.cleandoc."""
        text = raw_str.strip()
        for quote in ('"""', "'''", '"', "'"):
            if (
                text.startswith(quote)
                and text.endswith(quote)
                and len(text) >= 2 * len(quote)
            ):
                text = text[len(quote) : -len(quote)]
                break
        return inspect.cleandoc(text)

    def _extract_decorators_from_wrapper(
        self, decorated_node: Node, code_bytes: bytes
    ) -> list[str]:
        """Extracts decorator strings from a decorated_definition node."""
        decorators: list[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                dec_bytes = code_bytes[child.start_byte : child.end_byte]
                dec_text = dec_bytes.decode("utf-8", errors="replace").strip()
                decorators.append(dec_text)
        return decorators

    def _extract_parameters(
        self, params_node: Node | None, code_bytes: bytes
    ) -> list[ArgumentDefinition]:
        """Parses parameter list with annotations and default values."""
        args: list[ArgumentDefinition] = []
        if not params_node:
            return args

        for child in params_node.children:
            if child.type in ("(", ")", ","):
                continue

            if child.type == "identifier":
                name = code_bytes[child.start_byte : child.end_byte].decode("utf-8")
                args.append(ArgumentDefinition(name=name))

            elif child.type == "typed_parameter":
                name_node = child.child_by_field_name("name") or child.children[0]
                type_node = child.child_by_field_name("type") or (
                    child.children[2] if len(child.children) > 2 else None
                )
                name = code_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8"
                )
                type_ann = (
                    code_bytes[type_node.start_byte : type_node.end_byte].decode(
                        "utf-8"
                    )
                    if type_node
                    else None
                )
                args.append(ArgumentDefinition(name=name, type_annotation=type_ann))

            elif child.type == "default_parameter":
                name_node = child.child_by_field_name("name") or child.children[0]
                val_node = child.child_by_field_name("value") or child.children[-1]
                name = code_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8"
                )
                default_val = code_bytes[
                    val_node.start_byte : val_node.end_byte
                ].decode("utf-8")
                args.append(ArgumentDefinition(name=name, default_value=default_val))

            elif child.type == "typed_default_parameter":
                name_node = child.child_by_field_name("name") or child.children[0]
                type_node = child.child_by_field_name("type")
                val_node = child.child_by_field_name("value") or child.children[-1]
                name = code_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8"
                )
                type_ann = (
                    code_bytes[type_node.start_byte : type_node.end_byte].decode(
                        "utf-8"
                    )
                    if type_node
                    else None
                )
                default_val = code_bytes[
                    val_node.start_byte : val_node.end_byte
                ].decode("utf-8")
                args.append(
                    ArgumentDefinition(
                        name=name,
                        type_annotation=type_ann,
                        default_value=default_val,
                    )
                )

            elif child.type in (
                "list_splat_pattern",
                "dictionary_splat_pattern",
                "positional_separator",
                "keyword_separator",
            ):
                text = code_bytes[child.start_byte : child.end_byte].decode("utf-8")
                args.append(ArgumentDefinition(name=text))

        return args

    def _extract_function(
        self,
        node: Node,
        decorators: list[str],
        parent_class: str | None,
        code_bytes: bytes,
    ) -> FunctionDefinition:
        """Parses a function_definition node."""
        name_node = node.child_by_field_name("name")
        name = (
            code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node
            else "anonymous"
        )

        is_async = any(c.type == "async" for c in node.children)
        params_node = node.child_by_field_name("parameters")
        args = self._extract_parameters(params_node, code_bytes)

        return_type_node = node.child_by_field_name("return_type")
        return_type = (
            code_bytes[return_type_node.start_byte : return_type_node.end_byte].decode(
                "utf-8"
            )
            if return_type_node
            else None
        )

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, code_bytes)
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

    def _extract_class(
        self,
        node: Node,
        decorators: list[str],
        code_bytes: bytes,
    ) -> ClassDefinition:
        """Parses a class_definition node and its member methods."""
        name_node = node.child_by_field_name("name")
        name = (
            code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if name_node
            else "anonymous"
        )

        bases: list[str] = []
        superclasses_node = node.child_by_field_name("superclasses")
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type not in ("(", ")", ","):
                    base_name = code_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    bases.append(base_name)

        body_node = node.child_by_field_name("body")
        docstring = self._extract_docstring(body_node, code_bytes)
        line_span = self._extract_line_span(node)

        methods: list[FunctionDefinition] = []
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    method = self._extract_function(
                        child,
                        decorators=[],
                        parent_class=name,
                        code_bytes=code_bytes,
                    )
                    methods.append(method)
                elif child.type == "decorated_definition":
                    method_decorators = self._extract_decorators_from_wrapper(
                        child, code_bytes
                    )
                    def_node = child.child_by_field_name("definition")
                    if not def_node:
                        for sub in child.children:
                            if sub.type == "function_definition":
                                def_node = sub
                                break
                    if def_node and def_node.type == "function_definition":
                        method = self._extract_function(
                            def_node,
                            decorators=method_decorators,
                            parent_class=name,
                            code_bytes=code_bytes,
                        )
                        methods.append(method)

        return ClassDefinition(
            name=name,
            bases=bases,
            methods=methods,
            line_span=line_span,
            docstring=docstring,
            decorators=decorators,
        )

    def _extract_imports(self, node: Node, code_bytes: bytes) -> list[ImportDefinition]:
        """Parses import_statement and import_from_statement nodes."""
        line_span = self._extract_line_span(node)
        imports: list[ImportDefinition] = []

        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    mod_name = code_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    imports.append(
                        ImportDefinition(
                            module=mod_name,
                            names=[],
                            alias=None,
                            is_from_import=False,
                            line_span=line_span,
                        )
                    )
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    alias_node = child.child_by_field_name("alias")
                    mod_name = (
                        code_bytes[name_node.start_byte : name_node.end_byte].decode(
                            "utf-8"
                        )
                        if name_node
                        else ""
                    )
                    alias_name = (
                        code_bytes[alias_node.start_byte : alias_node.end_byte].decode(
                            "utf-8"
                        )
                        if alias_node
                        else None
                    )
                    imports.append(
                        ImportDefinition(
                            module=mod_name,
                            names=[],
                            alias=alias_name,
                            is_from_import=False,
                            line_span=line_span,
                        )
                    )

        elif node.type == "import_from_statement":
            module_name_node = node.child_by_field_name("module_name")
            mod_name = (
                code_bytes[
                    module_name_node.start_byte : module_name_node.end_byte
                ].decode("utf-8")
                if module_name_node
                else ""
            )

            names: list[str] = []
            for child in node.children:
                if child.type == "dotted_name" and child != module_name_node:
                    names.append(
                        code_bytes[child.start_byte : child.end_byte].decode("utf-8")
                    )
                elif child.type == "aliased_import":
                    aliased_str = code_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    names.append(aliased_str)

            imports.append(
                ImportDefinition(
                    module=mod_name,
                    names=names,
                    alias=None,
                    is_from_import=True,
                    line_span=line_span,
                )
            )

        return imports
