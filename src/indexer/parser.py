from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import tree_sitter_css as tscss
import tree_sitter_html as tshtml
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspy
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser, Tree

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


class SupportedLanguage(StrEnum):
    """Supported programming and markup languages for Tree-sitter parsing."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JAVA = "java"
    HTML = "html"
    CSS = "css"
    UNKNOWN = "unknown"


EXTENSION_LANGUAGE_MAP: dict[str, SupportedLanguage] = {
    ".py": SupportedLanguage.PYTHON,
    ".pyi": SupportedLanguage.PYTHON,
    ".js": SupportedLanguage.JAVASCRIPT,
    ".mjs": SupportedLanguage.JAVASCRIPT,
    ".cjs": SupportedLanguage.JAVASCRIPT,
    ".jsx": SupportedLanguage.JAVASCRIPT,
    ".ts": SupportedLanguage.TYPESCRIPT,
    ".tsx": SupportedLanguage.TSX,
    ".java": SupportedLanguage.JAVA,
    ".html": SupportedLanguage.HTML,
    ".htm": SupportedLanguage.HTML,
    ".css": SupportedLanguage.CSS,
    ".scss": SupportedLanguage.CSS,
    ".less": SupportedLanguage.CSS,
}


@dataclass
class LanguageConfig:
    """Configuration bundle for a Tree-sitter language parser."""

    language: SupportedLanguage
    tree_sitter_language: Language
    parser: Parser


def get_node_text(node: Node | None) -> str:
    """Safely extracts UTF-8 string from a Tree-sitter Node."""
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def clean_docstring(raw: str | None) -> str | None:
    """Cleans triple/single quotes, JSDoc comment blocks, and trims indentation."""
    if not raw:
        return None

    text = raw.strip()
    py_match = re.match(r"^[rRuUbBfF]*(\"\"\"|\'\'\'|\"|\')(.*)\1$", text, re.DOTALL)
    if py_match:
        content = py_match.group(2)
        lines = content.split("\n")
        if len(lines) > 1:
            indents = [
                len(line_str) - len(line_str.lstrip())
                for line_str in lines[1:]
                if line_str.strip()
            ]
            min_indent = min(indents) if indents else 0
            cleaned = [lines[0].strip()] + [
                line_str[min_indent:] for line_str in lines[1:]
            ]
            return "\n".join(cleaned).strip("\n")
        return content.strip("\n")

    if text.startswith("/**") and text.endswith("*/"):
        inner = text[3:-2].strip()
        lines = [re.sub(r"^\s*\*\s?", "", line) for line in inner.splitlines()]
        return "\n".join(lines).strip()

    if text.startswith("/*") and text.endswith("*/"):
        inner = text[2:-2].strip()
        return inner

    if text.startswith(("//", "#")):
        return text.lstrip("/# ").strip()

    return text


def node_to_linespan(node: Node) -> LineSpan:
    """Converts Tree-sitter Node start_point/end_point to 1-indexed LineSpan."""
    sp = node.start_point
    ep = node.end_point
    return LineSpan(
        start_line=int(sp[0]) + 1,
        end_line=int(ep[0]) + 1,
        start_col=int(sp[1]),
        end_col=int(ep[1]),
    )


class TreeSitterParser:
    """Unified multi-language AST parser powered by Tree-sitter."""

    def __init__(self, default_language: SupportedLanguage | str | None = None) -> None:
        if isinstance(default_language, str):
            try:
                self.default_language: SupportedLanguage | None = SupportedLanguage(
                    default_language.lower()
                )
            except ValueError:
                self.default_language = None
        else:
            self.default_language = default_language

        self._parsers: dict[SupportedLanguage, LanguageConfig] = {}

    def _get_language_config(self, lang: SupportedLanguage) -> LanguageConfig:
        """Lazily creates and caches a parser instance per language."""
        if lang in self._parsers:
            return self._parsers[lang]

        match lang:
            case SupportedLanguage.PYTHON:
                raw_lang = tspy.language()
            case SupportedLanguage.JAVASCRIPT:
                raw_lang = tsjs.language()
            case SupportedLanguage.TYPESCRIPT:
                raw_lang = tsts.language_typescript()
            case SupportedLanguage.TSX:
                raw_lang = tsts.language_tsx()
            case SupportedLanguage.JAVA:
                raw_lang = tsjava.language()
            case SupportedLanguage.HTML:
                raw_lang = tshtml.language()
            case SupportedLanguage.CSS:
                raw_lang = tscss.language()
            case _:
                raw_lang = tspy.language()

        ts_lang = Language(raw_lang)
        parser = Parser(ts_lang)
        config = LanguageConfig(
            language=lang,
            tree_sitter_language=ts_lang,
            parser=parser,
        )
        self._parsers[lang] = config
        return config

    def detect_language(self, file_path: Path | str) -> SupportedLanguage:
        """Determines the language from file path extension or returns default."""
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()
        if suffix in EXTENSION_LANGUAGE_MAP:
            return EXTENSION_LANGUAGE_MAP[suffix]
        return self.default_language or SupportedLanguage.PYTHON

    def parse(
        self,
        file_path: Path | str,
        source_code: str | bytes,
        language: SupportedLanguage | str | None = None,
    ) -> FileStructure:
        """Parses source code into a structured FileStructure with extracted symbols."""
        file_path_obj = Path(file_path)
        if isinstance(source_code, str):
            code_bytes = source_code.encode("utf-8")
        else:
            code_bytes = source_code

        target_lang = (
            SupportedLanguage(language.lower())
            if isinstance(language, str)
            else (language or self.detect_language(file_path_obj))
        )

        config = self._get_language_config(target_lang)
        tree: Tree = config.parser.parse(code_bytes)
        has_syntax_errors = tree.root_node.has_error

        file_structure = FileStructure(
            file_path=file_path_obj,
            has_syntax_errors=has_syntax_errors,
        )

        match target_lang:
            case SupportedLanguage.PYTHON:
                self._extract_python(tree.root_node, code_bytes, file_structure)
            case (
                SupportedLanguage.JAVASCRIPT
                | SupportedLanguage.TYPESCRIPT
                | SupportedLanguage.TSX
            ):
                self._extract_javascript_typescript(
                    tree.root_node, code_bytes, file_structure
                )
            case SupportedLanguage.JAVA:
                self._extract_java(tree.root_node, code_bytes, file_structure)
            case SupportedLanguage.HTML:
                self._extract_html(tree.root_node, code_bytes, file_structure)
            case SupportedLanguage.CSS:
                self._extract_css(tree.root_node, code_bytes, file_structure)
            case _:
                self._extract_generic(tree.root_node, code_bytes, file_structure)

        return file_structure

    def _extract_python(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Extracts Python functions, classes, imports, and symbols."""

        def walk_node(node: Node, current_decorators: list[str] | None = None) -> None:
            decorators = current_decorators or []

            if node.type == "decorated_definition":
                decs: list[str] = []
                target_def: Node | None = None
                for child in node.children:
                    if child.type == "decorator":
                        decs.append(get_node_text(child).strip())
                    elif child.type in ("function_definition", "class_definition"):
                        target_def = child
                if target_def:
                    walk_node(target_def, current_decorators=decs)
                return

            if node.type == "import_statement":
                line_span = node_to_linespan(node)
                for child in node.children:
                    if child.type == "dotted_name":
                        mod_name = get_node_text(child)
                        imp = ImportDefinition(
                            module=mod_name,
                            names=[],
                            alias=None,
                            is_from_import=False,
                            line_span=line_span,
                        )
                        file_structure.imports.append(imp)
                        sym = Symbol(
                            name=mod_name,
                            kind=SymbolKind.IMPORT,
                            file_path=file_structure.file_path,
                            line_span=line_span,
                            docstring=None,
                            signature=imp.statement,
                            definition=imp,
                        )
                        file_structure.symbols[f"import:{mod_name}"] = sym
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        mod_name = get_node_text(name_node)
                        alias_name = get_node_text(alias_node) if alias_node else None
                        imp = ImportDefinition(
                            module=mod_name,
                            names=[],
                            alias=alias_name,
                            is_from_import=False,
                            line_span=line_span,
                        )
                        file_structure.imports.append(imp)
                        sym_name = alias_name or mod_name
                        sym = Symbol(
                            name=sym_name,
                            kind=SymbolKind.IMPORT,
                            file_path=file_structure.file_path,
                            line_span=line_span,
                            docstring=None,
                            signature=imp.statement,
                            definition=imp,
                        )
                        file_structure.symbols[f"import:{sym_name}"] = sym

            elif node.type == "import_from_statement":
                line_span = node_to_linespan(node)
                mod_name = ""
                names: list[str] = []

                module_node = node.child_by_field_name("module_name")
                if module_node:
                    mod_name = get_node_text(module_node)
                else:
                    for child in node.children:
                        if child.type in ("relative_import", "dotted_name"):
                            mod_name = get_node_text(child)
                            break

                for child in node.children:
                    if child.type == "dotted_name" and child != module_node:
                        if child.start_byte > node.start_byte + 4:
                            names.append(get_node_text(child))
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node and alias_node:
                            target_name = get_node_text(name_node)
                            target_alias = get_node_text(alias_node)
                            names.append(f"{target_name} as {target_alias}")
                        else:
                            names.append(get_node_text(child))

                imp = ImportDefinition(
                    module=mod_name,
                    names=names,
                    alias=None,
                    is_from_import=True,
                    line_span=line_span,
                )
                file_structure.imports.append(imp)
                first_name = names[0].split(" as ")[0] if names else ""
                sym_name = f"{mod_name}.{first_name}" if first_name else mod_name
                sym = Symbol(
                    name=sym_name,
                    kind=SymbolKind.IMPORT,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=None,
                    signature=imp.statement,
                    definition=imp,
                )
                file_structure.symbols[f"import:{sym_name}"] = sym

            elif node.type == "function_definition":
                fn = self._extract_python_function(
                    node, code_bytes, decorators=decorators, parent_class=None
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

            elif node.type == "class_definition":
                cls = self._extract_python_class(
                    node, code_bytes, decorators=decorators
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

            else:
                for child in node.children:
                    if child.is_named:
                        walk_node(child)

        for child in root_node.children:
            walk_node(child)

    def _extract_python_function(
        self,
        node: Node,
        code_bytes: bytes,
        decorators: list[str],
        parent_class: str | None = None,
    ) -> FunctionDefinition:
        """Extracts FunctionDefinition from a Python function_definition node."""
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) if name_node else "anonymous"

        is_async = False
        for child in node.children:
            if child.type == "async":
                is_async = True
                break

        return_type_node = node.child_by_field_name("return_type")
        return_type = get_node_text(return_type_node) if return_type_node else None

        args = self._extract_python_arguments(node.child_by_field_name("parameters"))
        line_span = node_to_linespan(node)
        docstring = self._extract_python_docstring(node.child_by_field_name("body"))

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

    def _extract_python_arguments(
        self, params_node: Node | None
    ) -> list[ArgumentDefinition]:
        """Extracts typed arguments and default values from Python parameters node."""
        if not params_node:
            return []

        args: list[ArgumentDefinition] = []
        for child in params_node.children:
            if child.type in ("(", ")", ",", "/", "*"):
                continue

            if child.type == "identifier":
                args.append(
                    ArgumentDefinition(
                        name=get_node_text(child),
                        type_annotation=None,
                        default_value=None,
                    )
                )

            elif child.type == "typed_parameter":
                name_node = child.child_by_field_name("name") or (
                    child.children[0] if child.children else None
                )
                type_node = child.child_by_field_name("type")
                name_str = get_node_text(name_node)
                type_str = get_node_text(type_node) if type_node else None
                args.append(
                    ArgumentDefinition(
                        name=name_str,
                        type_annotation=type_str,
                        default_value=None,
                    )
                )

            elif child.type == "default_parameter":
                name_node = child.child_by_field_name("name")
                val_node = child.child_by_field_name("value")
                name_str = get_node_text(name_node)
                val_str = get_node_text(val_node) if val_node else None
                args.append(
                    ArgumentDefinition(
                        name=name_str,
                        type_annotation=None,
                        default_value=val_str,
                    )
                )

            elif child.type == "typed_default_parameter":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                val_node = child.child_by_field_name("value")
                name_str = get_node_text(name_node)
                type_str = get_node_text(type_node) if type_node else None
                val_str = get_node_text(val_node) if val_node else None
                args.append(
                    ArgumentDefinition(
                        name=name_str,
                        type_annotation=type_str,
                        default_value=val_str,
                    )
                )

            elif child.type in (
                "list_splat_pattern",
                "dictionary_splat_pattern",
            ):
                name_str = get_node_text(child)
                args.append(
                    ArgumentDefinition(
                        name=name_str,
                        type_annotation=None,
                        default_value=None,
                    )
                )

        return args

    def _extract_python_class(
        self,
        node: Node,
        code_bytes: bytes,
        decorators: list[str],
    ) -> ClassDefinition:
        """Extracts ClassDefinition, base classes, and methods."""
        name_node = node.child_by_field_name("name")
        name = get_node_text(name_node) if name_node else "AnonymousClass"

        bases: list[str] = []
        superclasses_node = node.child_by_field_name(
            "superclasses"
        ) or node.child_by_field_name("argument_list")
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type not in ("(", ")", ","):
                    bases.append(get_node_text(child))

        body_node = node.child_by_field_name("body")
        docstring = self._extract_python_docstring(body_node)
        line_span = node_to_linespan(node)

        methods: list[FunctionDefinition] = []
        if body_node:
            current_decs: list[str] = []
            for child in body_node.children:
                if child.type == "decorated_definition":
                    decs: list[str] = []
                    target_def: Node | None = None
                    for sub in child.children:
                        if sub.type == "decorator":
                            decs.append(get_node_text(sub).strip())
                        elif sub.type == "function_definition":
                            target_def = sub
                    if target_def:
                        methods.append(
                            self._extract_python_function(
                                target_def,
                                code_bytes,
                                decorators=decs,
                                parent_class=name,
                            )
                        )
                elif child.type == "function_definition":
                    methods.append(
                        self._extract_python_function(
                            child,
                            code_bytes,
                            decorators=current_decs,
                            parent_class=name,
                        )
                    )
                    current_decs = []

        return ClassDefinition(
            name=name,
            bases=bases,
            methods=methods,
            line_span=line_span,
            docstring=docstring,
            decorators=decorators,
        )

    def _extract_python_docstring(self, body_node: Node | None) -> str | None:
        """Extracts docstring from the first statement of a Python body/block node."""
        if not body_node:
            return None
        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        return clean_docstring(get_node_text(sub))
            elif child.type in ("comment", "\n"):
                continue
            else:
                break
        return None

    def _extract_javascript_typescript(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Extracts JS/TS functions, classes, interfaces, imports, and exports."""
        pending_comment: list[str | None] = [None]

        def walk_node(node: Node) -> None:
            if node.type == "comment":
                pending_comment[0] = clean_docstring(get_node_text(node))
                return

            doc_comment = pending_comment[0]

            if node.type == "import_statement":
                line_span = node_to_linespan(node)
                src_node = node.child_by_field_name("source")
                mod_name = get_node_text(src_node).strip("'\"") if src_node else ""
                names: list[str] = []
                alias: str | None = None

                for child in node.children:
                    if child.type == "import_clause":
                        for sub in child.children:
                            if sub.type == "identifier":
                                alias = get_node_text(sub)
                            elif sub.type == "named_imports":
                                for spec in sub.children:
                                    if spec.type == "import_specifier":
                                        names.append(get_node_text(spec))
                            elif sub.type == "namespace_import":
                                for n in sub.children:
                                    if n.type == "identifier":
                                        alias = get_node_text(n)

                imp = ImportDefinition(
                    module=mod_name,
                    names=names,
                    alias=alias,
                    is_from_import=bool(names),
                    line_span=line_span,
                )
                file_structure.imports.append(imp)
                sym_name = alias or (names[0] if names else mod_name)
                sym = Symbol(
                    name=sym_name,
                    kind=SymbolKind.IMPORT,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=None,
                    signature=imp.statement,
                    definition=imp,
                )
                file_structure.symbols[f"import:{sym_name}"] = sym

            elif node.type in (
                "function_declaration",
                "generator_function_declaration",
            ):
                name_node = node.child_by_field_name("name")
                name = get_node_text(name_node) if name_node else "anonymous"
                is_async = any(c.type == "async" for c in node.children)
                ret_node = node.child_by_field_name("return_type")
                ret_type = get_node_text(ret_node).lstrip(": ") if ret_node else None
                args = self._extract_js_parameters(
                    node.child_by_field_name("parameters")
                )
                line_span = node_to_linespan(node)

                fn = FunctionDefinition(
                    name=name,
                    args=args,
                    return_type=ret_type,
                    line_span=line_span,
                    docstring=doc_comment,
                    is_async=is_async,
                    decorators=[],
                    parent_class=None,
                )
                file_structure.functions.append(fn)
                kind = SymbolKind.ASYNC_FUNCTION if is_async else SymbolKind.FUNCTION
                sym = Symbol(
                    name=name,
                    kind=kind,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=doc_comment,
                    signature=fn.signature,
                    definition=fn,
                )
                file_structure.symbols[name] = sym
                pending_comment[0] = None

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                name = get_node_text(name_node) if name_node else "AnonymousClass"
                bases: list[str] = []
                for child in node.children:
                    if child.type == "class_heritage":
                        for sub in child.children:
                            if sub.type in (
                                "identifier",
                                "member_expression",
                                "type_identifier",
                            ):
                                bases.append(get_node_text(sub))
                            elif sub.type == "extends_clause":
                                for val in sub.children:
                                    if val.type in (
                                        "identifier",
                                        "member_expression",
                                        "type_identifier",
                                    ):
                                        bases.append(get_node_text(val))

                body_node = node.child_by_field_name("body")
                line_span = node_to_linespan(node)
                methods: list[FunctionDefinition] = []

                if body_node:
                    m_comment: str | None = None
                    for child in body_node.children:
                        if child.type == "comment":
                            m_comment = clean_docstring(get_node_text(child))
                        elif child.type == "method_definition":
                            m_name_node = child.child_by_field_name("name")
                            m_name = (
                                get_node_text(m_name_node)
                                if m_name_node
                                else "anonymous"
                            )
                            m_async = any(c.type == "async" for c in child.children)
                            m_ret_node = child.child_by_field_name("return_type")
                            m_ret = (
                                get_node_text(m_ret_node).lstrip(": ")
                                if m_ret_node
                                else None
                            )
                            m_args = self._extract_js_parameters(
                                child.child_by_field_name("parameters")
                            )
                            m_span = node_to_linespan(child)
                            m_fn = FunctionDefinition(
                                name=m_name,
                                args=m_args,
                                return_type=m_ret,
                                line_span=m_span,
                                docstring=m_comment,
                                is_async=m_async,
                                decorators=[],
                                parent_class=name,
                            )
                            methods.append(m_fn)
                            m_sym = Symbol(
                                name=f"{name}.{m_name}",
                                kind=SymbolKind.METHOD,
                                file_path=file_structure.file_path,
                                line_span=m_span,
                                docstring=m_comment,
                                signature=m_fn.signature,
                                definition=m_fn,
                            )
                            file_structure.symbols[f"{name}.{m_name}"] = m_sym
                            m_comment = None

                cls = ClassDefinition(
                    name=name,
                    bases=bases,
                    methods=methods,
                    line_span=line_span,
                    docstring=doc_comment,
                    decorators=[],
                )
                file_structure.classes.append(cls)
                sym = Symbol(
                    name=name,
                    kind=SymbolKind.CLASS,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=doc_comment,
                    signature=cls.signature,
                    definition=cls,
                )
                file_structure.symbols[name] = sym
                pending_comment[0] = None

            elif node.type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                name = get_node_text(name_node) if name_node else "AnonymousInterface"
                line_span = node_to_linespan(node)
                sym = Symbol(
                    name=name,
                    kind=SymbolKind.INTERFACE,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=doc_comment,
                    signature=f"interface {name}",
                    definition=None,
                )
                file_structure.symbols[name] = sym
                pending_comment[0] = None

            elif node.type in (
                "export_statement",
                "lexical_declaration",
                "variable_declaration",
            ):
                for child in node.children:
                    if child.is_named:
                        walk_node(child)
            else:
                for child in node.children:
                    if child.is_named:
                        walk_node(child)

        for child in root_node.children:
            walk_node(child)

    def _extract_js_parameters(
        self, params_node: Node | None
    ) -> list[ArgumentDefinition]:
        """Extracts argument definitions from JS/TS formal_parameters node."""
        if not params_node:
            return []

        args: list[ArgumentDefinition] = []
        for child in params_node.children:
            if child.type in ("(", ")", ",", "{", "}"):
                continue

            if child.type == "identifier":
                args.append(
                    ArgumentDefinition(
                        name=get_node_text(child),
                        type_annotation=None,
                        default_value=None,
                    )
                )

            elif child.type in ("required_parameter", "optional_parameter"):
                p_name_node = child.child_by_field_name("pattern") or (
                    child.children[0] if child.children else None
                )
                p_type_node = child.child_by_field_name("type")
                p_val_node = child.child_by_field_name("value")
                p_name = get_node_text(p_name_node)
                p_type = (
                    get_node_text(p_type_node).lstrip(": ") if p_type_node else None
                )
                p_val = get_node_text(p_val_node) if p_val_node else None
                args.append(
                    ArgumentDefinition(
                        name=p_name,
                        type_annotation=p_type,
                        default_value=p_val,
                    )
                )

            elif child.type == "assignment_pattern":
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                p_name = get_node_text(left)
                p_val = get_node_text(right) if right else None
                args.append(
                    ArgumentDefinition(
                        name=p_name,
                        type_annotation=None,
                        default_value=p_val,
                    )
                )

            elif child.type == "rest_pattern":
                name_str = get_node_text(child)
                args.append(
                    ArgumentDefinition(
                        name=name_str,
                        type_annotation=None,
                        default_value=None,
                    )
                )

        return args

    def _extract_java(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Extracts Java classes, interfaces, methods, packages, and imports."""
        doc_comment: str | None = None
        pending_annotations: list[str] = []

        for child in root_node.children:
            if child.type in ("block_comment", "line_comment"):
                doc_comment = clean_docstring(get_node_text(child))
                continue

            if child.type in ("marker_annotation", "annotation"):
                pending_annotations.append(get_node_text(child))
                continue

            if child.type == "package_declaration":
                pkg_name_node = child.child_by_field_name("name") or next(
                    (
                        c
                        for c in child.children
                        if c.type in ("scoped_identifier", "identifier")
                    ),
                    None,
                )
                pkg_name = get_node_text(pkg_name_node)
                line_span = node_to_linespan(child)
                sym = Symbol(
                    name=pkg_name,
                    kind=SymbolKind.VARIABLE,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=None,
                    signature=f"package {pkg_name}",
                    definition=None,
                )
                file_structure.symbols[f"package:{pkg_name}"] = sym

            elif child.type == "import_declaration":
                line_span = node_to_linespan(child)
                import_text = get_node_text(child)
                mod_name = (
                    import_text.replace("import", "")
                    .replace("static", "")
                    .replace(";", "")
                    .strip()
                )
                imp = ImportDefinition(
                    module=mod_name,
                    names=[],
                    alias=None,
                    is_from_import=False,
                    line_span=line_span,
                )
                file_structure.imports.append(imp)
                sym = Symbol(
                    name=mod_name,
                    kind=SymbolKind.IMPORT,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=None,
                    signature=import_text.strip(),
                    definition=imp,
                )
                file_structure.symbols[f"import:{mod_name}"] = sym

            elif child.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):
                name_node = child.child_by_field_name("name")
                name = get_node_text(name_node) if name_node else "AnonymousClass"
                bases: list[str] = []
                annotations: list[str] = list(pending_annotations)

                for sub in child.children:
                    if sub.type == "modifiers":
                        for mod_sub in sub.children:
                            if mod_sub.type in ("marker_annotation", "annotation"):
                                annotations.append(get_node_text(mod_sub))
                    elif sub.type == "superclass":
                        for val in sub.children:
                            if val.type in ("type_identifier", "identifier"):
                                bases.append(get_node_text(val))
                    elif sub.type == "super_interfaces":
                        for val in sub.children:
                            if val.type in ("type_identifier", "identifier"):
                                bases.append(get_node_text(val))
                            elif val.type == "type_list":
                                for item in val.children:
                                    if item.type in (
                                        "type_identifier",
                                        "identifier",
                                    ):
                                        bases.append(get_node_text(item))

                line_span = node_to_linespan(child)
                body_node = child.child_by_field_name("body")
                methods: list[FunctionDefinition] = []

                if body_node:
                    m_doc: str | None = None
                    m_decs: list[str] = []
                    for m_child in body_node.children:
                        if m_child.type in ("block_comment", "line_comment"):
                            m_doc = clean_docstring(get_node_text(m_child))
                        elif m_child.type in ("marker_annotation", "annotation"):
                            m_decs.append(get_node_text(m_child))
                        elif m_child.type in (
                            "method_declaration",
                            "constructor_declaration",
                        ):
                            for m_sub in m_child.children:
                                if m_sub.type == "modifiers":
                                    for mod_sub in m_sub.children:
                                        if mod_sub.type in (
                                            "marker_annotation",
                                            "annotation",
                                        ):
                                            m_decs.append(get_node_text(mod_sub))

                            m_name_node = m_child.child_by_field_name("name")
                            m_name = (
                                get_node_text(m_name_node)
                                if m_name_node
                                else "anonymous"
                            )
                            m_type_node = m_child.child_by_field_name("type")
                            m_type = get_node_text(m_type_node) if m_type_node else None
                            m_span = node_to_linespan(m_child)
                            m_args = self._extract_java_parameters(
                                m_child.child_by_field_name("parameters")
                            )

                            fn = FunctionDefinition(
                                name=m_name,
                                args=m_args,
                                return_type=m_type,
                                line_span=m_span,
                                docstring=m_doc,
                                is_async=False,
                                decorators=m_decs,
                                parent_class=name,
                            )
                            methods.append(fn)
                            m_sym = Symbol(
                                name=f"{name}.{m_name}",
                                kind=SymbolKind.METHOD,
                                file_path=file_structure.file_path,
                                line_span=m_span,
                                docstring=m_doc,
                                signature=fn.signature,
                                definition=fn,
                            )
                            file_structure.symbols[f"{name}.{m_name}"] = m_sym
                            m_doc = None
                            m_decs = []

                kind = (
                    SymbolKind.INTERFACE
                    if child.type == "interface_declaration"
                    else SymbolKind.CLASS
                )
                cls = ClassDefinition(
                    name=name,
                    bases=bases,
                    methods=methods,
                    line_span=line_span,
                    docstring=doc_comment,
                    decorators=annotations,
                )
                file_structure.classes.append(cls)
                sym = Symbol(
                    name=name,
                    kind=kind,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=doc_comment,
                    signature=cls.signature,
                    definition=cls,
                )
                file_structure.symbols[name] = sym
                doc_comment = None
                pending_annotations = []

    def _extract_java_parameters(
        self, params_node: Node | None
    ) -> list[ArgumentDefinition]:
        """Extracts parameters from Java formal_parameters node."""
        if not params_node:
            return []

        args: list[ArgumentDefinition] = []
        for child in params_node.children:
            if child.type == "formal_parameter":
                p_type_node = child.child_by_field_name("type")
                p_name_node = child.child_by_field_name("name")
                p_type = get_node_text(p_type_node) if p_type_node else None
                p_name = (
                    get_node_text(p_name_node) if p_name_node else get_node_text(child)
                )
                args.append(
                    ArgumentDefinition(
                        name=p_name,
                        type_annotation=p_type,
                        default_value=None,
                    )
                )
            elif child.type == "spread_parameter":
                args.append(
                    ArgumentDefinition(
                        name=get_node_text(child),
                        type_annotation=None,
                        default_value=None,
                    )
                )

        return args

    def _extract_html(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Extracts major HTML tags, structural sections, IDs, and classes."""

        def walk_html(node: Node) -> None:
            if node.type == "element":
                start_tag = node.child_by_field_name("start_tag") or (
                    node.children[0] if node.children else None
                )
                tag_name = ""
                element_id = ""
                element_classes: list[str] = []

                if start_tag:
                    for child in start_tag.children:
                        if child.type == "tag_name":
                            tag_name = get_node_text(child)
                        elif child.type == "attribute":
                            attr_name = ""
                            attr_val = ""
                            for sub in child.children:
                                if sub.type == "attribute_name":
                                    attr_name = get_node_text(sub)
                                elif sub.type in (
                                    "quoted_attribute_value",
                                    "attribute_value",
                                ):
                                    attr_val = get_node_text(sub).strip("'\"")
                            if attr_name == "id":
                                element_id = attr_val
                            elif attr_name == "class":
                                element_classes = attr_val.split()

                if tag_name:
                    sym_parts = [tag_name]
                    if element_id:
                        sym_parts.append(f"#{element_id}")
                    if element_classes:
                        sym_parts.append("." + ".".join(element_classes[:2]))
                    sym_name = "".join(sym_parts)

                    meaningful_tags = (
                        "html",
                        "head",
                        "body",
                        "header",
                        "nav",
                        "main",
                        "footer",
                        "section",
                        "article",
                        "form",
                        "script",
                        "style",
                        "table",
                        "template",
                        "dialog",
                    )
                    if tag_name in meaningful_tags or element_id or element_classes:
                        line_span = node_to_linespan(node)
                        sym = Symbol(
                            name=sym_name,
                            kind=SymbolKind.TAG,
                            file_path=file_structure.file_path,
                            line_span=line_span,
                            docstring=None,
                            signature=f"<{tag_name}>",
                            definition=None,
                        )
                        file_structure.symbols[sym_name] = sym

            for child in node.children:
                if child.is_named:
                    walk_html(child)

        for child in root_node.children:
            walk_html(child)

    def _extract_css(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Extracts CSS rule sets, class/ID selectors, and CSS variables."""

        def walk_css(node: Node) -> None:
            if node.type == "rule_set":
                selectors_node = node.child_by_field_name("selectors") or (
                    node.children[0] if node.children else None
                )
                selectors_text = (
                    get_node_text(selectors_node).strip() if selectors_node else "rule"
                )
                line_span = node_to_linespan(node)

                sym = Symbol(
                    name=selectors_text,
                    kind=SymbolKind.RULE,
                    file_path=file_structure.file_path,
                    line_span=line_span,
                    docstring=None,
                    signature=selectors_text,
                    definition=None,
                )
                file_structure.symbols[selectors_text] = sym

            for child in node.children:
                if child.is_named:
                    walk_css(child)

        for child in root_node.children:
            walk_css(child)

    def _extract_generic(
        self, root_node: Node, code_bytes: bytes, file_structure: FileStructure
    ) -> None:
        """Generic fallback extractor for arbitrary languages."""
        line_span = node_to_linespan(root_node)
        sym = Symbol(
            name=file_structure.file_path.name,
            kind=SymbolKind.VARIABLE,
            file_path=file_structure.file_path,
            line_span=line_span,
            docstring=None,
            signature=file_structure.file_path.name,
            definition=None,
        )
        file_structure.symbols[file_structure.file_path.name] = sym


PythonASTParser = TreeSitterParser
MultiLanguageParser = TreeSitterParser
