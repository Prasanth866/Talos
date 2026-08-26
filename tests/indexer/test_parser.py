from src.indexer import PythonASTParser, SymbolKind


def test_extract_function_definitions_with_correct_name_args_and_span() -> None:
    """Unit test: extract function definitions with correct name, args, line span."""
    parser = PythonASTParser()
    code = """
def calculate_total(price: float, tax_rate: float = 0.05, *items, **metadata) -> float:
    \"\"\"Calculates total price including taxes.\"\"\"
    return price * (1 + tax_rate)
"""
    structure = parser.parse("calc.py", code)

    assert len(structure.functions) == 1
    fn = structure.functions[0]
    assert fn.name == "calculate_total"
    assert fn.return_type == "float"
    assert fn.docstring == "Calculates total price including taxes."
    assert fn.line_span.start_line == 2
    assert fn.line_span.end_line == 4
    assert not fn.is_async

    # Verify arguments
    arg_names = [a.name for a in fn.args]
    assert "price" in arg_names
    assert "tax_rate" in arg_names
    assert "*items" in arg_names
    assert "**metadata" in arg_names

    tax_arg = next(a for a in fn.args if a.name == "tax_rate")
    assert tax_arg.type_annotation == "float"
    assert tax_arg.default_value == "0.05"

    # Verify symbol index
    assert "calculate_total" in structure.symbols
    sym = structure.symbols["calculate_total"]
    assert sym.kind == SymbolKind.FUNCTION
    assert "calculate_total(price: float" in sym.signature


def test_extract_class_definitions_with_docstrings_and_methods() -> None:
    """Unit test: extract class definitions with docstring, bases, methods."""
    parser = PythonASTParser()
    code = """
@dataclass
class UserService(BaseService, AuthMixin):
    \"\"\"Service handling user authentication and profiles.\"\"\"

    db_session: Session

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        \"\"\"Fetches a user by primary key ID.\"\"\"
        return await self.db.find(user_id)
"""
    structure = parser.parse("service.py", code)

    assert len(structure.classes) == 1
    cls = structure.classes[0]
    assert cls.name == "UserService"
    assert cls.bases == ["BaseService", "AuthMixin"]
    assert cls.docstring == "Service handling user authentication and profiles."
    assert len(cls.methods) == 1
    assert cls.decorators == ["@dataclass"]

    method = cls.methods[0]
    assert method.name == "get_user_by_id"
    assert method.is_async is True
    assert method.parent_class == "UserService"
    assert method.return_type == "Optional[User]"
    assert method.docstring == "Fetches a user by primary key ID."

    # Verify symbol dictionary includes qualified method name
    assert "UserService" in structure.symbols
    assert "UserService.get_user_by_id" in structure.symbols
    assert structure.symbols["UserService.get_user_by_id"].kind == SymbolKind.METHOD


def test_extract_imports_from_module() -> None:
    """Unit test: extract imports statement variations."""
    parser = PythonASTParser()
    code = """
import os
import sys as system
from typing import Optional, List as L, Dict
from ..core.config import Settings
"""
    structure = parser.parse("imports_test.py", code)

    assert len(structure.imports) == 4
    imp_modules = [i.module for i in structure.imports]
    assert "os" in imp_modules
    assert "sys" in imp_modules
    assert "typing" in imp_modules
    assert "..core.config" in imp_modules

    aliased_sys = next(i for i in structure.imports if i.module == "sys")
    assert aliased_sys.alias == "system"

    from_typing = next(i for i in structure.imports if i.module == "typing")
    assert from_typing.is_from_import is True
    assert "Optional" in from_typing.names
    assert "List as L" in from_typing.names
    assert "Dict" in from_typing.names


def test_parser_fault_tolerance_on_syntax_errors() -> None:
    """Unit test: parser recovers from syntax errors and extracts valid symbols."""

    parser = PythonASTParser()
    broken_code = """
def good_func_1(a: int) -> int:
    \"\"\"First good function.\"\"\"
    return a + 1

# Syntax error: missing colon on if statement
if True
    x = 100

def good_func_2(b: str) -> str:
    \"\"\"Second good function.\"\"\"
    return b.strip()
"""
    structure = parser.parse("broken.py", broken_code)

    assert structure.has_syntax_errors is True
    fn_names = [f.name for f in structure.functions]
    assert "good_func_1" in fn_names
    assert "good_func_2" in fn_names

    f1 = next(f for f in structure.functions if f.name == "good_func_1")
    assert f1.docstring == "First good function."

    f2 = next(f for f in structure.functions if f.name == "good_func_2")
    assert f2.docstring == "Second good function."
