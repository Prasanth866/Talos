from src.indexer import (
    PythonASTParser,
    SymbolKind,
    TreeSitterParser,
)


def test_extract_function_definitions_with_correct_name_args_and_span() -> None:
    """Unit test: extract function definitions with correct name, args, line span."""
    parser = TreeSitterParser()
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


def test_extract_javascript_functions_and_classes() -> None:
    """Unit test: extract JS functions, methods, classes, and imports."""
    parser = TreeSitterParser()
    js_code = """
import { authHelper, logger as log } from './auth';
import axios from 'axios';

/**
 * Account management service
 */
class AccountManager extends BaseManager {
    /**
     * Finds active account
     */
    async findAccount(accountId, active = true) {
        return await db.lookup(accountId);
    }
}

function processPayment(amount, currency = 'USD') {
    return amount > 0;
}
"""
    structure = parser.parse("account.js", js_code)

    assert len(structure.imports) == 2
    assert len(structure.classes) == 1
    assert len(structure.functions) == 1

    cls = structure.classes[0]
    assert cls.name == "AccountManager"
    assert cls.bases == ["BaseManager"]
    assert cls.docstring == "Account management service"
    assert len(cls.methods) == 1

    method = cls.methods[0]
    assert method.name == "findAccount"
    assert method.is_async is True
    assert method.docstring == "Finds active account"
    assert len(method.args) == 2

    fn = structure.functions[0]
    assert fn.name == "processPayment"
    assert "processPayment" in structure.symbols
    assert "AccountManager.findAccount" in structure.symbols


def test_extract_typescript_interfaces_and_types() -> None:
    """Unit test: extract TS interfaces, classes, and typed functions."""
    parser = TreeSitterParser()
    ts_code = """
import { User } from './models';

interface UserProfile {
    id: number;
    username: string;
}

export class ProfileService {
    getUserProfile(userId: number): UserProfile {
        return { id: userId, username: "test" };
    }
}
"""
    structure = parser.parse("profile.ts", ts_code)

    assert "UserProfile" in structure.symbols
    assert structure.symbols["UserProfile"].kind == SymbolKind.INTERFACE

    assert len(structure.classes) == 1
    cls = structure.classes[0]
    assert cls.name == "ProfileService"
    assert len(cls.methods) == 1
    method = cls.methods[0]
    assert method.name == "getUserProfile"
    assert method.return_type == "UserProfile"


def test_extract_java_classes_methods_and_packages() -> None:
    """Unit test: extract Java packages, imports, classes, and methods."""
    parser = TreeSitterParser()
    java_code = """
package com.talos.service;

import java.util.List;
import java.util.Optional;

/**
 * Order processing service
 */
@Service
public class OrderService extends BaseService implements IOrderService {

    /**
     * Retrieve order by ID
     */
    @Override
    public Optional<Order> getOrder(String orderId, int maxRetries) {
        return Optional.empty();
    }
}
"""
    structure = parser.parse("OrderService.java", java_code)

    assert "package:com.talos.service" in structure.symbols
    assert len(structure.imports) == 2

    assert len(structure.classes) == 1
    cls = structure.classes[0]
    assert cls.name == "OrderService"
    assert "BaseService" in cls.bases
    assert "IOrderService" in cls.bases
    assert cls.docstring == "Order processing service"
    assert cls.decorators == ["@Service"]

    assert len(cls.methods) == 1
    method = cls.methods[0]
    assert method.name == "getOrder"
    assert method.return_type == "Optional<Order>"
    assert method.docstring == "Retrieve order by ID"
    assert len(method.args) == 2
    assert method.args[0].name == "orderId"
    assert method.args[0].type_annotation == "String"


def test_extract_html_tags_and_structure() -> None:
    """Unit test: extract HTML structural tags, IDs, and classes."""
    parser = TreeSitterParser()
    html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Talos Web</title>
</head>
<body>
    <header id="main-header" class="navbar shadow">
        <h1>Dashboard</h1>
    </header>
    <main id="app-content">
        <section class="metrics">Metric Data</section>
    </main>
</body>
</html>
"""
    structure = parser.parse("index.html", html_code)

    assert "html" in structure.symbols
    assert "head" in structure.symbols
    assert "body" in structure.symbols
    assert "header#main-header.navbar.shadow" in structure.symbols
    assert "main#app-content" in structure.symbols
    assert structure.symbols["main#app-content"].kind == SymbolKind.TAG


def test_extract_css_rules_and_selectors() -> None:
    """Unit test: extract CSS selectors and rule sets."""
    parser = TreeSitterParser()
    css_code = """
:root {
    --primary-color: #4f46e5;
}

.main-header {
    background-color: var(--primary-color);
    display: flex;
}

#app-content {
    margin: 0 auto;
}
"""
    structure = parser.parse("styles.css", css_code)

    assert ":root" in structure.symbols
    assert ".main-header" in structure.symbols
    assert "#app-content" in structure.symbols
    assert structure.symbols[".main-header"].kind == SymbolKind.RULE
