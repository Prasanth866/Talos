from pathlib import Path

import pytest

from src.agent.dispatcher import ToolDispatcher, create_default_dispatcher
from src.agent.models import ToolCall


@pytest.mark.asyncio
async def test_tool_dispatcher_registration_and_execution() -> None:
    dispatcher = ToolDispatcher()

    def add_numbers(a: int, b: int) -> int:
        return a + b

    dispatcher.register_tool(
        name="add",
        description="Adds two integers.",
        handler=add_numbers,
        parameters_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    )

    assert dispatcher.has_tool("add")
    assert "add" in dispatcher.get_tool_names()

    call = ToolCall(tool_name="add", arguments={"a": 5, "b": 7})
    result = await dispatcher.execute_tool(call)

    assert result.success is True
    assert result.output == "12"
    assert result.error is None
    assert result.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_tool_dispatcher_unknown_tool() -> None:
    dispatcher = ToolDispatcher()
    call = ToolCall(tool_name="non_existent", arguments={})
    result = await dispatcher.execute_tool(call)

    assert result.success is False
    assert "Unknown tool 'non_existent'" in result.error  # type: ignore[operator]


@pytest.mark.asyncio
async def test_tool_dispatcher_invalid_arguments() -> None:
    dispatcher = ToolDispatcher()

    def greet(name: str) -> str:
        return f"Hello, {name}!"

    dispatcher.register_tool(name="greet", description="Greets person", handler=greet)

    call = ToolCall(tool_name="greet", arguments={"wrong_arg": "value"})
    result = await dispatcher.execute_tool(call)

    assert result.success is False
    assert "Invalid arguments" in result.error  # type: ignore[operator]


@pytest.mark.asyncio
async def test_default_dispatcher_file_and_shell_tools(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    dispatcher = create_default_dispatcher(sandbox)
    assert set(dispatcher.get_tool_names()) == {
        "read_file",
        "write_file",
        "list_dir",
        "run_shell",
        "get_symbol_definition",
        "list_file_structure",
        "semantic_search",
        "hybrid_search",
        "apply_patch",
    }

    write_call = ToolCall(
        tool_name="write_file",
        arguments={"path": "test.txt", "content": "hello talos"},
    )
    write_res = await dispatcher.execute_tool(write_call)
    assert write_res.success is True

    py_code = """
def add(a: int, b: int = 1) -> int:
    \"\"\"Adds two numbers.\"\"\"
    return a + b
"""
    await dispatcher.execute_tool(
        ToolCall(
            tool_name="write_file",
            arguments={"path": "math_mod.py", "content": py_code},
        )
    )

    sym_call = ToolCall(
        tool_name="get_symbol_definition",
        arguments={"symbol_name": "add"},
    )
    sym_res = await dispatcher.execute_tool(sym_call)
    assert sym_res.success is True
    assert "def add(a: int, b: int = 1) -> int" in sym_res.output
    assert "Adds two numbers." in sym_res.output

    struct_call = ToolCall(
        tool_name="list_file_structure",
        arguments={"path": "math_mod.py"},
    )
    struct_res = await dispatcher.execute_tool(struct_call)
    assert struct_res.success is True
    assert "Functions (1):" in struct_res.output
    assert "def add(a: int, b: int = 1) -> int" in struct_res.output

    sem_call = ToolCall(
        tool_name="semantic_search",
        arguments={"query": "function that adds numbers"},
    )
    sem_res = await dispatcher.execute_tool(sem_call)
    assert sem_res.success is True
    assert "add" in sem_res.output

    hyb_call = ToolCall(
        tool_name="hybrid_search",
        arguments={"query": "add"},
    )
    hyb_res = await dispatcher.execute_tool(hyb_call)
    assert hyb_res.success is True
    assert "EXACT" in hyb_res.output or "HYBRID" in hyb_res.output
    assert "add" in hyb_res.output

    list_call = ToolCall(tool_name="list_dir", arguments={"path": "."})
    list_res = await dispatcher.execute_tool(list_call)
    assert list_res.success is True
    assert "test.txt" in list_res.output

    read_call = ToolCall(tool_name="read_file", arguments={"path": "test.txt"})
    read_res = await dispatcher.execute_tool(read_call)
    assert read_res.success is True
    assert read_res.output == "hello talos"

    shell_call = ToolCall(
        tool_name="run_shell", arguments={"command": "echo 'running shell'"}
    )
    shell_res = await dispatcher.execute_tool(shell_call)
    assert shell_res.success is True
    assert "running shell" in shell_res.output

    bad_list = ToolCall(tool_name="list_dir", arguments={"path": "non_existent_folder"})
    bad_list_res = await dispatcher.execute_tool(bad_list)
    assert "does not exist" in bad_list_res.output

    file_list = ToolCall(tool_name="list_dir", arguments={"path": "test.txt"})
    file_list_res = await dispatcher.execute_tool(file_list)
    assert "is not a directory" in file_list_res.output

    patch_call = ToolCall(
        tool_name="apply_patch",
        arguments={
            "patch": (
                "--- a/test.txt\n"
                "+++ b/test.txt\n"
                "@@ -1,1 +1,1 @@\n"
                "-hello talos\n"
                "+hello talos patched\n"
            ),
        },
    )
    patch_res = await dispatcher.execute_tool(patch_call)
    assert patch_res.success is True
    assert "Successfully applied patch" in (patch_res.output or "")


@pytest.mark.asyncio
async def test_tool_dispatcher_async_and_various_outputs() -> None:
    dispatcher = ToolDispatcher()

    async def async_fetch() -> bytes:
        return b"binary data"

    def none_op() -> None:
        return None

    def dict_op() -> dict[str, str]:
        return {"status": "ok"}

    dispatcher.register_tool(name="fetch", description="Fetch", handler=async_fetch)
    dispatcher.register_tool(name="noop", description="Noop", handler=none_op)
    dispatcher.register_tool(name="dict_op", description="Dict", handler=dict_op)

    r1 = await dispatcher.execute_tool(ToolCall(tool_name="fetch", arguments={}))
    assert r1.success is True
    assert r1.output == "binary data"

    r2 = await dispatcher.execute_tool(ToolCall(tool_name="noop", arguments={}))
    assert r2.success is True
    assert r2.output == "Operation completed successfully."

    r3 = await dispatcher.execute_tool(ToolCall(tool_name="dict_op", arguments={}))
    assert r3.success is True
    assert r3.output == "{'status': 'ok'}"
