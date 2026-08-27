import sys
from pathlib import Path

import pytest

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)
from src.tools.shell import _truncate
from src.tools.system_tools import (
    MAX_OUTPUT_CHARS,
    FileSystemTool,
    ShellTool,
    async_read_bytes,
    async_read_file,
    async_write_bytes,
    async_write_file,
    read_bytes,
    read_file,
    run_shell,
    write_bytes,
    write_file,
)


def test_file_write_and_read_success(tmp_path: Path) -> None:
    """Verifies standard file creation and reading inside isolated tmp_path."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    file_path = "subfolder/test.txt"
    content = "Hello, defensive world!"

    fs.write_file(file_path, content)
    result = fs.read_file(file_path)

    assert result == content
    assert (tmp_path / file_path).exists()


def test_standalone_read_and_write_file_helpers(tmp_path: Path) -> None:
    """Verifies read_file and write_file standalone helper functions."""
    file_path = "helpers/test.txt"
    content = "Helper functions test"

    write_file(tmp_path, file_path, content)
    result = read_file(tmp_path, file_path)

    assert result == content


def test_path_traversal_prevention_on_read(tmp_path: Path) -> None:
    """Ensures read attempts outside sandbox trigger PathTraversalError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)

    with pytest.raises(PathTraversalError) as exc_info:
        fs.read_file("../../etc/passwd")

    assert "Access denied" in str(exc_info.value)
    assert exc_info.value.tool_name == "FileSystemTool"
    assert exc_info.value.attempted_path != ""
    assert "attempted_path" in exc_info.value.details
    assert exc_info.value.to_dict()["error"] == "PathTraversalError"


def test_path_traversal_prevention_on_write(tmp_path: Path) -> None:
    """Ensures write attempts outside sandbox trigger PathTraversalError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)

    with pytest.raises(PathTraversalError) as exc_info:
        fs.write_file("../../etc/malicious.txt", "hacked")

    assert "Access denied" in str(exc_info.value)
    assert exc_info.value.tool_name == "FileSystemTool"
    assert exc_info.value.attempted_path != ""


def test_read_non_existent_file(tmp_path: Path) -> None:
    """Ensures missing files raise structured ToolError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)

    with pytest.raises(ToolError) as exc_info:
        fs.read_file("non_existent.txt")

    assert "File not found" in str(exc_info.value)


def test_read_directory_path_raises_tool_error(tmp_path: Path) -> None:
    """Ensures attempting to read a directory raises ToolError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    dir_path = tmp_path / "sub_dir"
    dir_path.mkdir()

    with pytest.raises(ToolError) as exc_info:
        fs.read_file("sub_dir")

    assert "is a directory" in str(exc_info.value)


def test_empty_file_path_raises_tool_error(tmp_path: Path) -> None:
    """Ensures empty file path input raises ToolError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)

    with pytest.raises(ToolError) as exc_info:
        fs.read_file("   ")

    assert "File path cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_shell_execution_success(tmp_path: Path) -> None:
    """Verifies execution of a valid command."""
    shell = ShellTool(working_dir=tmp_path)
    result = await shell.run_shell("echo 'Hello pytest'")

    assert result["exit_code"] == 0
    assert result["stdout"] == "Hello pytest"


@pytest.mark.asyncio
async def test_standalone_run_shell_helper(tmp_path: Path) -> None:
    """Verifies the standalone run_shell helper function."""
    result = await run_shell(tmp_path, "echo 'Standalone run_shell'")

    assert result["exit_code"] == 0
    assert result["stdout"] == "Standalone run_shell"


@pytest.mark.asyncio
async def test_shell_command_failure(tmp_path: Path) -> None:
    """Verifies non-zero exit codes raise CommandExecutionError with details."""
    shell = ShellTool(working_dir=tmp_path)

    with pytest.raises(CommandExecutionError) as exc_info:
        await shell.run_shell("ls /non_existent_directory_path")

    assert exc_info.value.exit_code != 0
    assert exc_info.value.stderr is not None
    assert len(exc_info.value.stderr) > 0
    assert "stdout" in exc_info.value.details


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path) -> None:
    """Verifies long-running process raises ExecutionTimeoutError."""
    shell = ShellTool(working_dir=tmp_path, timeout_seconds=0.1)

    with pytest.raises(ExecutionTimeoutError) as exc_info:
        await shell.run_shell("sleep 1")

    assert "timed out" in str(exc_info.value)
    assert exc_info.value.timeout_seconds == 0.1
    assert exc_info.value.details["timeout_seconds"] == 0.1


@pytest.mark.asyncio
async def test_shell_output_truncation(tmp_path: Path) -> None:
    """Verifies output exceeding MAX_OUTPUT_CHARS is truncated."""
    shell = ShellTool(working_dir=tmp_path)
    large_cmd = f"{sys.executable} -c \"print('A' * 5000)\""
    result = await shell.run_shell(large_cmd)

    assert result["exit_code"] == 0
    stdout = str(result["stdout"])
    assert len(stdout) < 5000
    assert stdout.startswith("A" * MAX_OUTPUT_CHARS)
    assert "[truncated 1000 chars]" in stdout


def test_truncate_helper() -> None:
    """Verifies _truncate helper behavior."""
    assert _truncate("short", limit=10) == "short"
    truncated = _truncate("ABCDE" * 10, limit=20)
    assert truncated.startswith("ABCDE" * 4)
    assert "[truncated 30 chars]" in truncated


def test_custom_exception_initializers() -> None:
    """Verifies initializers and attributes of custom exception classes."""
    pt_err = PathTraversalError("Denied", tool_name="FS", attempted_path="/etc/passwd")
    assert pt_err.attempted_path == "/etc/passwd"
    assert pt_err.details["attempted_path"] == "/etc/passwd"

    et_err = ExecutionTimeoutError("Timeout", tool_name="Shell", timeout_seconds=5.0)
    assert et_err.timeout_seconds == 5.0
    assert et_err.details["timeout_seconds"] == 5.0

    ce_err = CommandExecutionError(
        "Failed", tool_name="Shell", exit_code=1, stderr="err msg"
    )
    assert ce_err.exit_code == 1
    assert ce_err.stderr == "err msg"
    assert ce_err.details["exit_code"] == 1
    assert ce_err.details["stderr"] == "err msg"


@pytest.mark.asyncio
async def test_shell_executable_not_found(tmp_path: Path) -> None:
    """Verifies missing executable raises ToolError."""
    shell = ShellTool(working_dir=tmp_path)

    with pytest.raises(ToolError) as exc_info:
        await shell.run_shell("invalid_command_xyz_123")

    assert "not found on system PATH" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_shell_command_raises_tool_error(tmp_path: Path) -> None:
    """Verifies empty shell command raises ToolError."""
    shell = ShellTool(working_dir=tmp_path)

    with pytest.raises(ToolError) as exc_info:
        await shell.run_shell("   ")

    assert "Command string cannot be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_file_write_and_read(tmp_path: Path) -> None:
    """Verifies async methods and helpers offload I/O safely."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    file_path = "async_test/file.txt"
    content = "Async defensive file content"

    await fs.async_write_file(file_path, content)
    result = await fs.async_read_file(file_path)
    assert result == content

    helper_file = "async_test/helper.txt"
    await async_write_file(tmp_path, helper_file, "Helper async content")
    helper_result = await async_read_file(tmp_path, helper_file)
    assert helper_result == "Helper async content"


def test_binary_file_write_and_read(tmp_path: Path) -> None:
    """Verifies binary file writing and reading without UnicodeDecodeError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    file_path = "binary/test.bin"
    raw_bytes = b"\x80\x81\xff\xfe\x00\x01\x02\x03\xde\xad\xbe\xef"

    fs.write_bytes(file_path, raw_bytes)
    result_bytes = fs.read_bytes(file_path)
    assert result_bytes == raw_bytes

    flag_file = "binary/flag_test.bin"
    fs.write_file(flag_file, raw_bytes, binary=True)
    assert fs.read_file(flag_file, binary=True) == raw_bytes

    helper_file = "binary/helper.bin"
    write_bytes(tmp_path, helper_file, raw_bytes)
    assert read_bytes(tmp_path, helper_file) == raw_bytes
    assert read_file(tmp_path, helper_file, binary=True) == raw_bytes


@pytest.mark.asyncio
async def test_async_binary_file_operations(tmp_path: Path) -> None:
    """Verifies async binary read and write operations."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    file_path = "async_binary/data.bin"
    raw_bytes = b"\x00\xff\xfe\xfd\x80\x90"

    await fs.async_write_bytes(file_path, raw_bytes)
    result = await fs.async_read_bytes(file_path)
    assert result == raw_bytes

    helper_file = "async_binary/helper.bin"
    await async_write_bytes(tmp_path, helper_file, raw_bytes)
    helper_res = await async_read_bytes(tmp_path, helper_file)
    assert helper_res == raw_bytes


@pytest.mark.asyncio
async def test_read_file_overload_signatures(tmp_path: Path) -> None:
    """Verifies read_file and async_read_file overloads work correctly."""
    fs = FileSystemTool(sandbox_dir=tmp_path)
    file_path = "overload_test.txt"
    fs.write_file(file_path, "overload text")

    str_res: str = fs.read_file(file_path, binary=False)
    assert isinstance(str_res, str)

    default_str_res: str = fs.read_file(file_path)
    assert isinstance(default_str_res, str)

    bin_res: bytes = fs.read_file(file_path, binary=True)
    assert isinstance(bin_res, bytes)

    dynamic_flag: bool = True
    union_res: str | bytes = fs.read_file(file_path, binary=dynamic_flag)
    assert isinstance(union_res, bytes)

    async_str: str = await fs.async_read_file(file_path, binary=False)
    assert isinstance(async_str, str)

    async_bin: bytes = await fs.async_read_file(file_path, binary=True)
    assert isinstance(async_bin, bytes)

    helper_str: str = read_file(tmp_path, file_path, binary=False)
    assert isinstance(helper_str, str)

    helper_bin: bytes = read_file(tmp_path, file_path, binary=True)
    assert isinstance(helper_bin, bytes)

    async_helper_str: str = await async_read_file(tmp_path, file_path, binary=False)
    assert isinstance(async_helper_str, str)

    async_helper_bin: bytes = await async_read_file(tmp_path, file_path, binary=True)
    assert isinstance(async_helper_bin, bytes)


@pytest.mark.asyncio
async def test_shell_denied_executables_and_patterns(tmp_path: Path) -> None:
    """Verifies that dangerous commands and executables are blocked by ShellTool."""
    tool = ShellTool(working_dir=tmp_path)

    with pytest.raises(ToolError, match="not permitted"):
        await tool.run_shell("sudo ls")

    with pytest.raises(ToolError, match="not permitted"):
        await tool.run_shell("curl https://example.com")

    with pytest.raises(ToolError, match="denied safety pattern"):
        await tool.run_shell("rm -rf /")

    with pytest.raises(ToolError, match="denied safety pattern"):
        await tool.run_shell("echo test | bash")
