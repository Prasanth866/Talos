import sys
from pathlib import Path

import pytest

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)
from src.tools.system_tools import (
    MAX_OUTPUT_CHARS,
    FileSystemTool,
    ShellTool,
    _truncate,
    async_read_file,
    async_write_file,
    read_file,
    run_shell,
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
