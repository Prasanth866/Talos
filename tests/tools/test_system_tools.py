from pathlib import Path

import pytest

from src.tools.exceptions import (
    CommandExecutionError,
    ExecutionTimeoutError,
    PathTraversalError,
    ToolError,
)
from src.tools.system_tools import (
    FileSystemTool,
    ShellTool,
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
    assert exc_info.value.to_dict()["error"] == "PathTraversalError"


def test_path_traversal_prevention_on_write(tmp_path: Path) -> None:
    """Ensures write attempts outside sandbox trigger PathTraversalError."""
    fs = FileSystemTool(sandbox_dir=tmp_path)

    with pytest.raises(PathTraversalError) as exc_info:
        fs.write_file("../../etc/malicious.txt", "hacked")

    assert "Access denied" in str(exc_info.value)
    assert exc_info.value.tool_name == "FileSystemTool"


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

    assert exc_info.value.details["exit_code"] != 0
    assert len(exc_info.value.details["stderr"]) > 0


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path: Path) -> None:
    """Verifies long-running process raises ExecutionTimeoutError."""
    shell = ShellTool(working_dir=tmp_path, timeout_seconds=0.1)

    with pytest.raises(ExecutionTimeoutError) as exc_info:
        await shell.run_shell("sleep 1")

    assert "timed out" in str(exc_info.value)


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
