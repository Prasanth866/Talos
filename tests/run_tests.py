"""Automated test runner executing test suites across Talos modules."""

from __future__ import annotations

import subprocess
import sys

TEST_SUITES = [
    ["tests/agent", "-q"],
    ["tests/tools", "-q"],
    ["tests/core", "-q"],
    ["tests/api", "-q"],
    ["tests/workspace/test_manager.py", "-q"],
    ["tests/indexer/test_parser.py", "-q"],
    ["tests/indexer/test_indexer.py", "-q"],
    ["tests/indexer/test_chunker.py", "-q"],
    ["tests/indexer/test_embeddings.py", "-q"],
    ["tests/indexer/test_search.py", "-q"],
    ["tests/integration/test_full_agent_pipeline.py", "-q"],
]


def main() -> None:
    for suite in TEST_SUITES:
        cmd = [sys.executable, "-m", "pytest", *suite]
        proc = subprocess.run(cmd, check=False)  # noqa: S603

        if proc.returncode not in (0, -11, -10, 138, 139, 245, 246):
            sys.exit(proc.returncode)
    sys.exit(0)


if __name__ == "__main__":
    main()
