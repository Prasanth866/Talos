from __future__ import annotations

from pathlib import Path
from typing import Any

import git
import structlog

from src.workspace.exceptions import GitCloneError

logger = structlog.get_logger(__name__)


def shallow_clone(
    repo_url: str,
    target_dir: Path,
    commit_sha: str | None = None,
    depth: int = 1,
) -> str:
    """Performs a Git shallow clone into target_dir and optionally checks out a commit.

    Returns the resulting HEAD commit SHA.
    Raises GitCloneError on any failure.
    """
    logger.info(
        "git_shallow_clone_started",
        repo_url=repo_url,
        target_dir=str(target_dir),
        commit_sha=commit_sha,
        depth=depth,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = git.Repo.clone_from(
            url=repo_url,
            to_path=str(target_dir),
            depth=depth,
        )

        if commit_sha:
            try:
                repo.git.checkout(commit_sha)
            except git.exc.GitError:
                repo.git.fetch("origin", commit_sha, depth=depth)
                repo.git.checkout(commit_sha)

        resolved_sha = str(repo.head.commit.hexsha)
        logger.info(
            "git_shallow_clone_completed",
            repo_url=repo_url,
            commit_sha=resolved_sha,
        )
        return resolved_sha

    except git.exc.GitError as exc:
        details: dict[str, Any] = {"error": str(exc)}
        if isinstance(exc, git.exc.GitCommandError):
            details["status"] = exc.status
            details["stderr"] = exc.stderr
            details["stdout"] = exc.stdout
        logger.error(
            "git_shallow_clone_failed",
            repo_url=repo_url,
            error=str(exc),
            details=details,
        )
        raise GitCloneError(
            message=f"Failed to clone repository '{repo_url}': {exc}",
            repo_url=repo_url,
            commit_sha=commit_sha,
            details=details,
        ) from exc
    except Exception as exc:
        logger.error(
            "git_shallow_clone_unexpected_error",
            repo_url=repo_url,
            error=str(exc),
        )
        raise GitCloneError(
            message=f"Unexpected error cloning repository '{repo_url}': {exc}",
            repo_url=repo_url,
            commit_sha=commit_sha,
            details={"error": str(exc)},
        ) from exc
