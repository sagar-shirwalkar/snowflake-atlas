"""Git repository source adapter."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

from .base import MarkdownSource


class GitSource(MarkdownSource):
    """Source adapter for a local git clone of a markdown documentation repo.

    Expects the standard layout:
    ``repo_root/markdown/<publication>/*.md``
    """

    def __init__(self, repo_path: Path, repo_url: str, branch: str) -> None:
        """Initialize the Git source adapter."""
        self.repo_path = Path(repo_path).resolve()
        self.repo_url = repo_url
        self.branch = branch
        self._sha: str | None = None

    def walk_markdown(self) -> Iterator[Path]:
        """Yield all ``.md`` file paths under the markdown directory."""
        md_root = self.repo_path / "markdown"
        if not md_root.is_dir():
            raise FileNotFoundError(f"No 'markdown/' directory at {md_root}")
        yield from sorted(md_root.rglob("*.md"))

    def get_metadata(self, path: Path) -> dict:
        """Return publication, file path, and source info for a markdown file."""
        rel = path.relative_to(self.repo_path / "markdown")
        parts = rel.parts
        return {
            "publication": parts[0] if parts else "unknown",
            "file": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "repo_url": self.repo_url,
            "branch": self.branch,
            "sha": self._get_sha(),
        }

    def get_release_info(self) -> dict:
        """Return branch, SHA, repo URL, and file count."""
        return {
            "branch": self.branch,
            "sha": self._get_sha(),
            "repo_url": self.repo_url,
            "file_count": sum(1 for _ in self.walk_markdown()),
        }

    def _get_sha(self) -> str:
        if self._sha is None:
            try:
                self._sha = subprocess.run(
                    ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
                self._sha = "unknown"
        return self._sha
