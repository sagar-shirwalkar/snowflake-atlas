"""Local directory source adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .base import MarkdownSource


class LocalSource(MarkdownSource):
    """Source adapter for a local directory of markdown files.

    No git, no crawl metadata — just a plain directory tree.
    Useful for ad-hoc collections or private docs.
    """

    def __init__(self, mirror_root: Path, repo_url: str = "local://docs", branch: str = "local") -> None:
        self.mirror_root = Path(mirror_root).resolve()
        self.repo_url = repo_url
        self.branch = branch

    def walk_markdown(self) -> Iterator[Path]:
        if not self.mirror_root.is_dir():
            raise FileNotFoundError(f"Mirror root not found: {self.mirror_root}")
        yield from sorted(self.mirror_root.rglob("*.md"))

    def get_metadata(self, path: Path) -> dict:
        rel = path.relative_to(self.mirror_root)
        parts = rel.parts
        return {
            "publication": parts[0] if parts else "unknown",
            "file": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "repo_url": self.repo_url,
            "branch": self.branch,
            "sha": "local",
        }

    def get_release_info(self) -> dict:
        return {
            "branch": self.branch,
            "sha": "local",
            "repo_url": self.repo_url,
            "file_count": sum(1 for _ in self.walk_markdown()),
        }
