"""Web crawl source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .base import MarkdownSource


class WebCrawlSource(MarkdownSource):
    """Source adapter for a local mirror created by a web crawler.

    Expects the crawler to have created a ``crawl_meta.json`` file at
    the mirror root with at least:
    - ``source_url``: the original documentation URL (e.g. "https://docs.snowflake.com")
    - ``crawled_at``: ISO timestamp of when the crawl was performed
    - ``crawler_sha``: (optional) git SHA of the crawler code used
    """

    def __init__(self, mirror_root: Path, crawl_meta_path: Path | None = None) -> None:
        """Initialize the web crawl source adapter."""
        self.mirror_root = Path(mirror_root).resolve()
        self.crawl_meta = self._load_meta(crawl_meta_path)

    def _load_meta(self, crawl_meta_path: Path | None) -> dict:
        if crawl_meta_path is None:
            crawl_meta_path = self.mirror_root / "crawl_meta.json"
        if crawl_meta_path.is_file():
            return json.loads(crawl_meta_path.read_text())
        # Fallback for backwards compatibility
        return {
            "source_url": "https://docs.snowflake.com",
            "crawled_at": "unknown",
            "crawler_sha": "unknown",
        }

    def walk_markdown(self) -> Iterator[Path]:
        """Yield all ``.md`` file paths under the mirror root."""
        if not self.mirror_root.is_dir():
            raise FileNotFoundError(f"Mirror root not found: {self.mirror_root}")
        yield from sorted(self.mirror_root.rglob("*.md"))

    def get_metadata(self, path: Path) -> dict:
        """Return publication, file path, and source info for a markdown file."""
        rel = path.relative_to(self.mirror_root)
        parts = rel.parts
        return {
            "publication": parts[0] if parts else "unknown",
            "file": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "repo_url": self.crawl_meta.get("source_url", "https://docs.snowflake.com"),
            "branch": f"crawl-{self.crawl_meta.get('crawled_at', 'unknown')[:10]}",
            "sha": self.crawl_meta.get("crawler_sha", "unknown"),
        }

    def get_release_info(self) -> dict:
        """Return crawl source URL, branch, SHA, and file count."""
        return {
            "branch": f"crawl-{self.crawl_meta.get('crawled_at', 'unknown')[:10]}",
            "sha": self.crawl_meta.get("crawler_sha", "unknown"),
            "repo_url": self.crawl_meta.get("source_url", "https://docs.snowflake.com"),
            "file_count": sum(1 for _ in self.walk_markdown()),
        }
