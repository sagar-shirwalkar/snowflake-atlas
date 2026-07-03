"""Abstract base class for markdown source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class MarkdownSource(ABC):
    """Abstract source of markdown files with metadata.

    All source adapters must implement this interface to provide
    a uniform way for ``make_bundle.py`` to walk files and extract
    metadata regardless of the underlying source (git, web crawl, local, API).
    """

    @abstractmethod
    def walk_markdown(self) -> Iterator[Path]:
        """Yield absolute paths to all ``.md`` files in the source."""
        ...

    @abstractmethod
    def get_metadata(self, path: Path) -> dict:
        """Return source metadata for a file.

        Returns a dict with keys:
        - ``publication``: top-level folder name (e.g. "user-guide")
        - ``file``: path relative to publication folder
        - ``repo_url``: source URL (git repo or web source)
        - ``branch``: branch name or crawl identifier
        - ``sha``: git commit SHA or crawl SHA/timestamp
        """
        ...

    @abstractmethod
    def get_release_info(self) -> dict:
        """Return release metadata for the entire source.

        Returns a dict with keys:
        - ``branch``: branch name or crawl identifier
        - ``sha``: git commit SHA or crawl SHA/timestamp
        - ``repo_url``: source URL
        - ``file_count``: total number of markdown files
        """
        ...
