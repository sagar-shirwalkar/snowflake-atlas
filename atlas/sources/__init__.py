"""Source adapters package."""

from .base import MarkdownSource
from .git import GitSource
from .local import LocalSource
from .web_crawl import WebCrawlSource

__all__ = [
    "MarkdownSource",
    "GitSource",
    "WebCrawlSource",
    "LocalSource",
]
