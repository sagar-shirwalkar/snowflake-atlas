"""Tests for the source adapters (atlas/sources/)."""

import json
from pathlib import Path

import pytest

from atlas.sources import GitSource, LocalSource, MarkdownSource, WebCrawlSource

# ---------------------------------------------------------------------------
# MarkdownSource ABC
# ---------------------------------------------------------------------------

def test_markdown_source_abstract():
    with pytest.raises(TypeError):
        MarkdownSource()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LocalSource
# ---------------------------------------------------------------------------

class TestLocalSource:
    def test_walk_markdown_finds_md_files(self, tmp_path: Path):
        (tmp_path / "pub1").mkdir()
        (tmp_path / "pub1" / "page1.md").write_text("# Hello")
        (tmp_path / "pub1" / "page2.md").write_text("# World")
        source = LocalSource(tmp_path)
        files = list(source.walk_markdown())
        assert len(files) == 2
        assert all(f.suffix == ".md" for f in files)

    def test_walk_markdown_ignores_non_md(self, tmp_path: Path):
        (tmp_path / "pub").mkdir()
        (tmp_path / "pub" / "doc.md").write_text("# Doc")
        (tmp_path / "pub" / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "pub" / "notes.txt").write_text("text")
        source = LocalSource(tmp_path)
        files = list(source.walk_markdown())
        assert len(files) == 1
        assert files[0].suffix == ".md"

    def test_walk_markdown_nested_dirs(self, tmp_path: Path):
        (tmp_path / "pub" / "sub").mkdir(parents=True)
        (tmp_path / "pub" / "a.md").write_text("# A")
        (tmp_path / "pub" / "sub" / "b.md").write_text("# B")
        source = LocalSource(tmp_path)
        files = list(source.walk_markdown())
        assert len(files) == 2

    def test_walk_markdown_missing_dir(self, tmp_path: Path):
        source = LocalSource(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="Mirror root not found"):
            list(source.walk_markdown())

    def test_walk_markdown_empty_dir(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        files = list(source.walk_markdown())
        assert files == []

    def test_get_metadata(self, tmp_path: Path):
        (tmp_path / "pub1").mkdir()
        f = tmp_path / "pub1" / "doc.md"
        f.write_text("# Doc")
        source = LocalSource(tmp_path, repo_url="local://test", branch="local")
        meta = source.get_metadata(f)
        assert meta["publication"] == "pub1"
        assert meta["file"] == "doc.md"
        assert meta["repo_url"] == "local://test"
        assert meta["branch"] == "local"
        assert meta["sha"] == "local"

    def test_get_metadata_nested(self, tmp_path: Path):
        (tmp_path / "pub" / "sub").mkdir(parents=True)
        f = tmp_path / "pub" / "sub" / "deep.md"
        f.write_text("# Deep")
        source = LocalSource(tmp_path)
        meta = source.get_metadata(f)
        assert meta["publication"] == "pub"
        assert meta["file"] == "sub/deep.md"

    def test_get_release_info(self, tmp_path: Path):
        (tmp_path / "pub").mkdir()
        (tmp_path / "pub" / "a.md").write_text("# A")
        source = LocalSource(tmp_path, repo_url="local://docs", branch="local")
        info = source.get_release_info()
        assert info["branch"] == "local"
        assert info["sha"] == "local"
        assert info["repo_url"] == "local://docs"
        assert info["file_count"] == 1


# ---------------------------------------------------------------------------
# WebCrawlSource
# ---------------------------------------------------------------------------

class TestWebCrawlSource:
    def test_basic_metadata(self, tmp_path: Path):
        (tmp_path / "docs").mkdir()
        f = tmp_path / "docs" / "page.md"
        f.write_text("# Page")
        source = WebCrawlSource(tmp_path)
        meta = source.get_metadata(f)
        assert meta["publication"] == "docs"
        assert meta["file"] == "page.md"
        assert meta["repo_url"] == "https://docs.snowflake.com"

    def test_with_crawl_meta(self, tmp_path: Path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "page.md").write_text("# Page")
        meta_path = tmp_path / "crawl_meta.json"
        meta_path.write_text(
            json.dumps({
                "source_url": "https://docs.snowflake.com/en",
                "crawled_at": "2025-06-01T12:00:00Z",
                "crawler_sha": "abc123def456",
            })
        )
        source = WebCrawlSource(tmp_path, meta_path)
        meta = source.get_metadata(tmp_path / "docs" / "page.md")
        assert meta["repo_url"] == "https://docs.snowflake.com/en"
        assert "crawl-2025-06-01" in meta["branch"]
        assert meta["sha"] == "abc123def456"

    def test_walk_markdown(self, tmp_path: Path):
        (tmp_path / "ug").mkdir()
        (tmp_path / "ug" / "intro.md").write_text("# Intro")
        (tmp_path / "sql").mkdir()
        (tmp_path / "sql" / "create.md").write_text("# Create")
        source = WebCrawlSource(tmp_path)
        files = list(source.walk_markdown())
        assert len(files) == 2

    def test_get_release_info(self, tmp_path: Path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "page.md").write_text("# Page")
        source = WebCrawlSource(tmp_path)
        info = source.get_release_info()
        assert "crawl-" in info["branch"]
        assert info["sha"] == "unknown"

    def test_missing_mirror_raises(self, tmp_path: Path):
        source = WebCrawlSource(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="Mirror root not found"):
            list(source.walk_markdown())


# ---------------------------------------------------------------------------
# GitSource
# ---------------------------------------------------------------------------

class TestGitSource:
    def test_walk_markdown_missing_md_dir(self, tmp_path: Path):
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        with pytest.raises(FileNotFoundError, match="No 'markdown/' directory"):
            list(source.walk_markdown())

    def test_walk_markdown_finds_files(self, tmp_path: Path):
        md_dir = tmp_path / "markdown" / "pub"
        md_dir.mkdir(parents=True)
        (md_dir / "doc.md").write_text("# Doc")
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        files = list(source.walk_markdown())
        assert len(files) == 1
        assert files[0].parent.name == "pub"

    def test_get_metadata(self, tmp_path: Path):
        md_dir = tmp_path / "markdown" / "pub"
        md_dir.mkdir(parents=True)
        f = md_dir / "doc.md"
        f.write_text("# Doc")
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        meta = source.get_metadata(f)
        assert meta["publication"] == "pub"
        assert meta["file"] == "doc.md"
        assert meta["repo_url"] == "https://example.com/repo.git"
        assert meta["branch"] == "main"

    def test_get_metadata_nested(self, tmp_path: Path):
        md_dir = tmp_path / "markdown" / "pub" / "sub"
        md_dir.mkdir(parents=True)
        f = md_dir / "deep.md"
        f.write_text("# Deep")
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        meta = source.get_metadata(f)
        assert meta["publication"] == "pub"
        assert meta["file"] == "sub/deep.md"

    def test_get_release_info_unknown_sha(self, tmp_path: Path):
        """Without a git repo, SHA should be 'unknown'."""
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        info = source.get_release_info()
        assert info["repo_url"] == "https://example.com/repo.git"
        assert info["branch"] == "main"
        assert info["sha"] == "unknown"
        assert info["file_count"] == 0


# ---------------------------------------------------------------------------
# Round-trip: walk -> get_metadata (all adapters)
# ---------------------------------------------------------------------------

class TestWalkMetadataRoundtrip:
    def test_local_source(self, tmp_path: Path):
        (tmp_path / "pub1").mkdir()
        (tmp_path / "pub1" / "a.md").write_text("# A")
        (tmp_path / "pub1" / "b.md").write_text("# B")
        source = LocalSource(tmp_path)
        for f in source.walk_markdown():
            meta = source.get_metadata(f)
            assert meta["publication"] == "pub1"
            assert meta["file"] in ("a.md", "b.md")

    def test_web_crawl_source(self, tmp_path: Path):
        (tmp_path / "ref").mkdir()
        (tmp_path / "ref" / "doc.md").write_text("# Doc")
        source = WebCrawlSource(tmp_path)
        for f in source.walk_markdown():
            meta = source.get_metadata(f)
            assert meta["publication"] == "ref"

    def test_git_source(self, tmp_path: Path):
        md = tmp_path / "markdown" / "sql"
        md.mkdir(parents=True)
        (md / "select.md").write_text("# SELECT")
        source = GitSource(tmp_path, "https://example.com/repo.git", "main")
        for f in source.walk_markdown():
            meta = source.get_metadata(f)
            assert meta["publication"] == "sql"
