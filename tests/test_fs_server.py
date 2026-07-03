"""Tests for the filesystem MCP server (atlas/fs_server.py)."""

import json
import tempfile
from pathlib import Path

import pytest

from atlas.fs_server import (
    _parse_md,
    _preview_line,
    create_source,
    full_text_search,
    list_publication_files,
    list_publications,
    read_publication_file,
)
from atlas.sources import LocalSource


# ---------------------------------------------------------------------------
# create_source factory
# ---------------------------------------------------------------------------

class TestCreateSource:
    def test_git_source_requires_all_args(self):
        with pytest.raises(ValueError, match="requires --repo-path"):
            create_source("git")

    def test_web_crawl_source_requires_mirror(self):
        with pytest.raises(ValueError, match="requires --mirror-path"):
            create_source("web-crawl")

    def test_local_source_requires_mirror(self):
        with pytest.raises(ValueError, match="requires --mirror-path"):
            create_source("local")

    def test_unknown_source_type(self):
        with pytest.raises(ValueError, match="Unknown source type"):
            create_source("unknown")

    def test_local_source_created(self, tmp_path: Path):
        source = create_source("local", mirror_path=str(tmp_path))
        assert isinstance(source, LocalSource)
        assert source.mirror_root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _parse_md
# ---------------------------------------------------------------------------

class TestParseMd:
    def test_with_frontmatter(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\nproduct_area: Core\n---\nBody text")
        result = _parse_md(f)
        assert result["frontmatter"]["title"] == "Test"
        assert result["frontmatter"]["product_area"] == "Core"
        assert "Body text" in result["content"]

    def test_no_frontmatter(self, tmp_path: Path):
        f = tmp_path / "plain.md"
        f.write_text("Just body content")
        result = _parse_md(f)
        assert result["frontmatter"] == {}
        assert result["content"] == "Just body content"

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.md"
        f.write_text("")
        result = _parse_md(f)
        assert result["frontmatter"] == {}
        assert result["content"] == ""

    def test_malformed_yaml(self, tmp_path: Path):
        f = tmp_path / "bad.md"
        f.write_text("---\nnot: valid: yaml: [\n---\nBody")
        result = _parse_md(f)
        assert result["frontmatter"] == {}
        assert "Body" in result["content"]

    def test_incomplete_frontmatter(self, tmp_path: Path):
        f = tmp_path / "partial.md"
        f.write_text("---\ntitle: Only meta")
        result = _parse_md(f)
        # No closing --- means no frontmatter detected
        assert result["frontmatter"] == {}
        assert "---" in result["content"]


# ---------------------------------------------------------------------------
# list_publications
# ---------------------------------------------------------------------------

def _make_source(tmp_path: Path, subdirs: list[str]) -> LocalSource:
    for s in subdirs:
        d = tmp_path / s
        d.mkdir(parents=True)
        (d / "page.md").write_text("# Page")
    return LocalSource(tmp_path)


class TestListPublications:
    def test_single_publication(self, tmp_path: Path):
        source = _make_source(tmp_path, ["user-guide"])
        pubs = list_publications(source)
        assert len(pubs) == 1
        assert pubs[0]["name"] == "user-guide"
        assert pubs[0]["file_count"] == 1

    def test_multiple_publications(self, tmp_path: Path):
        source = _make_source(tmp_path, ["ug", "sql-ref", "dev-guide"])
        pubs = list_publications(source)
        assert len(pubs) == 3
        names = [p["name"] for p in pubs]
        assert "ug" in names
        assert "sql-ref" in names
        assert "dev-guide" in names
        assert all(p["file_count"] == 1 for p in pubs)

    def test_empty_source(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        pubs = list_publications(source)
        assert pubs == []


# ---------------------------------------------------------------------------
# list_publication_files
# ---------------------------------------------------------------------------

class TestListPublicationFiles:
    def test_basic(self, tmp_path: Path):
        pub_dir = tmp_path / "user-guide"
        pub_dir.mkdir()
        (pub_dir / "intro.md").write_text("---\ntitle: Intro\n---\nContent")
        (pub_dir / "advanced.md").write_text("---\ntitle: Advanced\nproduct_area: Pro\n---\nContent")
        source = LocalSource(tmp_path)
        files = list_publication_files(source, "user-guide")
        assert len(files) == 2
        titles = {f["file"]: f["title"] for f in files}
        assert titles["intro.md"] == "Intro"
        assert titles["advanced.md"] == "Advanced"

    def test_publication_not_found(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            list_publication_files(source, "nonexistent")

    def test_skips_non_md_files(self, tmp_path: Path):
        pub_dir = tmp_path / "docs"
        pub_dir.mkdir()
        (pub_dir / "page.md").write_text("---\ntitle: Page\n---\nC")
        (pub_dir / "image.png").write_bytes(b"\x89PNG")
        source = LocalSource(tmp_path)
        files = list_publication_files(source, "docs")
        assert len(files) == 1


# ---------------------------------------------------------------------------
# read_publication_file
# ---------------------------------------------------------------------------

class TestReadPublicationFile:
    @pytest.fixture
    def source(self, tmp_path: Path):
        pub_dir = tmp_path / "test-pub"
        pub_dir.mkdir()
        (pub_dir / "page.md").write_text(
            "---\ntitle: Test\nproduct_area: Core\nlast_updated: 2025-01-01\n---\nBody content here."
        )
        return LocalSource(tmp_path)

    def test_read_file(self, source, tmp_path: Path):
        result = read_publication_file(source, "test-pub", "page.md")
        assert result["publication"] == "test-pub"
        assert result["file"] == "page.md"
        assert result["frontmatter"]["title"] == "Test"
        assert "Body content here" in result["content"]
        assert result["truncated"] is False

    def test_path_traversal_blocked(self, source, tmp_path: Path):
        with pytest.raises(ValueError, match="Path traversal blocked"):
            read_publication_file(source, "test-pub", "../../etc/passwd")

    def test_file_not_found(self, source, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_publication_file(source, "test-pub", "nonexistent.md")

    def test_truncation(self, source, tmp_path: Path):
        result = read_publication_file(source, "test-pub", "page.md", max_chars=5)
        assert result["truncated"] is True
        assert len(result["content"]) <= 5

    def test_no_truncation_needed(self, source, tmp_path: Path):
        result = read_publication_file(source, "test-pub", "page.md", max_chars=1_000_000)
        assert result["truncated"] is False


# ---------------------------------------------------------------------------
# full_text_search
# ---------------------------------------------------------------------------

class TestFullTextSearch:
    @pytest.fixture
    def source(self, tmp_path: Path):
        pub_dir = tmp_path / "docs"
        pub_dir.mkdir()
        (pub_dir / "intro.md").write_text("# Intro\n\nWelcome to the documentation.\n\n## Setup\n\nRun `install` to get started.")
        (pub_dir / "advanced.md").write_text("# Advanced\n\nThis covers advanced topics.\n\n## Config\n\nSet `DEBUG=false` in the config file.")
        return LocalSource(tmp_path)

    def test_search_basic(self, source, tmp_path: Path):
        """Basic search should return matching files."""
        results = full_text_search(source, "advanced")
        assert len(results) >= 1
        file_names = [r["file"] for r in results]
        assert any("advanced.md" in f for f in file_names)

    def test_search_regex(self, source, tmp_path: Path):
        results = full_text_search(source, r"install|DEBUG", regex=True)
        assert len(results) >= 1

    def test_search_nonexistent_query(self, source, tmp_path: Path):
        results = full_text_search(source, "xyznonexistent12345")
        assert results == []

    def test_search_scope(self, source, tmp_path: Path):
        results = full_text_search(source, "install", scope="docs")
        file_names = [r["file"] for r in results]
        assert any("intro.md" in f for f in file_names)

    def test_search_scope_not_found(self, source, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            full_text_search(source, "test", scope="nonexistent")

    def test_search_respects_max_results(self, source, tmp_path: Path):
        results = full_text_search(source, "the", max_results=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# _preview_line
# ---------------------------------------------------------------------------

class TestPreviewLine:
    def test_basic(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        preview = _preview_line(f, 3)
        assert "line2" in preview
        assert "line3" in preview
        assert "line4" in preview

    def test_first_line(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("first\nsecond\nthird\n")
        preview = _preview_line(f, 1)
        assert "first" in preview
        assert "second" in preview

    def test_last_line(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("first\nsecond\nthird\n")
        preview = _preview_line(f, 3)
        assert "second" in preview
        assert "third" in preview

    def test_file_not_found(self):
        preview = _preview_line(Path("/nonexistent/file.md"), 1)
        assert preview == ""


# ---------------------------------------------------------------------------
# get_release_info (via LocalSource)
# ---------------------------------------------------------------------------

def test_get_release_info_local(tmp_path: Path):
    source = LocalSource(tmp_path, repo_url="local://test", branch="local")
    info = source.get_release_info()
    assert info["branch"] == "local"
    assert info["repo_url"] == "local://test"
    assert info["file_count"] == 0
