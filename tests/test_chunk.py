"""Tests for the chunker and frontmatter parser."""

import tempfile
from pathlib import Path

import pytest

from atlas.chunk import (
    chunk_file,
    chunk_markdown,
    parse_frontmatter,
    _split_on_h2,
    _is_code_chunk,
    _hard_split,
)


def test_parse_frontmatter():
    text = "---\ntitle: Test\nproduct_area: Core\n---\nBody"
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "Test"
    assert meta["product_area"] == "Core"
    assert body == "Body"


def test_no_frontmatter():
    text = "No frontmatter here"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_malformed_frontmatter():
    text = "---\nnot: valid: yaml: [\n---\nBody"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_h2_split():
    body = "Overview\n\n## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2"
    sections = _split_on_h2(body)
    assert len(sections) == 3
    assert sections[0][0] == "Overview"
    assert sections[1][0] == "Section 1"
    assert sections[2][0] == "Section 2"
    assert "Content 1" in sections[1][1]
    assert "Content 2" in sections[2][1]


def test_h2_split_no_h2():
    body = "Just a single section"
    sections = _split_on_h2(body)
    assert len(sections) == 1
    assert sections[0][0] == "Overview"
    assert sections[0][1] == "Just a single section"


def test_code_flag():
    assert _is_code_chunk("Text\n```python\ncode\n```\nMore")
    assert _is_code_chunk("```sql\nSELECT 1;\n```")
    assert not _is_code_chunk("Just text")
    assert not _is_code_chunk("Text with `inline code` only")


def test_hard_split():
    long = "a " * 5000  # ~10000 chars
    parts = _hard_split(long, 8000)
    assert len(parts) == 2
    assert all(len(p) <= 8000 for p in parts)


def test_hard_split_paragraph():
    text = "Para 1\n\nPara 2\n\nPara 3"
    parts = _hard_split(text, 15)
    assert len(parts) >= 2


def test_chunk_markdown():
    text = """---
title: "Test Page"
product_area: "Core"
---
# Test Page

Intro text.

## Section 1

Content with code:

```python
print("hello")
```

## Section 2

More content.
"""
    chunks = chunk_markdown(text, "test-pub", "test.md")
    assert len(chunks) == 3  # Overview + 2 H2 sections
    assert chunks[0]["heading"] == "Overview"
    assert chunks[1]["heading"] == "Section 1"
    assert chunks[2]["heading"] == "Section 2"
    assert chunks[0]["publication"] == "test-pub"
    assert chunks[0]["file"] == "test.md"
    assert chunks[1]["is_code"] is True  # Has fenced code block
    assert chunks[2]["is_code"] is False
    assert all(c["frontmatter"]["title"] == "Test Page" for c in chunks)


def test_chunk_file(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    md_root = repo_root / "markdown" / "test-pub"
    md_root.mkdir(parents=True)
    (md_root / "test.md").write_text("""---
title: "Test Page"
---
# Test Page

## Section 1

Content 1.

## Section 2

Content 2.
""")

    chunks = chunk_file(md_root / "test.md", repo_root)
    assert len(chunks) == 3
    assert chunks[0]["publication"] == "test-pub"
    assert chunks[0]["file"] == "test.md"
    assert chunks[0]["heading"] == "Overview"
    assert chunks[1]["heading"] == "Section 1"
    assert chunks[2]["heading"] == "Section 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
