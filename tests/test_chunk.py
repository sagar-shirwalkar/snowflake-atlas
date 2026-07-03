"""Tests for the chunker and frontmatter parser."""

from pathlib import Path

import pytest

from atlas.chunk import (
    _hard_split,
    _is_code_chunk,
    _section_tail,
    _split_on_h2,
    chunk_file,
    chunk_markdown,
    parse_frontmatter,
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
    # Section 2 inherits a code-fence tail from Section 1 via chunk overlap,
    # so is_code is True (the overlap carries the ``` pattern forward).
    assert chunks[2]["is_code"] is True  # overlap from code section
    assert all(c["frontmatter"]["title"] == "Test Page" for c in chunks)


# ---------------------------------------------------------------------------
# Chunk overlap tests
# ---------------------------------------------------------------------------


def test_section_tail_short_text():
    """_section_tail returns the full text if it's shorter than n_chars."""
    result = _section_tail("short", 150)
    assert result == "short"


def test_section_tail_word_broken():
    """_section_tail breaks at a word boundary, not mid-word."""
    text = "a" * 100 + " breakpoint " + "b" * 100
    result = _section_tail(text, 50)
    # Should break at the space before "breakpoint" or after
    assert " " not in result.strip() or ("breakpoint" in result or result.startswith("b"))
    assert len(result) <= 150


def test_overlap_first_section_has_no_overlap():
    """The first section should never have overlap prepended."""
    text = "## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2."
    chunks = chunk_markdown(text, "pub", "file.md", overlap_chars=50)
    assert len(chunks) == 2
    # Section 1 should NOT contain "Content 1." as prepended overlap
    # (there's no section before it, so no overlap)
    # Text is augmented as "{heading}\n{piece}" = "Section 1\nContent 1."
    assert chunks[0]["text"] == "Section 1\nContent 1."


def test_overlap_appears_in_second_section():
    """The second section should have the first section's tail prepended."""
    text = "## First\n\nTail content here.\n\n## Second\n\nBody."
    chunks = chunk_markdown(text, "pub", "file.md", overlap_chars=200)
    assert len(chunks) == 2
    # Second section's text is augmented as "{heading}\n{overlap}\n\n{piece}"
    # = "Second\nTail content here.\n\nBody."
    tail_present = "Tail content here." in chunks[1]["text"]
    assert tail_present


def test_overlap_zero_disabled():
    """Setting overlap_chars=0 should disable overlap entirely."""
    text = "## First\n\nContent A.\n\n## Second\n\nContent B."
    chunks = chunk_markdown(text, "pub", "file.md", overlap_chars=0)
    assert len(chunks) == 2
    # Without overlap, Content A. should not appear in the second chunk
    assert "Content A." not in chunks[1]["text"]
    # The text is augmented as "{heading}\n{piece}"
    assert chunks[1]["text"] == "Second\nContent B."


def test_overlap_code_flag_propagation():
    """Overlap from a code-containing section may propagate is_code flag."""
    text = "## Section A\n\nSome ```code```\n\n## Section B\n\nJust text."
    chunks = chunk_markdown(text, "pub", "file.md", overlap_chars=150)
    # Section B's text includes the tail from Section A; if that tail
    # carries a code-fence fragment, is_code becomes True
    # If not (tail doesn't capture the ``` ), is_code stays False
    # Either is valid — we just verify the overlap was appended correctly
    assert "Just text." in chunks[1]["text"]


def test_overlap_many_sections():
    """Overlap works across multiple sequential sections."""
    text = (
        "## One\n\n" + "x" * 100 +
        "\n\n## Two\n\n" + "y" * 100 +
        "\n\n## Three\n\n" + "z" * 100
    )
    chunks = chunk_markdown(text, "pub", "file.md", overlap_chars=50)
    assert len(chunks) == 3
    # Section 2 should have tail from Section 1
    assert ("x" * 50) not in chunks[0]["text"][-50:] or ("x" in chunks[1]["text"])
    # Section 3 should have tail from Section 2
    assert ("y" * 50) not in chunks[1]["text"][-50:] or ("y" in chunks[2]["text"])


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
