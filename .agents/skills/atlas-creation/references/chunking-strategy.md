# Chunking Strategy Reference

## Overview

Atlas uses **H2-boundary chunking** — each `## ` (level-2 header) in a markdown file becomes a chunk boundary. This respects the documentation team's deliberate information architecture rather than imposing arbitrary sliding windows.

## Algorithm

```python
def chunk_markdown(text: str, publication: str, file: str) -> list[dict]:
    # 1. Parse YAML frontmatter
    frontmatter, body = parse_frontmatter(text)
    
    # 2. Split on H2 boundaries
    sections = _split_on_h2(body)  # [(heading, section_text), ...]
    
    # 3. For each section, hard-split if > MAX_CHUNK_CHARS
    chunks = []
    for heading, section_text in sections:
        for piece in _hard_split(section_text, MAX_CHUNK_CHARS):
            chunks.append({
                "heading": heading,
                "text": piece,
                "frontmatter": frontmatter,
                "publication": publication,
                "file": file,
                "is_code": _is_code_chunk(piece),
            })
    return chunks
```

## Step-by-Step

### 1. Frontmatter Parsing

```regex
\A---\s*\n(.*?)\n---\s*\n(.*)\Z  (DOTALL)
```

- Extracts YAML between `---` delimiters at file start
- Returns `(metadata_dict, body_text)`
- Malformed frontmatter → empty dict + full text (never fails the build)

**Expected frontmatter fields** (carried into every chunk):
```yaml
title: "Incident Management"
product_area: "IT Service Management"
last_updated: "2026-01-15"
canonical_url: "https://docs.servicenow.com/.../incident-management.html"
```

### 2. H2 Splitting

```regex
^##\s+(.+?)\s*$  (MULTILINE)
```

- Finds all `## ` headers
- First section before first H2 → heading = `"Overview"`
- Each H2 → heading = header text (stripped)
- Section text = content between this H2 and next H2 (or EOF)

**Example**:
```markdown
---
title: "Incident Management"
---

Incident management handles the lifecycle of incidents...

## Creating Incidents

To create an incident...

## Resolving Incidents

To resolve an incident...
```

**Produces**:
```
Chunk 0: heading="Overview", text="Incident management handles..."
Chunk 1: heading="Creating Incidents", text="To create an incident..."
Chunk 2: heading="Resolving Incidents", text="To resolve an incident..."
```

### 3. Code Chunk Detection

```regex
^```  (MULTILINE)
```

- Heuristic: any fenced code block → `is_code = true`
- Line-based, robust against prose interleaved with code
- Used by `search_code` tool to filter results

### 4. Hard Split (Fallback)

```python
def _hard_split(text: str, max_chars: int = 8000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    
    parts = []
    while text:
        if len(text) <= max_chars:
            parts.append(text)
            break
        
        # Try paragraph break
        cut = text.rfind("\n\n", 0, max_chars)
        if cut < max_chars // 2:
            # Try line break
            cut = text.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            # Hard cut
            cut = max_chars
        
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    
    return [p for p in parts if p]
```

- Only triggers for oversized H2 sections (rare)
- Prefers paragraph breaks → line breaks → hard cut
- Each piece becomes a separate chunk with same heading

## Output Chunk Schema

```python
{
    "id": "publication/file#chunk_idx",     # Generated in make_bundle.py
    "heading": "Creating Incidents",        # H2 text or "Overview"
    "text": "To create an incident...",     # Chunk body
    "frontmatter": {...},                   # Full YAML frontmatter
    "publication": "it-service-management", # Top-level folder
    "file": "incident-management.md",       # Relative to markdown/
    "is_code": False,                       # Has fenced code block?
    "title": "Incident Management",         # frontmatter.title
    "product_area": "IT Service Management",# frontmatter.product_area
    "last_updated": "2026-01-15",           # frontmatter.last_updated
    "canonical_url": "https://..."          # frontmatter.canonical_url
}
```

## Why H2-Boundary?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **H2-boundary** | Respects docs structure; semantic coherence; natural size | Uneven chunk sizes | ✓ **Chosen** |
| Fixed window (512 tokens) | Uniform size; predictable | Cuts across semantic boundaries; loses context | ✗ |
| Paragraph-based | Natural breaks | Too granular; loses section context | ✗ |
| Semantic (LLM) | Optimal boundaries | Slow; non-deterministic; model-dependent | ✗ |

**Key insight**: The ServiceNowDocs (and similar) repos are **authored for LLMs** — one topic per file, H2 = sub-topic. The chunk size *is* the author's intent.

## Edge Cases Handled

| Case | Behavior |
|------|----------|
| No frontmatter | Empty dict; chunk still created |
| No H2 headers | Single chunk with heading="Overview" |
| Empty H2 section | Dropped (no empty chunks) |
| H2 section > 8000 chars | Split on paragraph/line boundaries |
| Multiple fenced blocks | `is_code = True` (any fence triggers) |
| Malformed YAML | Logged, treated as no frontmatter |
| Unicode in headings | Preserved (regex uses Unicode-aware `\s`) |
| Nested headers (###) | Ignored — only H2 splits |

## Configuration

```python
# In chunk.py
_MAX_CHUNK_CHARS = 8000  # Hard split threshold
```

- 8000 chars ≈ 2000 tokens (BGE-base max 512, but we embed full chunks)
- Embedder truncates at 512008 tokens (512 - special tokens)
- Oversized chunks are rare (<1% in practice)

## Parquet Column Mapping

In `make_bundle.py → build_chunk_table()`:

| Chunk Dict Key | Parquet Column | Type |
|----------------|----------------|------|
| `id` | `id` | string |
| `text` | `text` | string |
| `publication` | `publication` | string |
| `file` | `file` | string |
| `heading` | `heading` | string |
| `is_code` | `is_code` | bool |
| `frontmatter.title` | `title` | string |
| `frontmatter.product_area` | `product_area` | string |
| `frontmatter.last_updated` | `last_updated` | string |
| `frontmatter.canonical_url` | `canonical_url` | string |

## Testing the Chunker

```bash
# Quick test
python -c "
from atlas.chunk import chunk_file
from pathlib import Path
chunks = chunk_file(Path('test.md'), Path('.'))
for c in chunks:
    print(f'{c[\"heading\"][:40]:40} {len(c[\"text\"]):5} chars  code={c[\"is_code\"]}')
"

# Run test suite
uv run pytest tests/test_chunk.py -v
```

## Extending for Other Corpora

If a docs repo uses different conventions:

```python
# In a platform-specific skill, override chunk_markdown()
def chunk_markdown_custom(text: str, publication: str, file: str) -> list[dict]:
    # e.g., H1-boundary, or custom delimiter
    # Must return same dict schema
    pass

# Then in make_bundle.py, import and use the custom chunker
from my_config.chunker import chunk_markdown_custom
```

The rest of the pipeline (embedding, bundle format, servers) remains unchanged.