# Snowflake Docs Nav Skill — Reference Index

## Quick Navigation

| Document | Purpose |
|----------|---------|
| [`SKILL.md`](SKILL.md) | Main skill — branches, conventions, usage with atlas-creation |
| [`references/section-map.md`](references/section-map.md) | All 30 sections with page counts and URLs |
| [`references/crawler.py`](references/crawler.py) | Python script to mirror docs locally |
| [`references/page-patterns.md`](references/page-patterns.md) | URL patterns, frontmatter schemas, chunking notes |
| [`references/atlas-config.md`](references/atlas-config.md) | Config values and CLI commands for atlas-creation integration |

## Skill Usage

```bash
# Model-invoked when you say:
# "explore Snowflake docs structure"
# "find pages about Cortex AI"
# "plan Atlas build for Snowflake"
# "fetch Snowflake doc page"
```

## Quick Reference

```python
# Core URLs
ROOT_LLMSTXT = "https://docs.snowflake.com/llms.txt"
BASE_URL = "https://docs.snowflake.com/en/"

# Crawler usage
python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs

# Atlas build (after crawling)
atlas-build --repo-path ./data/snowflake-docs --output ./data/snowflake-rag-bundle

# User download
atlas-download --repo your-org/bundles --output ./data/snowflake-rag-bundle
```

## Key Findings

- **All content is `.md`** — no `.txt` files referencing `.md` (user concern addressed)
- **~6,800 pages** across 30 sections
- **Frontmatter varies** — 95% have title, 60% have product_area, 50% have last_updated
- **H2-boundary chunking works well** — estimated 36,000 chunks
- **Bundle size** — ~740 MB (embeddings + model + metadata)
- **Web-only source** — requires crawler step before atlas-build

## Directory Structure

```
snowflake-docs-nav/
├── SKILL.md                      # Main skill (this file's parent)
├── references/
│   ├── README.md                 # This index
│   ├── section-map.md            # Section URLs + page counts
│   ├── crawler.py                # Async crawler script
│   ├── page-patterns.md          # Frontmatter, content structure
│   └── atlas-config.md           # Integration config + CLI
```