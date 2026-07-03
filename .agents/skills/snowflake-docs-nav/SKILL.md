---
name: snowflake-docs-nav
description: Navigate and understand Snowflake's LLM-friendly documentation structure. Use when the user asks about Snowflake docs organization, needs to find specific pages, wants to understand the llms.txt hierarchy, or plans to build an Atlas for Snowflake docs. Branches: explore-structure (map the doc hierarchy), find-pages (locate pages for a topic), plan-atlas-build (prepare config for atlas-creation), fetch-content (retrieve specific page content).
disable-model-invocation: false
---

Snowflake's documentation is published as **LLM-friendly markdown** at `https://docs.snowflake.com/` with a hierarchical `llms.txt` index system.

## Structure Overview

```
https://docs.snowflake.com/llms.txt                          # Root index (30 sections)
    ├── https://docs.snowflake.com/en/user-guide/llms.txt    # User Guide (881 pages)
    ├── https://docs.snowflake.com/en/sql-reference/functions/llms.txt  # SQL Functions (994 pages)
    ├── https://docs.snowflake.com/en/sql-reference/sql/llms.txt       # SQL Commands (676 pages)
    ├── https://docs.snowflake.com/en/user-guide/data-integration/llms.txt  # Loading Data (682 pages)
    ├── https://docs.snowflake.com/en/user-guide/snowflake-cortex/llms.txt  # Cortex AI/ML (102 pages)
    ├── https://docs.snowflake.com/en/developer-guide/snowpark/llms.txt  # Snowpark (67 pages)
    ├── https://docs.snowflake.com/en/developer-guide/snowflake-ml/llms.txt  # Snowflake ML (84 pages)
    ├── https://docs.snowflake.com/en/developer-guide/native-apps/llms.txt  # Native Apps (143 pages)
    ├── https://docs.snowflake.com/en/developer-guide/streamlit/llms.txt  # Streamlit (34 pages)
    ├── https://docs.snowflake.com/en/developer-guide/snowflake-cli/llms.txt  # CLI (244 pages)
    ├── https://docs.snowflake.com/en/migrations/llms.txt    # Migrations (742 pages)
    ├── https://docs.snowflake.com/en/release-notes/llms.txt # Release Notes (1666 pages)
    └── ... (20+ more section llms.txt files)
```

Each section `llms.txt` lists individual `.md` pages (e.g., `https://docs.snowflake.com/en/sql-reference/functions/abs.md`).

**All content is `.md` files** — no `.txt` files referencing `.md`. The user's concern about mixed formats doesn't apply to Snowflake's current publication.

---

## Key Conventions

| Aspect | Convention |
|--------|------------|
| **Base URL** | `https://docs.snowflake.com/en/` |
| **Section index** | `/{section}/llms.txt` |
| **Page URL** | `/{section}/{page}.md` |
| **Language** | `/en/` prefix (only English currently) |
| **Page naming** | kebab-case, descriptive titles |
| **Frontmatter** | YAML with `title`, `description`, sometimes `product_area`, `last_updated` |

---

## Branches

### explore-structure
Map the full doc hierarchy. Fetches root llms.txt, then each section llms.txt. Outputs a tree.

### find-pages
Given a topic (e.g., "Cortex AI functions", "Dynamic Tables", "Iceberg"), search section llms.txt files and return matching page URLs.

### plan-atlas-build
Generate the config values needed by `atlas-creation` skill:
- `REPO_URL` — not a git repo; docs are web-only. Use `atlas-download` pattern instead.
- `DOC_URLS` — list of all `.md` URLs to fetch
- `CHUNKING_STRATEGY` — H2-boundary works; frontmatter is minimal
- `BUNDLE_OUTPUT` — `./data/snowflake-rag-bundle`

### fetch-content
Retrieve a specific page's markdown content. Handles redirects, rate limits, and extracts frontmatter + body.

---

## Usage with atlas-creation

Snowflake docs are **web-only** (no public git repo). Adapt the atlas-creation pipeline:

1. **Replace `atlas-build` repo clone** with a crawler that:
   - Fetches root `llms.txt`
   - Recursively fetches all section `llms.txt`
   - Downloads each `.md` page
   - Saves to local `markdown/` mirror with same folder structure

2. **Run standard `atlas-build`** on the local mirror

3. **Distribute bundle** via `atlas-download` (GitHub Releases)

The `snowflake-docs-nav` skill provides the crawling logic; `atlas-creation` handles chunking, embedding, bundling.

---

## Quick Reference

```python
# Core URLs
ROOT_LLMSTXT = "https://docs.snowflake.com/llms.txt"
BASE_URL = "https://docs.snowflake.com/en/"

# Major sections (from root llms.txt)
SECTIONS = {
    "general": "reference.md",
    "user-guide": "user-guide/llms.txt",
    "loading-data": "user-guide/data-integration/llms.txt",
    "cortex-ai": "user-guide/snowflake-cortex/llms.txt",
    "sql-functions": "sql-reference/functions/llms.txt",
    "sql-commands": "sql-reference/sql/llms.txt",
    "account-usage": "sql-reference/account-usage/llms.txt",
    "org-usage": "sql-reference/organization-usage/llms.txt",
    "info-schema": "sql-reference/info-schema/llms.txt",
    "sql-classes": "sql-reference/classes/llms.txt",
    "sql-general": "sql-reference/llms.txt",
    "connectors": "connectors/llms.txt",
    "collaboration": "collaboration/llms.txt",
    "migrations": "migrations/llms.txt",
    "release-notes": "release-notes/llms.txt",
    "developer-guide": "developer-guide/llms.txt",
    "snowpark": "developer-guide/snowpark/llms.txt",
    "snowflake-ml": "developer-guide/snowflake-ml/llms.txt",
    "native-apps": "developer-guide/native-apps/llms.txt",
    "streamlit": "developer-guide/streamlit/llms.txt",
    "snowflake-cli": "developer-guide/snowflake-cli/llms.txt",
    "snowpark-containers": "developer-guide/snowpark-container-services/llms.txt",
    "rest-api": "developer-guide/snowflake-rest-api/llms.txt",
    "programmatic-access": "progaccess/llms.txt",
}
```

---

## Reference Files (in `references/`)

- `section-map.md` — Full section-to-llms.txt mapping with page counts
- `crawler.py` — Python script to mirror all docs locally
- `page-patterns.md` — Common page URL patterns and frontmatter schemas
- `atlas-config.md` — Recommended config values for atlas-creation