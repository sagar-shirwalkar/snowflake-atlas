# Snowflake Docs Page Patterns

Analysis of common page structures, frontmatter schemas, and URL patterns in Snowflake documentation.

## URL Patterns

### Section Index (llms.txt)
```
https://docs.snowflake.com/en/{section-path}/llms.txt
```

Examples:
- `https://docs.snowflake.com/en/user-guide/llms.txt`
- `https://docs.snowflake.com/en/sql-reference/functions/llms.txt`
- `https://docs.snowflake.com/en/developer-guide/snowpark/llms.txt`

### Individual Pages
```
https://docs.snowflake.com/en/{section-path}/{page-name}.md
```

Examples:
- `https://docs.snowflake.com/en/sql-reference/functions/abs.md`
- `https://docs.snowflake.com/en/user-guide/data-loading.md`
- `https://docs.snowflake.com/en/developer-guide/snowpark/python-api.md`

### Special Pages
- Root index: `https://docs.snowflake.com/llms.txt` (no `/en/`)
- Landing pages: `https://docs.snowflake.com/en/reference.md`

---

## Frontmatter Schemas

### Full Frontmatter (Common in User Guide, Developer Guide)
```yaml
---
title: "Loading Data into Snowflake"
description: "Learn how to load data using COPY INTO, Snowpipe, and stages."
product_area: "Data Loading"
last_updated: "2026-01-15"
canonical_url: "https://docs.snowflake.com/en/user-guide/data-loading.md"
---
```

### Minimal Frontmatter (Common in SQL Reference)
```yaml
---
title: "ABS"
---
```

### No Frontmatter (Some older pages)
Pages without `---` delimiters — treated as having empty frontmatter.

### Function-Specific Frontmatter (SQL Functions)
```yaml
---
title: "AI_COMPLETE"
description: "Generates a completion using a Cortex LLM function."
category: "Cortex AI Functions"
syntax: "AI_COMPLETE( <model>, <prompt> [, <options> ] )"
arguments:
  - name: "model"
    type: "STRING"
    required: true
  - name: "prompt"
    type: "STRING"
    required: true
returns: "STRING"
---
```

---

## Content Structure Patterns

### Standard Documentation Page
```markdown
---
title: "Page Title"
description: "Brief description"
product_area: "Product Area"
last_updated: "YYYY-MM-DD"
---

# Page Title

Introductory paragraph.

## Section 1

Content with code examples.

```sql
SELECT * FROM table;
```

## Section 2

More content.

### Subsection

Details.
```

### SQL Function Reference Page
```markdown
---
title: "FUNCTION_NAME"
category: "Category Name"
---

# FUNCTION_NAME

Description of what the function does.

## Syntax

```sql
FUNCTION_NAME( <arg1> [, <arg2> ... ] )
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| arg1 | STRING | Yes | Description |
| arg2 | INTEGER | No | Description |

## Returns

Return type and description.

## Examples

```sql
SELECT FUNCTION_NAME('value');
```

## Notes

Additional notes, limitations, version info.
```

### Tutorial/Guide Page
```markdown
---
title: "Tutorial Title"
description: "Step-by-step guide"
product_area: "Product Area"
last_updated: "YYYY-MM-DD"
---

# Tutorial Title

## Prerequisites

- Requirement 1
- Requirement 2

## Step 1: Do Something

Instructions with code.

## Step 2: Do Something Else

More instructions.

## Clean Up

How to clean up resources.
```

---

## Chunking Considerations for Atlas

### H2-Boundary Works Well
- Most pages use `## ` for major sections
- `### ` for subsections (not split by default chunker)
- Code blocks are fenced with ``` — detected by `is_code` flag

### Frontmatter Fields Available for Metadata
| Field | Availability | Use in Atlas |
|-------|--------------|--------------|
| `title` | ~95% | Chunk title, search boost |
| `description` | ~80% | Summary, embedding context |
| `product_area` | ~60% | Filter by product area |
| `last_updated` | ~50% | Freshness sorting |
| `canonical_url` | ~40% | Source citation |
| `category` | SQL functions only | Filter by function category |

### Special Handling Needed

1. **SQL Function pages** — Very regular structure; could use custom chunker that extracts syntax, arguments, returns as structured metadata
2. **Release Notes** — Chronological entries; each version = natural chunk boundary
3. **Migration guides** — Step-by-step; each step = chunk
4. **API Reference** — Class/method listings; each class = chunk

---

## Page Naming Conventions

| Pattern | Example | Meaning |
|---------|---------|---------|
| `kebab-case.md` | `data-loading.md` | Standard |
| `verb-noun.md` | `create-stage.md` | Action-oriented |
| `noun-verb.md` | `stage-create.md` | Less common |
| `acronym.md` | `udf.md` | Short names |
| `feature-area.md` | `snowpipe-streaming.md` | Feature-specific |

---

## Link Patterns

### Internal Links
- Relative: `[Link](../other-page.md)`
- Absolute: `[Link](https://docs.snowflake.com/en/section/page.md)`
- Anchor: `[Link](#section-name)`

### External Links
- GitHub: `https://github.com/snowflakedb/...`
- Community: `https://community.snowflake.com/...`
- Blogs: `https://www.snowflake.com/blog/...`

---

## Content Volume Estimates

| Section | Pages | Avg Size | Est. Chunks |
|---------|-------|----------|-------------|
| SQL Functions | 994 | ~3 KB | ~3,000 |
| SQL Commands | 676 | ~4 KB | ~2,500 |
| User Guide | 881 | ~8 KB | ~7,000 |
| Loading Data | 682 | ~6 KB | ~4,000 |
| Cortex AI | 102 | ~10 KB | ~1,000 |
| Developer Guide | 317 | ~7 KB | ~2,000 |
| Migrations | 742 | ~12 KB | ~9,000 |
| Release Notes | 1,666 | ~5 KB | ~8,000 |
| **Total** | **~6,800** | **~7 KB** | **~36,000** |

*Estimates based on H2-boundary chunking with ~1,500 chars/chunk average*