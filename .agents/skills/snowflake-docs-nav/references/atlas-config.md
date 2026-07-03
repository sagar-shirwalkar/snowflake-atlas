# Atlas Config for Snowflake Docs

Recommended configuration values for using the `atlas-creation` skill with Snowflake documentation.

## Crawler → Atlas Pipeline

Since Snowflake docs are web-only (no public git repo), the pipeline is:

```
1. Crawler (snowflake-docs-nav)     → downloads all .md to local mirror
2. atlas-build (atlas-creation)     → chunks, embeds, creates bundle
3. atlas-download (atlas-creation)  → users download bundle
```

## Config Values

```python
# In your project's config or passed as CLI args

# Source (after crawling)
REPO_LOCAL_PATH = "./data/snowflake-docs/snowflake-docs-main"
# Structure after crawler:
# data/snowflake-docs/snowflake-docs-main/
#   └── markdown/
#       ├── user-guide/
#       ├── sql-reference/
#       ├── developer-guide/
#       └── ... (30+ top-level sections)

# Build output
BUNDLE_OUTPUT = "./data/snowflake-rag-bundle"

# Embedding model (BGE-base is good general purpose)
DEFAULT_MODEL_ID = "Xenova/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# Chunking (atlas-creation defaults work well)
# H2-boundary with frontmatter parsing
# Max 8000 chars per chunk

# Backend preference (auto-detects)
# Apple Silicon: MLX
# NVIDIA Linux: ONNX-CUDA
# Else: ONNX-CPU
PREFER_BACKEND = "auto"
```

## CLI Commands

### 1. Crawl Docs (one-time, maintainer)
```bash
# Full crawl (~6,800 pages, ~30 min)
python -m snowflake_docs_nav.crawler \
    --output ./data/snowflake-docs/snowflake-docs-main \
    --rate-limit 0.5

# Incremental (only changed pages)
python -m snowflake_docs_nav.crawler \
    --output ./data/snowflake-docs/snowflake-docs-main \
    --incremental
```

### 2. Build Bundle (maintainer)
```bash
# Full build with embeddings (~2-4 hours depending on hardware)
atlas-build \
    --repo-path ./data/snowflake-docs/snowflake-docs-main \
    --output ./data/snowflake-rag-bundle \
    --prefer auto

# Test build (skip embeddings, limit files)
atlas-build \
    --repo-path ./data/snowflake-docs/snowflake-docs-main \
    --output ./data/test-bundle \
    --limit 100 \
    --skip-embed
```

### 3. Test Locally
```bash
# Smoke test both servers
atlas-smoke \
    --bundle ./data/snowflake-rag-bundle \
    --repo ./data/snowflake-docs/snowflake-docs-main
```

### 4. Publish Bundle (maintainer)
```bash
# Create archive
tar --zstd -cf snowflake-bundle.tar.zst -C ./data/snowflake-rag-bundle .

# Upload to GitHub Releases
gh release create v2026.07 snowflake-bundle.tar.zst \
    --title "Snowflake Atlas Bundle v2026.07" \
    --notes "Built from docs.snowflake.com as of 2026-07-03"
```

### 5. User Install
```bash
# Install package
uv sync --extra mlx  # or --extra gpu / default

# Download bundle
atlas-download \
    --repo your-org/snowflake-atlas-bundles \
    --output ./data/snowflake-rag-bundle

# Use with any MCP client
# snowflake-fs --repo ./data/snowflake-docs/snowflake-docs-main
# snowflake-rag --bundle ./data/snowflake-rag-bundle
```

## Bundle Manifest Expectations

After build, `manifest.json` should contain:

```json
{
  "schema_version": 1,
  "source_repo": "https://docs.snowflake.com/ (web crawl)",
  "source_branch": "crawl-2026-07-03",
  "source_sha": "<git-sha-of-crawler-repo-or-timestamp>",
  "source_published": "2026-07-03T00:00:00Z",
  "built_at": "2026-07-03T14:22:10.123456+00:00",
  "chunk_count": 36000,
  "embedding_model": "Xenova/bge-base-en-v1.5",
  "embedding_dim": 768,
  "embedding_backend": "mlx",
  "embedding_active_provider": "mlx",
  "artifacts": {
    "chunks": "chunks.parquet",
    "chunks_sha256": "...",
    "embeddings": "embeddings.f16.npy",
    "embeddings_sha256": "...",
    "norms": "norms.f32.npy",
    "norms_sha256": "...",
    "model_dir": "model/"
  }
}
```

## Chunking Considerations for Snowflake Docs

| Doc Type | Frontmatter Quality | Chunking Notes |
|----------|---------------------|----------------|
| SQL Functions | Good (title, description) | Many short pages → many small chunks |
| SQL Commands | Good | Medium chunks |
| User Guide | Variable | Some pages lack frontmatter → "Overview" heading |
| Tutorials | Good | Longer pages → more chunks |
| Release Notes | Minimal | Many pages, thin content → consider filtering |
| Openflow (Data Integration) | Good | Technical reference, code-heavy |

### Recommended Filters

```bash
# Exclude release notes (too many, low value for RAG)
# Exclude migrations (very specific, less reusable)
atlas-build \
    --repo-path ./data/snowflake-docs/snowflake-docs-main \
    --output ./data/snowflake-rag-bundle \
    --exclude-patterns "release-notes/**,migrations/**"
```

## Search Optimization

### Title Boost (already in snowflake-rag)
- +0.05 per query token matching document title
- Helps SQL function/command pages rank higher

### Product Area Filtering
```python
# In snowflake-rag, chunks have product_area from frontmatter
# Use filters:
search_docs(query="vector search", product_area="Snowflake Cortex")
search_docs(query="copy into", product_area="Data Loading")
```

### Code Search
```python
# is_code flag set when chunk contains fenced code block
search_code(query="CREATE STAGE", top_k=10)
```

## Size Estimates

| Metric | Estimate |
|--------|----------|
| Markdown files | ~6,800 |
| Total raw markdown | ~500 MB |
| Chunks (H2-boundary) | ~36,000 |
| embeddings.f16.npy | ~210 MB |
| chunks.parquet (snappy) | ~80 MB |
| norms.f32.npy | ~0.14 MB |
| model/ (ONNX) | ~450 MB |
| **Total bundle** | **~740 MB** |

## Update Cadence

- **Monthly**: Snowflake releases new features monthly
- **Quarterly**: Major releases (behavior changes)
- **Recommended**: Rebuild bundle monthly via scheduled workflow

```yaml
# .github/workflows/update-bundle.yml
on:
  schedule:
    - cron: '0 2 1 * *'  # Monthly on 1st at 2 AM
  workflow_dispatch:

jobs:
  rebuild:
    runs-on: macos-latest  # For MLX speed
    steps:
      - uses: actions/checkout@v4
      - name: Crawl latest docs
        run: python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs --incremental
      - name: Build bundle
        run: atlas-build --repo-path ./data/snowflake-docs --output ./data/bundle --prefer auto
      - name: Publish
        run: scripts/publish-bundle.sh
```

## Integration with atlas-creation Skill

The `snowflake-docs-nav` skill provides:
1. **Crawler** → creates the `markdown/` structure `atlas-build` expects
2. **Section map** → knows which sections to include/exclude
3. **Page patterns** → understands frontmatter variability
4. **Config template** → drop-in values for `atlas-build` args

The `atlas-creation` skill remains **unchanged** — it works on any local markdown repo with the expected structure.