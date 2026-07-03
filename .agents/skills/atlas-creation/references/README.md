# Atlas Creation Skill — Reference Index

This directory contains the reference documentation for the **atlas-creation** skill, which guides building a local-first AI knowledge layer (Atlas) for any markdown documentation corpus.

## Quick Navigation

| Document | Purpose |
|----------|---------|
| [`SKILL.md`](SKILL.md) | Main skill — step-by-step build guide with completion criteria |
| [`references/architecture.md`](references/architecture.md) | Component diagram, data flow, process model, design decisions |
| [`references/bundle-format.md`](references/bundle-format.md) | Manifest schema, parquet columns, numpy layouts, integrity |
| [`references/backend-selection.md`](references/backend-selection.md) | Resolution algorithm, backend comparison, overrides, CI |
| [`references/chunking-strategy.md`](references/chunking-strategy.md) | H2-boundary algorithm, edge cases, parquet mapping |
| [`references/testing.md`](references/testing.md) | Test fixtures, unit/integration tests, smoke test, eval |

## Skill Usage

```bash
# This skill is model-invoked — the agent will reach for it when you say:
# "build an Atlas for X docs", "create a knowledge layer for Y", "set up dual MCP servers"
```

## For a New Documentation Corpus

1. **Read** `SKILL.md` — understand the 10 phases
2. **Reference** the architecture and bundle format docs as you implement
3. **Configure** a platform-specific skill (e.g., `snowflake-atlas-config`) with:
   - `REPO_URL`, `BRANCH`, `DEFAULT_MODEL_ID`
   - `BUNDLE_OUTPUT`, `REPO_LOCAL_PATH`
4. **Run** the build: `atlas-build --repo-url ... --branch ... --output ...`
5. **Test**: `atlas-smoke --bundle ... --repo ...`
6. **Publish**: `scripts/publish-bundle.sh`
7. **Users install**: `uv sync && atlas-download --repo Org/Bundles --output ...`

## Key Leading Words

| Word | Meaning |
|------|---------|
| **dual-server** | FS for exact citations + RAG for fuzzy discovery |
| **bundle** | Portable artifact (chunks + embeddings + model + manifest) |
| **pinned** | Source repo + branch + SHA recorded for reproducibility |
| **backend** | MLX / ONNX-CUDA / ONNX-CPU — resolved at run time |
| **H2-chunk** | Split on `## ` headers, preserve frontmatter |
| **doctor** | Diagnostic probe showing platform + selected backend |

## Directory Structure

```
atlas-creation/
├── SKILL.md                      # Main skill (this file's parent)
├── references/
│   ├── architecture.md           # System diagram, data flow, decisions
│   ├── bundle-format.md          # Manifest, parquet, numpy, SHA256
│   ├── backend-selection.md      # Resolution logic, fallback, overrides
│   ├── chunking-strategy.md      # H2-boundary, code flag, hard split
│   └── testing.md                # Fixtures, unit/integration/smoke/eval
```

## Extending for a New Platform

Create a **separate skill** (e.g., `snowflake-atlas-config`) that provides:

```python
# In the skill's reference or as defaults
REPO_URL = "https://github.com/Snowflake-Labs/snowflake-docs.git"
BRANCH = "main"
DEFAULT_MODEL_ID = "Xenova/bge-base-en-v1.5"
BUNDLE_OUTPUT = "./data/rag-bundle"
REPO_LOCAL_PATH = "./data/snowflake-docs/snowflake-docs-main"
```

The atlas-creation skill remains **platform-agnostic** — all corpus-specific values are injected via CLI args or config.