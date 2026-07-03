---
name: atlas-creation
description: Build a local-first AI knowledge layer (Atlas) for any markdown documentation corpus via two MCP servers (filesystem + RAG). Use when the user wants to create an Atlas system for a new documentation source — e.g., "build an Atlas for Snowflake docs", "create a knowledge layer for Kubernetes docs", "set up dual MCP servers for my markdown repo". Branches: build-atlas (full pipeline), build-fs-server (filesystem only), build-rag-server (RAG only), build-bundle (precompute embeddings), eval-rag (measure retrieval quality).
disable-model-invocation: false
---

An **Atlas** is a local-first AI knowledge layer built from a markdown documentation corpus. It exposes two Model Context Protocol (MCP) servers:

1. **Filesystem server** (`atlas-fs`) — deterministic, zero-infra, backed by `ripgrep`. Tools: list publications, list files, read file, full-text search, get release info. No model, no embeddings, no state. Works with any markdown repo.

2. **RAG server** (`atlas-rag`) — semantic search over precomputed embeddings. Tools: search_docs, search_code, get_chunk, get_bundle_info. Loads a portable bundle once at startup; answers queries via single matrix multiply. Supports MLX (Apple Silicon), ONNX+CUDA (NVIDIA), ONNX+CPU (portable floor).

The bundle is built **once** by the maintainer (`atlas-build`), distributed as a single artifact, and consumed by end users with zero embedding/chunking/model work.

This skill is **platform-agnostic**. Snowflake-specific, Kubernetes-specific, etc. configs live in separate skills.

---

## Core concepts (leading words)

- **dual-server** — FS for verbatim citations, RAG for fuzzy discovery
- **bundle** — portable artifact (chunks.parquet + embeddings.npy + norms.npy + model/ + manifest.json)
- **pinned** — source repo URL, branch, and git SHA recorded for reproducibility
- **backend** — inference runtime resolved at run time: MLX, ONNX-CUDA, ONNX-CPU
- **H2-chunk** — split on `## ` headers, parse YAML frontmatter, flag code chunks
- **doctor** — diagnostic probe (`atlas-doctor`) showing platform, backend probes, selected backend + reason
- **source-adapter** — pluggable interface to fetch markdown from git, web crawl, local dir, or API

---

## 10-Phase Build Process

| Phase | Task | Completion Criterion |
|-------|------|---------------------|
| 0 | Prerequisites | `python3 uv rg git` available; Python 3.11+ |
| 1 | Scaffold project | `pyproject.toml` with entry points + deps; `atlas/__init__.py` |
| 2 | Chunker (`atlas/chunk.py`) | `chunk_file()` returns dicts with 10 required keys; H2-boundary split |
| 3 | Embedding backends (`atlas/embed/`) | 3 backends (MLX, ONNX-CUDA, ONNX-CPU); factory picks best at runtime |
| 4 | FS MCP server (`atlas/fs_server.py`) | 5 tools registered; deterministic; ripgrep-backed; path traversal blocked |
| 5 | RAG MCP server (`atlas/rag_server.py`) | 4 tools registered; bundle loaded once; cosine via matrix multiply; hybrid modes |
| 6 | Bundle builder (`atlas/make_bundle.py`) | End-to-end: fetch → chunk → embed → write artifacts → stage model → manifest w/ SHA256 |
| 7 | Download + verify (`atlas/download.py`) | GitHub Releases download; SHA256 verify; backup existing before overwrite |
| 8 | Utilities | backup, restore, doctor, smoke_test, evaluate, log, rerank — all working standalone |
| 9 | Tests (`tests/`) | Pytest suite passes: chunker, embed backends, FS/RAG servers, bundle build, download |
| 10 | CI + Release | GitHub Actions smoke test; `scripts/publish-bundle.sh` creates release |

---

## Source Adapters (NEW — Generic Input Handling)

The bundle builder (`make_bundle.py`) now accepts a **source adapter** that provides a uniform interface over different documentation sources.

### Adapter Interface

```python
# atlas/sources/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

class MarkdownSource(ABC):
    """Abstract source of markdown files with metadata."""
    
    @abstractmethod
    def walk_markdown(self) -> Iterator[Path]:
        """Yield absolute paths to all .md files."""
    
    @abstractmethod
    def get_metadata(self, path: Path) -> dict:
        """Return source metadata for a file: {'publication': str, 'file': str, 'repo_url': str, 'branch': str, 'sha': str}"""
    
    @abstractmethod
    def get_release_info(self) -> dict:
        """Return {'branch': str, 'sha': str, 'repo_url': str, 'file_count': int}"""
```

### Built-in Adapters

| Adapter | Source Type | Use Case |
|---------|-------------|----------|
| `GitSource` | Git repository | ServiceNow, Kubernetes, self-hosted docs |
| `WebCrawlSource` | Local mirror from web crawl | Snowflake, AWS, Azure docs (no public git) |
| `LocalSource` | Local directory | Ad-hoc collections, private docs |
| `APISource` | REST/GraphQL API | Notion, GitBook, Confluence exports |

### GitSource (Default)

```python
# atlas/sources/git.py
class GitSource(MarkdownSource):
    def __init__(self, repo_path: Path, repo_url: str, branch: str):
        self.repo_path = repo_path
        self.repo_url = repo_url
        self.branch = branch
    
    def walk_markdown(self) -> Iterator[Path]:
        md_root = self.repo_path / "markdown"
        return md_root.rglob("*.md")
    
    def get_metadata(self, path: Path) -> dict:
        rel = path.relative_to(self.repo_path / "markdown")
        parts = rel.parts
        return {
            "publication": parts[0] if parts else "unknown",
            "file": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "repo_url": self.repo_url,
            "branch": self.branch,
            "sha": self._get_sha(),
        }
```

### WebCrawlSource (for Snowflake)

```python
# atlas/sources/web_crawl.py
class WebCrawlSource(MarkdownSource):
    def __init__(self, mirror_root: Path, crawl_meta: dict):
        self.mirror_root = mirror_root
        self.crawl_meta = crawl_meta  # {'source_url': 'https://docs.snowflake.com', 'crawled_at': '...'}
    
    def walk_markdown(self) -> Iterator[Path]:
        return self.mirror_root.rglob("*.md")
    
    def get_metadata(self, path: Path) -> dict:
        rel = path.relative_to(self.mirror_root)
        parts = rel.parts
        return {
            "publication": parts[0] if parts else "unknown",
            "file": "/".join(parts[1:]) if len(parts) > 1 else parts[0],
            "repo_url": self.crawl_meta["source_url"],
            "branch": f"crawl-{self.crawl_meta['crawled_at'][:10]}",
            "sha": self.crawl_meta.get("crawler_sha", "unknown"),
        }
```

### CLI Integration

```bash
# Git source (default)
atlas-build --source-type git --repo-path ./data/docs --repo-url https://github.com/org/docs.git --branch main --output ./bundle

# Web crawl source
atlas-build --source-type web-crawl --mirror-path ./data/snowflake-docs --crawl-meta ./data/snowflake-docs/crawl_meta.json --output ./bundle

# Local directory
atlas-build --source-type local --mirror-path ./my-docs --output ./bundle
```

---

## Configuration (for platform-specific skills)

A separate skill (e.g., `snowflake-atlas-config`) provides corpus-specific values:

```python
# Git-based source
SOURCE_TYPE = "git"
REPO_URL = "https://github.com/<org>/<repo>.git"
BRANCH = "<release-branch>"
REPO_LOCAL_PATH = "./data/<org>-docs/<repo>-<branch>"

# Web-crawl source
SOURCE_TYPE = "web-crawl"
MIRROR_PATH = "./data/snowflake-docs/snowflake-docs-main"
CRAWL_META = "./data/snowflake-docs/crawl_meta.json"

# Shared
DEFAULT_MODEL_ID = "Xenova/bge-base-en-v1.5"
BUNDLE_OUTPUT = "./data/rag-bundle"
```

Passed as CLI args to `atlas-build`, `atlas-fs`, `atlas-rag`, `atlas-download`.

---

## Quick Start

```bash
# 1. Scaffold & install
mkdir my-atlas && cd my-atlas
# Copy pyproject.toml, atlas/ from reference
uv sync --extra build --extra mlx  # or --extra gpu

# 2a. Build from git repo (maintainer)
atlas-build --source-type git \
    --repo-url https://github.com/Org/Docs.git \
    --branch main \
    --output ./data/rag-bundle

# 2b. Build from web crawl mirror (maintainer)
atlas-build --source-type web-crawl \
    --mirror-path ./data/snowflake-docs \
    --crawl-meta ./data/snowflake-docs/crawl_meta.json \
    --output ./data/rag-bundle

# 3. Test
atlas-smoke --bundle ./data/rag-bundle --repo ./data/docs/Repo-main

# 4. Publish (maintainer)
scripts/publish-bundle.sh

# 5. User installs
uv sync --extra mlx
atlas-download --repo Org/AtlasBundles --output ./data/rag-bundle
```

---

## Reference Files (in `references/`)

| File | Contents |
|------|----------|
| `README.md` | This index + quick navigation |
| `architecture.md` | Component diagram, data flow, process model, design decisions |
| `bundle-format.md` | Manifest schema, parquet columns, numpy layouts, SHA256 integrity |
| `backend-selection.md` | Resolution algorithm, backend comparison, overrides, CI guidance |
| `chunking-strategy.md` | H2-boundary algorithm, frontmatter, code flag, hard split, edge cases |
| `testing.md` | Test fixtures, unit/integration/smoke/eval tests, CI integration |