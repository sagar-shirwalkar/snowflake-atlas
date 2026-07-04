---
name: atlas-creation
description: Build a local-first AI knowledge layer (Atlas) for any markdown documentation corpus via two MCP servers (filesystem + RAG). Use when the user wants to create an Atlas system for a new documentation source — e.g., "build an Atlas for Kubernetes docs", "create a knowledge layer for OpenStack docs", "set up dual MCP servers for my markdown repo". Branches: build-atlas (full pipeline), build-fs-server (filesystem only), build-rag-server (RAG only), build-bundle (precompute embeddings), eval-rag (measure retrieval quality).
disable-model-invocation: false
---

An **Atlas** is a local-first AI knowledge layer built from a markdown documentation corpus. It exposes two Model Context Protocol (MCP) servers:

1. **Filesystem server** (`atlas-fs`) — deterministic, zero-infra, backed by `ripgrep`. Tools: list publications, list files, read file, full-text search, get release info. No model, no embeddings, no state. Works with any markdown repo.

2. **RAG server** (`atlas-rag`) — semantic search over precomputed embeddings. Tools: search_docs, search_code, get_chunk, get_bundle_info. Loads a portable bundle once at startup; answers queries via single matrix multiply. Supports MLX (Apple Silicon), ONNX+CUDA (NVIDIA), ONNX+CPU (portable floor).

The bundle is built **once** by the maintainer (`atlas-build`), distributed as a single artifact, and consumed by end users with zero embedding/chunking/model work.

This skill is **platform-agnostic**. Corpus-specific configs (repo URLs, branch names, model IDs) live in separate skills.

---

## Advanced Retrieval Techniques (optional, build-time only)

The base Atlas system uses dense embedding search with BM25 hybrid as default.
The techniques below improve retrieval quality without adding any LLM calls at
query time — all work happens once during bundle creation.

---

### 1. Chunk Overlap

**Idea:** Adjacent H2 sections share a small text overlap (the tail of the previous
section prepended to the next) so that queries straddling a section boundary still
match. The overlap text is prepended without a special marker — the embedding model
treats it as natural context.

**Implementation:**

- `atlas/chunk.py` extracts the last 150 chars (`_OVERLAP_CHARS`) of each H2
  section via `_section_tail()`, word-broken cleanly.
- `chunk_markdown()` prepends the tail to the next section before embedding.
- Controlled by `overlap_chars` parameter (default 150).

**Cost profile:** Zero at query time. Adds ~150 tokens per chunk boundary at build
time (negligible memory/vector storage impact).

---

### 2. Hierarchical FS Chunking with Path Metadata

**Idea:** The file-system hierarchy of the documentation corpus carries semantic
information — documents in the same directory are about related topics, and sibling
filenames form implicit clusters. This technique tags each chunk with its directory
neighbourhood at build time, then applies a small relevance boost at query time.

**Implementation (build-time):**

- `atlas/make_bundle.py::build_chunk_table()` pre-computes a sibling map before
  chunking: for each file, it collects the stems (filenames without `.md`) of all
  other `.md` files in the same directory.
- Each chunk gets a `cluster_tags` column: space-joined sibling stems (excluding
  self). Example: a chunk from `provider-listings-auto-fulfillment-setup.md` in
  `collaboration/views/` gets cluster_tags = `"provider-listings-auto-fulfillment provider-listings-auto-fulfillment-setup"`.
- The column is stored in `chunks.parquet` alongside all other metadata.

**Implementation (query-time):**

- `Bundle.search()` does substring matching (`pc.match_substring`) on
  `cluster_tags` against query tokens.
- Matches get a small boost (`0.03` weight, vs `0.05` for title boost).
- Graceful degradation: old bundles without `cluster_tags` skip the boost.

**Cost profile:** Zero query-time LLM. Build-time overhead is negligible (set
operations + one extra parquet column). Query-time adds one `pc.match_substring`
pass per query token.

---

### 3. Hybrid Search as Default

**Idea:** Dense embedding search captures semantic similarity but misses exact term
matches. BM25 keyword search captures term precision but misses semantically related
misses. Score-level fusion (weighted sum of normalised BM25 and dense cosine similarity) combines both. Making hybrid the default means
clients get better recall without any configuration.

**Implementation:**

- `Bundle.search()` defaults `mode="hybrid"` in both `search_docs` and
  `search_code` tools.
- Fusion formula: `0.6 * dense_score + 0.4 * bm25_norm` (weights tuned empirically).
- BM25 index built at bundle-build time via `rank-bm25::BM25Okapi` from chunk texts.
- Default BM25 parameters: `k1=2.0, b=0.7` (tuned for longer reference docs — higher TF saturation ceiling, reduced length normalisation penalty).
- **Interleaved candidate pool:** For hybrid and keyword modes, the candidate pool is the
  union of the top-N vector-selected and top-N BM25-selected indices. This ensures
  BM25-favoured documents (short, exact-match, low vector similarity) still enter the
  fusion. Zero extra BM25 computation (already O(N) per query).
- Graceful fallback: if `bm25.pkl` is missing (older bundles), hybrid falls back to
  title-boost heuristic with a log warning.

**Cost profile:** Zero query-time LLM. Build-time indexing is ~1-2s for 250k chunks.
Query-time adds a BM25 score call (microseconds on the candidate pool).

---

### 4. Cross-Encoder Re-ranker (opt-in, MLX or ONNX)

**Idea:** A cross-encoder jointly encodes the query and each candidate document,
producing a relevance score that is more accurate than cosine similarity alone.
Applied to the top-100 candidates from hybrid search, it eliminates false positives
that rank high on embedding similarity but are actually irrelevant.

**Implementation:**

- `atlas/rerank_mlx.py` — `MlxCrossEncoderReranker` uses a BERT-base
  classifier architecture (same BERT-base as the embedder) accelerated via MLX
  on Apple Silicon.
- `atlas/rerank.py` — `CrossEncoderReranker` uses ONNX Runtime (portable,
  `cross-encoder/ms-marco-MiniLM-L6-v2`).
- Runtime resolves: try MLX first, fall back to ONNX on `(ImportError,
  FileNotFoundError, RuntimeError)`.
- Enabled via `--rerank` flag on the RAG server (off by default — opt-in
  because it adds ~45 MB RAM and ~5 ms/query).
- Model weights: `BAAI/bge-reranker-v2-base` converted to MLX `.npy` format
  via `tools/convert_reranker_to_mlx.py`.

**Cost profile:** Zero LLM calls. Requires a pre-converted MLX weight cache or
HF-downloaded ONNX model. Adds ~5 ms/query on Apple Silicon, ~50-200 ms on CPU.

---

### 5. Fielded BM25 with File-Path Signals

**Idea:** Standard BM25 treats every document as a bag of words from the body
text. Fielded BM25 gives the index a notion of document structure by building
separate BM25 instances per field (body text, title, heading, file path) and
combining their scores at query time as a weighted sum. This lets the index
reward matches in semantically important fields (a match in the title or
filename is more relevant than a match deep in the body).

**Implementation (build-time):**

- `atlas/bm25_search.py::build_fielded_index()` accepts a dict of
  `field_name -> list[document_text]` and a dict of `field_name -> weight`.
- Builds a separate `BM25Okapi` instance per field, each with the same
  tuned parameters (`k1=2.0, b=0.7`).
- Scores are combined at query time: `final_score = sum(weight_f * bm25_f(tokens))`.
- Persisted as a version-2 pickle with backward-compatible loading (old
  version-1 pickles still work).

**Default field weights:**

| Field | Weight | Rationale |
|-------|--------|-----------|
| `text` (body) | 1.0 | Baseline — the main content |
| `heading` | 2.0 | Section headings summarise the chunk's topic |
| `file_path` | 2.5 | Filename stems are expert-curated relevance tags |
| `title` | 3.0 | Document titles are concise human-written summaries |

**File-path field in detail:** The file path of each markdown document is
treated as a separate BM25 field. Filenames in well-structured documentation
corpuses are hand-crafted identifiers that directly encode topic relevance:

```
api/endpoints/create-user.md      → tokens: api, endpoints, create, user
guides/permissions-roles.md       → tokens: guides, permissions, roles
```

The `.md` extension is stripped before tokenization so the tokenizer produces
clean, meaningful tokens. A query containing "create warehouse" matches
directly against the file-path tokens of `create-warehouse.md`.

**Request-side query expansion (corpus-free):** The BM25 tokenizer applies
morphological variant generation at query time — for each query token it generates
plural/singular forms and common verb variants (e.g., ``"warehouse"`` →
``"warehouses"``, ``"warehousing"``; ``"create"`` → ``"creating"``,
``"created"``). This bridges the morphological gap without stemming the corpus,
preserving the full word forms for any future NLP use. SQL identifiers like
``TO_DATE`` pass through unchanged. Short tokens (≤3 chars) skip expansion.
Non-matching generated variants contribute zero to BM25 scores — false positives
are harmless. Cost: ~0.01ms per query, zero build-time overhead.

**Cost profile:** Zero query-time LLM cost. Build-time overhead is negligible
(~1-2 s for 250k chunks). Query-time scoring iterates over fields (microseconds).
Pickle size increases marginally (~100 KB for 50k chunks with 4 fields).

---

### 6. Adaptive Diversity Ranking (AdaGReS)

**Idea:** Hybrid search fusion often returns multiple chunks from the same file
in the top-k results — a long document split across many sections can dominate
the ranking, crowding out relevant chunks from other files. AdaGReS (Adaptive
Greedy Selection) is a diversity-aware ranking algorithm that adaptively trades
relevance for diversity as rank increases. At rank 1, relevance is prioritised;
by rank 5, diversity dominates. This ensures the top-k results are both relevant
and file-diverse — critical for AI agents that scan only the top 3-5 results.

**How it works:**

1. Start with the highest-scoring item (pure relevance).
2. For each subsequent pick, compute an adaptive lambda:
   ``λ_k = exp(-α × (k-1))`` where ``α = 0.5``.
3. Score each remaining candidate:
   ``Score(j) = λ_k × relevance(j) + (1 - λ_k) × (1 - max_sim(j, selected))``
4. ``max_sim`` is the candidate's maximum similarity to already-selected items.

**Similarity rules for heading-aware dedup:**

| Condition | sim | Meaning |
|-----------|-----|---------|
| Same file + same heading | 1.0 | Fully redundant — blocked |
| Same file + different heading | 0.5 | Different section, partially penalised |
| Different file | 0.0 | No penalty — diversity automatically favoured |

**Implementation (query-time only, no rebuild needed):**

- ``atlas/rag_server.py::Bundle._adagres_select()`` is called after hybrid fusion
  and before result collection.
- Uses ``self._files`` and ``self._headings`` already loaded from the bundle.
- O(n × k) complexity with n = pool size (~200-400) and k = top_k (~5-10).
  Imperceptible at query time (~0.01ms).

**Cost profile:** Zero LLM cost. Zero build-time cost. Query-time overhead of
~0.01ms per search. No bundle rebuild required.

---

### 7. Contextual Retrieval

**Source:** [Anthropic Blog — Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Sep 2024)

**Idea:** A naive chunking strategy splits documents at heading boundaries, but many chunks become
ambiguous when viewed in isolation — a code block showing `TO_DATE` usage, a table of SQL function
signatures, a CLI command reference. These chunks carry no surrounding context about which section
they belong to or what problem they solve.

Contextual Retrieval solves this by having an LLM generate a short (50-100 token) **context snippet**
for each chunk before embedding: a plain-text explanation of what the chunk is and how it relates
to the overall document. This context is prepended to the chunk text before it enters the embedding
model, so the vector representation captures both the chunk content _and_ its role in the document.

**Claimed impact:** Anthropic reports up to **49% reduction in retrieval failure rate** compared to
naive chunking without context — the single highest-leverage improvement in their test suite.

**Status:** Planned, not yet implemented. Context generation requires LLM calls
at build time, so it depends on provider selection (local via MLX, Claude API, OpenRouter, HF).
The best-quality results come from frontier models (Claude 3.5 Haiku/Sonnet), but smaller local
models (Qwen 2.5 7B, Phi-3-mini) can produce adequate context for simpler chunks. Consider
Contextual Retrieval after the build-time-only techniques above are exhausted.

---

#### Provider Options

Context generation is a **build-time** step — it runs once per bundle release, so per-chunk cost
matters less than context quality. Bad context pollutes the embedding and makes retrieval _worse_
than no context at all. Choose the provider accordingly.

| Provider | Models | Quality | Cost | Setup |
|----------|--------|---------|------|-------|
| **Local via MLX** | Qwen 9B, LLaMA 3.2, Phi-3-mini | Variable — test first | Free | `pip install mlx-lm`, no API key |
| **Local via Ollama** | Any Ollama model | Variable — test first | Free | `ollama pull <model>`, no API key |
| **Claude API** | Claude 3 Haiku / Sonnet | Highest | ~$1/M chunks w/ prompt caching | `ANTHROPIC_API_KEY` env var |
| **OpenRouter** | Dozens of models, pay-per-token | Depends on model chosen | $0.15-2/M tokens (many free tier models) | `OPENROUTER_API_KEY` env var + `OPENROUTER_BASE_URL` |
| **HuggingFace Inference** | Free serverless models (e.g. `HuggingFaceH4/zephyr-7b-beta`) | Lower — good for prototyping | Free (rate-limited) | `HF_TOKEN` env var (optional for free tier) |
| **opencode** | Inherits the editor's configured provider | Same as whatever opencode uses | Same as whatever opencode uses | Auto — reads opencode config from `~/.config/opencode/` |

**Recommendation:** Use Claude Haiku via API for the highest quality at low cost. Use a local
model (via MLX or Ollama) when data must not leave the machine or for air-gapped deployments.
OpenRouter and HF free tiers are good for prototyping. opencode integration is useful when the
build runs in an environment already configured for the editor.

---

#### Implementation Pattern

Context generation is a **build-time** step, integrated into Phase 6 (bundle builder) of the
10-phase build process. It runs once per bundle release with zero query-time cost.

**Security: API keys from environment only.** Never store API keys in
`pyproject.toml`, `config.json`, or any file that could be committed. The build tool
should fail early with a clear error if a required env var is missing:

```python
os.environ["ANTHROPIC_API_KEY"]  # raises KeyError if unset — fail fast
```

**Integration into bundle build:**

```python
# In make_bundle.py Phase 6, after chunking, before embedding:
context_llm = get_context_llm(args.context_provider)
for chunk in chunks:
    context = generate_chunk_context(
        chunk["text"], chunk["document"], chunk["section"], context_llm,
    )
    chunk["text"] = f"<context>{context}</context>\n{chunk['text']}"
```

The `--context-provider` CLI flag selects the backend; `--context-model` overrides the
default model for that provider. Both are optional — when omitted, context generation is
skipped entirely (maintaining backward compatibility).

**Bundle considerations:** Context is prepended _before_ embedding, so the vector captures
it. The stored text in `chunks.parquet` includes the context prefix. No bundle format
change — the context becomes part of the chunk text.

---

#### When to use

- **Build-time only** — context is generated once during bundle creation, not at query time.
  The per-chunk LLM call adds to build time but costs nothing at query time.
- **Essential for technical docs** where code blocks, SQL snippets, API signatures, and CLI
  commands make up most of the content — these are the chunks that benefit most from
  disambiguation.
- **Skip** when your corpus is prose-heavy with long, self-contained paragraphs (blog posts,
  essays) or when you cannot run an LLM during build (air-gapped build environment with no
  local model available).

---

### Future: ColBERT Late Interaction

**Source:** [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction
over BERT](https://arxiv.org/abs/2004.12832) (SIGIR'20)

**Idea:** ColBERT replaces the single-score dot product of dense retrieval with a
**late interaction** mechanism: each query token is compared against each document
token independently, and the scores are summed (MaxSim). This preserves fine-grained
term-level matching while still using contextualized BERT representations — bridging
the gap between dense and sparse retrieval.

**Tradeoff vs. cross-encoder reranker:** ColBERT is faster (can be fully indexed
and searched in a single pass) but requires a specialized inference pipeline. The
cross-encoder reranker is simpler to integrate as a second-pass re-ranker on top
of existing dense+hybrid results.

**Status:** Not implemented. ColBERT's late interaction is a
promising direction for a future retrieval upgrade, particularly for the hybrid
search pipeline where it could replace the separate BM25 + dense fusion with a
single unified retriever.

---

### Summary: Which technique for what?

| Goal | Technique | Effort | Impact | Cost profile |
|------|-----------|--------|--------|-------------|
| Seamless section boundaries | Chunk Overlap (#1) | Low | Medium | Build-time tail extraction, zero query cost |
| Path-aware retrieval boost | Hierarchical FS Chunking (#2) | Low | Medium | Build-time sibling map, query-time substring match |
| Maximum recall without config | Hybrid Search as Default (#3) | Low | High | Build-time BM25 indexing, zero query-time LLM |
| Precision boost for top results | Cross-Encoder Reranker (#4) | Medium | High | Build-time weight conversion, passive at query |
| Structured keyword relevance | Fielded BM25 + File-Path (#5) | Low | Medium | Build-time per-field indexing, zero query-time cost |
| Context-disambiguated chunks | Contextual Retrieval (#7) | High | High | Build-time LLM call per chunk, zero query-time cost |
| Diverse top-k for AI agents | AdaGReS Diversity (#6) | Low | Medium | Query-time O(n×k) selection, zero LLM, no rebuild |
| Unified dense+sparse retrieval | ColBERT (Future) | High | High | Specialized indexing pipeline, single-pass search |

## Core concepts (leading words)

- **dual-server** — FS for verbatim citations, RAG for fuzzy discovery
- **bundle** — portable artifact (chunks.parquet + embeddings.npy + norms.npy + model/ + manifest.json)
- **pinned** — source repo URL, branch, and git SHA recorded for reproducibility
- **backend** — inference runtime resolved at run time: MLX, ONNX-CUDA, ONNX-CPU
- **H2-chunk** — split on `## ` headers, parse YAML frontmatter, flag code chunks
- **overlap** — tail of previous H2 section prepended to next for boundary-spanning queries (150 chars default)
- **cluster-tags** — space-joined sibling document stems in the same directory, used for path-aware relevance boost
- **hybrid-default** — RAG server defaults to hybrid (dense + BM25) mode; client can override per-query
- **fielded-bm25** — multi-field BM25 index (text + title + heading + file_path) with per-field weights, built at bundle time
- **file-path-bm25** — document file paths as BM25 field tokens; filenames like `create-warehouse` are expert-curated relevance signals
- **reranker** — optional cross-encoder (MLX or ONNX) that re-scores top-100 candidates for precision
- **query-expansion** — request-side morphological variant generation for BM25 tokens (singular/plural, verb forms), no corpus modification needed
- **adagres** — heading-aware diversity ranking that adaptively trades relevance for diversity across rank positions, zero rebuild cost
- **doctor** — diagnostic probe (`atlas-doctor`) showing platform, backend probes, selected backend + reason
- **source-adapter** — pluggable interface to fetch markdown from git, web crawl, local dir, or API

---

## 10-Phase Build Process

| Phase | Task | Completion Criterion |
|-------|------|---------------------|
| 0 | Prerequisites | `python3 uv rg git` available; Python 3.11+ |
| 1 | Scaffold project | `pyproject.toml` with entry points + deps; `atlas/__init__.py` |
| 2 | Chunker (`atlas/chunk.py`) | `chunk_file()` returns dicts with 10 required keys; H2-boundary split + chunk overlap (150 chars) |
| 3 | Embedding backends (`atlas/embed/`) | 3 backends (MLX, ONNX-CUDA, ONNX-CPU); factory picks best at runtime |
| 4 | FS MCP server (`atlas/fs_server.py`) | 5 tools registered; deterministic; ripgrep-backed; path traversal blocked |
| 5 | RAG MCP server (`atlas/rag_server.py`) | 4 tools registered; bundle loaded once; cosine via matrix multiply; hybrid as default mode with interleaved candidate pool and AdaGReS diversity ranking |
| 6 | Bundle builder (`atlas/make_bundle.py`) | End-to-end: fetch → chunk (w/ overlap + cluster_tags) → embed → fielded BM25 index (text + title + heading + file_path) → write artifacts → stage model → manifest w/ SHA256 |
| 7 | Download + verify (`atlas/download.py`) | GitHub Releases download; SHA256 verify; backup existing before overwrite |
| 8 | Utilities | backup, restore, doctor, smoke_test, evaluate (with `--mode vector|keyword|hybrid`), log, rerank (MLX + ONNX) — all working standalone |
| 9 | Tests (`tests/`) | Pytest suite passes: chunker (incl. overlap), embed backends, FS/RAG servers, bundle build, BM25, reranker |
| 10 | CI + Release | GitHub Actions smoke test; `scripts/publish-bundle.sh` creates release |

---

## Source Adapters

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
| `GitSource` | Git repository | Publicly maintained docs (Kubernetes, open-source projects) |
| `WebCrawlSource` | Local mirror from web crawl | Docs without public git mirrors (cloud platforms, SaaS) |
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

### WebCrawlSource

```python
# atlas/sources/web_crawl.py
class WebCrawlSource(MarkdownSource):
    def __init__(self, mirror_root: Path, crawl_meta: dict):
        self.mirror_root = mirror_root
        self.crawl_meta = crawl_meta  # {'source_url': 'https://docs.example.com', 'crawled_at': '2025-01-01'}
    
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
atlas-build --source-type web-crawl --mirror-path ./data/docs-mirror --crawl-meta ./data/docs-mirror/crawl_meta.json --output ./bundle

# Local directory
atlas-build --source-type local --mirror-path ./my-docs --output ./bundle
```

---

## Configuration (for platform-specific skills)

A separate platform-specific skill provides corpus-specific values:

```python
# Git-based source
SOURCE_TYPE = "git"
REPO_URL = "https://github.com/<org>/<repo>.git"
BRANCH = "<release-branch>"
REPO_LOCAL_PATH = "./data/<org>-docs/<repo>-<branch>"

# Web-crawl source
SOURCE_TYPE = "web-crawl"
MIRROR_PATH = "./data/docs-mirror"
CRAWL_META = "./data/docs-mirror/crawl_meta.json"

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
    --mirror-path ./data/docs-mirror \
    --crawl-meta ./data/docs-mirror/crawl_meta.json \
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