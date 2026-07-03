# Atlas Architecture Reference

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ATLAS ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐      │
│   │  DOCS REPO   │      │  MAKE_BUNDLE │      │      RAG BUNDLE      │      │
│   │  (markdown/) │─────▶│   (atlas-    │─────▶│  (portable artifact) │      │
│   │  git clone   │      │    build)    │      │                      │      │
│   └──────────────┘      └──────────────┘      │  chunks.parquet      │      │
│                                                │  embeddings.f16.npy  │      │
│                                                │  norms.f32.npy       │      │
│                                                │  model/ (ONNX/MLX)   │      │
│                                                │  manifest.json       │      │
│                                                └──────────┬───────────┘      │
│                                                           │                │
│                        ┌─────────────────────────────────────────────┼────────────┐  │
│                        ▼                                 ▼            ▼  │
│               ┌─────────────────┐               ┌─────────────────┐     │
│               │   ATLAS-FS      │               │   ATLAS-RAG     │     │
│               │ (Filesystem MCP)│               │ (RAG MCP)       │     │
│               │                 │               │                 │     │
│               │ Tools:          │               │ Tools:          │     │
│               │ • list_pubs     │               │ • search_docs   │     │
│               │ • list_files    │               │ • search_code   │     │
│               │ • read_file     │               │ • get_chunk     │     │
│               │ • search (rg)   │               │ • bundle_info   │     │
│               │ • release_info  │               │                 │     │
│               │                 │               │ Backend:        │     │
│               │ Zero model      │               │ • MLX (Apple)   │     │
│               │ Zero embeddings │               │ • ONNX-CUDA     │     │
│               │ Zero state      │               │ • ONNX-CPU      │     │
│               └────────┬────────┘               └────────┬────────┘     │
│                        │                                 │              │
│                        └─────────────────────────────────┘              │
│                                          │                              │
│                                          ▼                              │
│                              ┌─────────────────────┐                   │
│                              │   MCP CLIENT        │                   │
│                              │ (Zed, opencode,     │                   │
│                              │  Claude Desktop,    │                   │
│                              │  custom agent)      │                   │
│                              └─────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Build Time (Maintainer)
```
1. git clone/fetch ──────▶ 2. walk markdown ──────▶ 3. H2-chunk + frontmatter
                                                                      │
4. write chunks.parquet ◀──── 5. embed all chunks ◀──────────────────┘
                                                                      │
6. save embeddings.npy + norms.npy ◀────────────────────────────────┘
                                                                      │
7. stage model files ◀──────────────────────────────────────────────┘
                                                                      │
8. write manifest.json (provenance + SHA256s)
```

### Run Time (End User)
```
ATLAS-FS (stdio)                          ATLAS-RAG (stdio)
    │                                         │
    ├─ list_publications()                    ├─ search_docs(query)
    ├─ list_files(pub)                        ├─ search_code(query)
    ├─ read_file(pub, file)                   ├─ get_chunk(id)
    ├─ search(query, scope?)                  └─ get_bundle_info()
    └─ get_release_info()
           │                                        │
           └────────────────────┬───────────────────┘
                                ▼
                      MCP CLIENT (any)
```

## Process Model

| Component | Process | Startup Cost | Per-Query Cost |
|-----------|---------|--------------|----------------|
| `atlas-fs` | Long-lived MCP server | ~50 ms | ~10-50 ms (rg) |
| `atlas-rag` | Long-lived MCP server | ~2-5 s (loads bundle) | ~1-5 ms (MLX), ~50-200 ms (CPU) |
| `atlas-build` | One-shot CLI | N/A | N/A (batch) |
| `atlas-download` | One-shot CLI | N/A | N/A (network) |

## Key Design Decisions

### Why Two Servers?
1. **Different strengths** — FS for exact citations, RAG for fuzzy discovery
2. **Different trust levels** — FS returns verbatim markdown; RAG returns ranked candidates
3. **Different cost profiles** — FS ~zero startup; RAG loads ~1GB vectors but answers in ~ms
4. **Different model fits** — Weak models struggle with FS navigation but handle RAG top-k well

### Why No Vector Database?
- At 250k × 768 dims, single `numpy` matrix multiply: ~50 ms on Apple Silicon
- FAISS/ChromaDB add complexity, deps, and are not portable across runtimes
- Bundle is self-contained: no external service, no network, no persistence layer

### Why ONNX + MLX?
- **ONNX+CPU**: Portable floor, runs everywhere (Linux, Windows, Intel Mac, CI)
- **ONNX+CUDA**: NVIDIA GPU acceleration where available
- **MLX**: Apple Silicon native, uses ANE/GPU directly, 5-10x faster than CPU
- Bundle stores embeddings only — inference backend chosen at **run time**, not build time

### Why H2-Boundary Chunking?
- Respects docs team's deliberate structure (one topic per file, H2 = sub-topic)
- Frontmatter provides rich metadata (title, product_area, canonical_url) per chunk
- Code flag enables `search_code` tool
- Max 8000 chars with paragraph fallback handles edge cases

### Why Float16 Embeddings?
- Cosine similarity is rank-preserving under half precision
- Halves on-disk size (360 MB → 180 MB for 250k chunks)
- Norms stored separately in float32 for accurate cosine at query time

## File Layout

```
atlas/
├── __init__.py           # version, package docstring
├── chunk.py              # H2-chunker + frontmatter parser
├── embed/
│   ├── __init__.py
│   ├── base.py           # ABC, factory, resolve_backend, helpers
│   ├── onnx.py           # OnnxEmbedder (CPU + CUDA)
│   └── mlx.py            # MlxEmbedder (Apple Silicon)
├── fs_server.py          # Filesystem MCP server
├── rag_server.py         # RAG MCP server
├── make_bundle.py        # Build orchestrator
├── download.py           # Download + verify from Releases
├── backup.py             # Snapshot bundle
├── restore.py            # Roll back snapshot
├── smoke_test.py         # E2E validation
├── doctor.py             # Installation diagnosis
├── evaluate.py           # RAG quality eval (P@10, MRR)
├── log.py                # Structured logging
├── rerank.py             # Cross-encoder re-ranker
├── agent.py              # [planned] Reasoning agent
└── training.py           # [planned] Fine-tuning pipeline

tools/
└── convert_bge_to_mlx.py # One-time HF→MLX weight conversion

scripts/
└── publish-bundle.sh     # Local build + gh release create

data/
├── .gitkeep
├── <org>-docs/           # Local clone (gitignored)
│   └── <repo>-<branch>/
└── rag-bundle/           # Pre-built bundle (gitignored)

tests/
├── test_chunk.py
├── test_embed.py
├── test_fs_server.py
├── test_rag_server.py
├── test_make_bundle.py
└── test_download.py

.github/workflows/
└── build-bundle.yml      # Smoke tests + release automation
```

## Extension Points

| Extension | Where | How |
|-----------|-------|-----|
| New embedding model | `atlas/embed/base.py` | Change `DEFAULT_MODEL_ID`, ensure dim matches |
| New backend | `atlas/embed/` | Implement `Embedder` ABC, add to `get_embedder()` |
| New chunking strategy | `atlas/chunk.py` | Replace `chunk_markdown()`, keep output schema |
| New FS tool | `atlas/fs_server.py` | Add to `list_tools()`, handle in `call_tool()` |
| New RAG tool | `atlas/rag_server.py` | Add to `list_tools()`, handle in `call_tool()` |
| New search mode | `atlas/rag_server.py` | Extend `Bundle.search()` `mode` parameter |
| New eval metric | `atlas/evaluate.py` | Add function, wire into CLI |