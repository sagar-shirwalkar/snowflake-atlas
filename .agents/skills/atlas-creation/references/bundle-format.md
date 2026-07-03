# Bundle Format Reference

## Overview

The Atlas RAG bundle is a **self-contained, portable directory** that contains everything needed for semantic search at query time. No external dependencies, no database, no network calls.

```
rag-bundle/
├── manifest.json           # Provenance + integrity metadata
├── chunks.parquet          # Chunk text + metadata (snappy compressed)
├── embeddings.f16.npy      # L2-normalized embeddings (float16, row-major)
├── norms.f32.npy           # Precomputed L2 norms (float32, 1D)
└── model/                  # Embedding model files (ONNX + tokenizer)
    ├── onnx/
    │   └── model.onnx
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt           # (if applicable)
```

## Manifest Schema (`manifest.json`)

```json
{
  "schema_version": 1,
  "source_repo": "https://github.com/Org/Docs.git",
  "source_branch": "main",
  "source_sha": "a1b2c3d4e5f6...",
  "source_published": "2026-01-15T00:00:00Z",
  "built_at": "2026-07-03T14:22:10.123456+00:00",
  "chunk_count": 135420,
  "embedding_model": "Xenova/bge-base-en-v1.5",
  "embedding_dim": 768,
  "embedding_backend": "mlx",
  "embedding_active_provider": "mlx",
  "artifacts": {
    "chunks": "chunks.parquet",
    "chunks_sha256": "d4e5f6...",
    "embeddings": "embeddings.f16.npy",
    "embeddings_sha256": "e5f6a7...",
    "norms": "norms.f32.npy",
    "norms_sha256": "f6a7b8...",
    "model_dir": "model/"
  }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Bundle format version (current: 1) |
| `source_repo` | string | Git URL of the docs repository |
| `source_branch` | string | Branch/tag that was built |
| `source_sha` | string | Full git commit SHA (40 hex chars) |
| `built_at` | ISO 8601 | Timestamp when bundle was created |
| `chunk_count` | int | Number of chunks in the bundle |
| `embedding_model` | string | HF model ID or local path used |
| `embedding_dim` | int | Embedding dimension (768 for BGE-base) |
| `artifacts` | object | Map of artifact names to relative paths + SHA256 |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_published` | ISO 8601 | When the source docs were published (if known) |
| `embedding_backend` | string | Backend used at build time (mlx/onnx-cpu/onnx-gpu) |
| `embedding_active_provider` | string | Actual provider string from embedder |

## Chunks Parquet (`chunks.parquet`)

### Schema (PyArrow)

```python
pa.schema([
    ("id", pa.string()),                    # "publication/file#chunk_idx"
    ("text", pa.string()),                  # Chunk body text
    ("publication", pa.string()),           # Top-level folder name
    ("file", pa.string()),                  # Path relative to markdown/
    ("heading", pa.string()),               # H2 heading text (or "Overview")
    ("is_code", pa.bool_()),               # True if chunk has fenced code block
    ("title", pa.string()),                 # From frontmatter.title
    ("product_area", pa.string()),          # From frontmatter.product_area
    ("last_updated", pa.string()),          # From frontmatter.last_updated
    ("canonical_url", pa.string()),         # From frontmatter.canonical_url
])
```

### Compression

- Codec: `snappy` (fast, good compression for text)
- Row groups: ~50k rows each (default pyarrow behavior)

### Chunk ID Format

```
{publication}/{file}#{chunk_index}
```

Examples:
- `it-service-management/incident-management.md#0`
- `it-service-management/incident-management.md#3`
- `it-operations-management/event-management/alert-management.md#1`

## Embeddings (`embeddings.f16.npy` / `embeddings.f32.npy`)

### Format

- NumPy `.npy` file (little-endian, row-major C-contiguous)
- Shape: `(chunk_count, embedding_dim)` — e.g., `(135420, 768)`
- Dtype: `float16` (preferred) or `float32`
- **Rows are L2-normalized** (unit vectors)

### Float16 vs Float32

| Property | float16 | float32 |
|----------|---------|---------|
| Disk size (250k × 768) | ~375 MB | ~750 MB |
| Cosine ranking | Preserved | Exact |
| Load time | Faster | Slower |
| Recommended | ✓ Default | For validation only |

**Why float16 works**: Cosine similarity = `u·v / (|u||v|)`. With L2-normalized vectors, `|u|=|v|=1`, so cosine = `u·v`. Half precision preserves dot product ranking with negligible error (<0.001% rank change empirically).

## Norms (`norms.f32.npy`)

### Format

- NumPy `.npy` file
- Shape: `(chunk_count,)` — 1D array
- Dtype: `float32`
- Values: `||embedding_row||_2` computed from the **stored** embeddings (float16→float32)

### Purpose

At query time: `cosine = (embeddings @ query) / norms`

Precomputing norms avoids `sqrt(sum(x²))` per row at query time. Stored in float32 for numerical stability even when embeddings are float16.

## Model Directory (`model/`)

### Layout

Mirrors the Hugging Face / Xenova repo structure so the runtime can load identically:

```
model/
├── onnx/
│   └── model.onnx          # ONNX graph (exported from BGE-base)
├── tokenizer.json          # Fast tokenizer (tokenizers library)
├── tokenizer_config.json   # Tokenizer config
├── special_tokens_map.json # Special tokens
└── vocab.txt               # Vocabulary (if BPE/WordPiece)
```

### For MLX Backend

The MLX embedder expects weights converted to `.npy` files in a cache directory (default `~/.cache/atlas/mlx/bge-base-en-v1.5/`). The bundle's `model/` dir is ONNX-only; MLX conversion is a separate one-time step via `tools/convert_bge_to_mlx.py`.

## Integrity Verification

### SHA256 Computation

```python
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

### Verification Points

1. **At download** (`atlas-download`): Verify `chunks.parquet` SHA matches manifest
2. **At doctor** (`atlas-doctor --bundle`): Verify all artifacts with SHA in manifest
3. **At bundle build** (`atlas-build`): Write SHA for each artifact to manifest

### What's Not SHA-Verified

- `model/` directory (too many files, not critical for retrieval correctness)
- `norms.f32.npy` (derived from embeddings; verified indirectly)

## Versioning

| Schema Version | Changes | Compatibility |
|----------------|---------|---------------|
| 1 | Initial release | Current |

Future versions should:
- Increment `schema_version`
- Add new optional fields only
- Never remove or rename required fields
- Maintain backward-compatible parquet schema (add columns, don't remove)

## Size Estimates

| Corpus Size | Chunks | embeddings.f16.npy | chunks.parquet | Total |
|-------------|--------|-------------------|----------------|-------|
| Small (5k files) | ~25k | ~38 MB | ~15 MB | ~60 MB |
| Medium (50k files) | ~135k | ~200 MB | ~80 MB | ~300 MB |
| Large (200k files) | ~500k | ~750 MB | ~300 MB | ~1.1 GB |

*Assumes 768-dim BGE-base, snappy compression, avg 1.5k chars/chunk*

## Loading at Runtime (`rag_server.py`)

```python
class Bundle:
    def __init__(self, bundle_dir: Path, prefer: str = "auto"):
        # 1. Load manifest
        self.manifest = json.loads((bundle_dir / "manifest.json").read_text())
        
        # 2. Load chunks metadata
        table = pq.read_table(bundle_dir / "chunks.parquet")
        
        # 3. Load embeddings (prefer float16)
        f16 = bundle_dir / "embeddings.f16.npy"
        if f16.exists():
            self.embeddings = np.load(f16).astype(np.float32)
        else:
            self.embeddings = np.load(bundle_dir / "embeddings.f32.npy")
        
        # 4. Load or compute norms
        norms_path = bundle_dir / "norms.f32.npy"
        if norms_path.exists():
            self.norms = np.load(norms_path)
        else:
            self.norms = np.linalg.norm(self.embeddings, axis=1).astype(np.float32)
        
        # 5. Resolve embedder (bundled model preferred)
        bundled_model = bundle_dir / "model" / "onnx" / "model.onnx"
        model_id = bundle_dir / "model" if bundled_model.exists() else DEFAULT_MODEL_ID
        self.embedder = get_embedder(model_id, prefer=prefer)
        
        # 6. Extract columns to numpy for fast filtering
        self._pub = table.column("publication").to_numpy()
        self._area = table.column("product_area").to_numpy()
        self._code = table.column("is_code").to_numpy().astype(bool)
        self._texts = table.column("text").to_pylist()
        # ... etc
```

## Query-Time Cosine Similarity

```python
# Query embedding (already L2-normalized by embedder)
q = self.embedder.embed([query])[0]  # shape (768,)

# Single matrix multiply: (n_chunks, 768) @ (768,) → (n_chunks,)
scores = (self.embeddings @ q) / self.norms  # cosine similarity
```

This is the entire retrieval computation — one BLAS call, no index, no ANN.