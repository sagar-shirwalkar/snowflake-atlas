# Testing Reference

## Test Suite Overview

```
tests/
├── test_chunk.py           # Chunker + frontmatter (unit)
├── test_embed.py           # All 3 embedding backends (integration)
├── test_fs_server.py       # FS MCP server tools (integration)
├── test_rag_server.py      # RAG MCP server tools (integration)
├── test_make_bundle.py     # Bundle build pipeline (integration)
└── test_download.py        # Download + verify (unit + mock)
```

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Single test file
uv run pytest tests/test_chunk.py -v

# With coverage
uv run pytest tests/ --cov=atlas --cov-report=term-missing

# Smoke test (full E2E)
atlas-smoke --bundle ./data/rag-bundle --repo ./data/servicenow-docs/ServiceNowDocs-australia
```

## Test Fixtures

### Temporary Markdown Repo (`tests/conftest.py`)

```python
@pytest.fixture
def temp_markdown_repo(tmp_path: Path) -> Path:
    """Create a minimal markdown repo structure for testing."""
    repo = tmp_path / "TestDocs-main"
    md_root = repo / "markdown"
    
    # Publication 1
    pub1 = md_root / "pub-one"
    pub1.mkdir(parents=True)
    (pub1 / "intro.md").write_text("""---
title: "Introduction"
product_area: "Core"
last_updated: "2026-01-01"
canonical_url: "https://example.com/intro"
---
# Introduction

Overview text.

## Getting Started

Start here.

## Advanced Topics

More details.
""")
    
    # Publication 2
    pub2 = md_root / "pub-two"
    pub2.mkdir(parents=True)
    (pub2 / "guide.md").write_text("""---
title: "User Guide"
product_area: "Apps"
last_updated: "2026-02-01"
canonical_url: "https://example.com/guide"
---
# Guide

## Installation

\`\`\`bash
pip install foo
\`\`\`

## Usage

Use it.
""")
    
    return repo
```

### Test Bundle (`tests/conftest.py`)

```python
@pytest.fixture
def test_bundle(tmp_path: Path, temp_markdown_repo: Path) -> Path:
    """Build a minimal bundle for testing (skip embed for speed)."""
    bundle_dir = tmp_path / "test-bundle"
    bundle_dir.mkdir()
    
    # Run make_bundle with --limit 5 --skip-embed
    import subprocess
    result = subprocess.run([
        "atlas-build",
        "--repo-path", str(temp_markdown_repo),
        "--output", str(bundle_dir),
        "--limit", "5",
        "--skip-embed"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    
    # Manually add dummy embeddings + norms for RAG tests
    import numpy as np
    import json
    
    # Read chunk count
    import pyarrow.parquet as pq
    table = pq.read_table(bundle_dir / "chunks.parquet")
    n_chunks = len(table)
    
    # Write dummy embeddings (L2-normalized random)
    embeddings = np.random.randn(n_chunks, 768).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(bundle_dir / "embeddings.f16.npy", embeddings.astype(np.float16))
    np.save(bundle_dir / "norms.f32.npy", np.ones(n_chunks, dtype=np.float32))
    
    # Update manifest
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    manifest["artifacts"]["embeddings_sha256"] = "dummy"
    manifest["artifacts"]["norms_sha256"] = "dummy"
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    
    # Stage dummy model
    model_dir = bundle_dir / "model" / "onnx"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"dummy")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (bundle_dir / "model" / name).write_text("{}")
    
    return bundle_dir
```

## Test Chunk (`test_chunk.py`)

```python
def test_parse_frontmatter():
    text = "---\ntitle: Test\n---\nBody"
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "Test"
    assert body == "Body"

def test_no_frontmatter():
    text = "No frontmatter here"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text

def test_h2_split():
    body = "Overview\n\n## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2"
    sections = _split_on_h2(body)
    assert sections[0][0] == "Overview"
    assert sections[1][0] == "Section 1"
    assert sections[2][0] == "Section 2"

def test_code_flag():
    assert _is_code_chunk("Text\n```python\ncode\n```\nMore")
    assert not _is_code_chunk("Just text")

def test_hard_split():
    long = "a " * 5000  # ~10000 chars
    parts = _hard_split(long, 8000)
    assert len(parts) == 2
    assert all(len(p) <= 8000 for p in parts)

def test_chunk_file(temp_markdown_repo):
    chunks = chunk_file(temp_markdown_repo / "markdown" / "pub-one" / "intro.md", temp_markdown_repo)
    assert len(chunks) == 3  # Overview + 2 H2 sections
    assert chunks[0]["heading"] == "Overview"
    assert chunks[1]["heading"] == "Getting Started"
    assert chunks[2]["heading"] == "Advanced Topics"
    assert all(c["publication"] == "pub-one" for c in chunks)
    assert all(c["file"] == "intro.md" for c in chunks)
```

## Test Embed (`test_embed.py`)

```python
def test_onnx_embedder_shape():
    embedder = OnnxEmbedder("Xenova/bge-base-en-v1.5", prefer_gpu=False)
    out = embedder.embed(["hello", "world"])
    assert out.shape == (2, 768)
    assert out.dtype == np.float32
    # L2 normalized
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

@pytest.mark.skipif(not has_mlx() or not is_apple_silicon(), reason="MLX requires Apple Silicon")
def test_mlx_embedder_shape():
    embedder = MlxEmbedder("Xenova/bge-base-en-v1.5")
    out = embedder.embed(["hello", "world"])
    assert out.shape == (2, 768)
    assert out.dtype == np.float32

@pytest.mark.skipif(not has_onnxruntime_gpu(), reason="CUDA provider not available")
def test_onnx_gpu_embedder_shape():
    embedder = OnnxEmbedder("Xenova/bge-base-en-v1.5", prefer_gpu=True)
    out = embedder.embed(["hello", "world"])
    assert out.shape == (2, 768)

def test_resolve_backend_auto():
    backend, reason = resolve_backend("auto")
    assert backend in ("mlx", "onnx-gpu", "onnx-cpu")
    assert isinstance(reason, str) and len(reason) > 0

def test_resolve_backend_override():
    assert resolve_backend("cpu")[0] == "onnx-cpu"
    assert resolve_backend("onnx-cpu")[0] == "onnx-cpu"

def test_embed_with_progress():
    embedder = OnnxEmbedder("Xenova/bge-base-en-v1.5", prefer_gpu=False)
    texts = ["text " * 100] * 10  # 10 chunks
    out = embedder.embed_with_progress(texts, batch_size=4, show_progress=False)
    assert out.shape == (10, 768)
```

## Test FS Server (`test_fs_server.py`)

```python
@pytest.fixture
def fs_server(temp_markdown_repo):
    from atlas.fs_server import app
    # Create a test server instance with the temp repo
    server = create_test_server(temp_markdown_repo)
    return server

async def test_list_publications(fs_server):
    result = await fs_server.call_tool("list_publications", {})
    data = json.loads(result[0].text)
    assert len(data) == 2
    pub_names = {p["name"] for p in data}
    assert pub_names == {"pub-one", "pub-two"}

async def test_list_files(fs_server):
    result = await fs_server.call_tool("list_files", {"publication": "pub-one"})
    data = json.loads(result[0].text)
    assert len(data) == 1
    assert data[0]["file"] == "intro.md"
    assert data[0]["title"] == "Introduction"
    assert data[0]["product_area"] == "Core"

async def test_read_file(fs_server):
    result = await fs_server.call_tool("read_file", {"publication": "pub-one", "file": "intro.md"})
    data = json.loads(result[0].text)
    assert data["frontmatter"]["title"] == "Introduction"
    assert "Overview text" in data["content"]
    assert "Getting Started" in data["content"]

async def test_search(fs_server):
    result = await fs_server.call_tool("search", {"query": "installation"})
    data = json.loads(result[0].text)
    assert len(data) == 1
    assert "installation" in data[0]["preview"].lower()

async def test_search_scoped(fs_server):
    result = await fs_server.call_tool("search", {"query": "guide", "scope": "pub-two"})
    data = json.loads(result[0].text)
    assert len(data) == 1

async def test_get_release_info(fs_server):
    result = await fs_server.call_tool("get_release_info", {})
    data = json.loads(result[0].text)
    assert "branch" in data
    assert "sha" in data
    assert data["file_count"] >= 2

async def test_path_traversal_blocked(fs_server):
    result = await fs_server.call_tool("read_file", {"publication": "pub-one", "file": "../../../etc/passwd"})
    data = json.loads(result[0].text)
    assert "error" in data
```

## Test RAG Server (`test_rag_server.py`)

```python
@pytest.fixture
def rag_server(test_bundle):
    from atlas.rag_server import app
    server = create_test_server(test_bundle)
    return server

async def test_search_docs(rag_server):
    result = await rag_server.call_tool("search_docs", {"query": "getting started", "top_k": 5})
    data = json.loads(result[0].text)
    assert len(data) <= 5
    assert all("score" in r for r in data)
    assert all("text" in r for r in data)
    assert all("publication" in r for r in data)

async def test_search_code(rag_server):
    result = await rag_server.call_tool("search_code", {"query": "pip install", "top_k": 5})
    data = json.loads(result[0].text)
    assert len(data) >= 1
    assert all(r["is_code"] for r in data)

async def test_get_chunk(rag_server):
    # First search to get a chunk ID
    search_result = await rag_server.call_tool("search_docs", {"query": "test", "top_k": 1})
    search_data = json.loads(search_result[0].text)
    chunk_id = search_data[0]["id"]
    
    result = await rag_server.call_tool("get_chunk", {"chunk_id": chunk_id})
    data = json.loads(result[0].text)
    assert data["id"] == chunk_id
    assert "text" in data

async def test_get_bundle_info(rag_server):
    result = await rag_server.call_tool("get_bundle_info", {})
    data = json.loads(result[0].text)
    assert data["chunk_count"] > 0
    assert data["embedding_model"] == "Xenova/bge-base-en-v1.5"
    assert data["embedding_dim"] == 768

async def test_search_filters(rag_server):
    result = await rag_server.call_tool("search_docs", {
        "query": "test",
        "publication": "pub-one",
        "product_area": "Core",
        "is_code": False,
        "min_score": 0.1
    })
    data = json.loads(result[0].text)
    for r in data:
        assert r["publication"] == "pub-one"
        assert r["product_area"] == "Core"
        assert r["is_code"] == False
        assert r["score"] >= 0.1

async def test_search_modes(rag_server):
    for mode in ["vector", "hybrid", "keyword"]:
        result = await rag_server.call_tool("search_docs", {"query": "test", "mode": mode, "top_k": 3})
        data = json.loads(result[0].text)
        assert len(data) <= 3
```

## Test Make Bundle (`test_make_bundle.py`)

```python
def test_build_chunk_table(temp_markdown_repo):
    files = walk_markdown(temp_markdown_repo)
    table = build_chunk_table(files, temp_markdown_repo)
    assert len(table) == 5  # 3 chunks from intro.md + 2 from guide.md
    assert set(table.column_names) == {"id", "text", "publication", "file", "heading", "is_code", "title", "product_area", "last_updated", "canonical_url"}

def test_ensure_repo_clone(tmp_path):
    # Uses a tiny test repo or mocks git
    repo_path = ensure_repo(tmp_path / "clone", "https://github.com/test/repo.git", "main")
    assert (repo_path / ".git").is_dir()

def test_write_manifest(test_bundle):
    manifest_path = test_bundle / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert "chunks_sha256" in manifest["artifacts"]

def test_sha256_file(test_bundle):
    sha = sha256_file(test_bundle / "chunks.parquet")
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

def test_full_build_smoke(temp_markdown_repo, tmp_path):
    """End-to-end build with --skip-embed (fast)."""
    output = tmp_path / "bundle"
    result = subprocess.run([
        "atlas-build",
        "--repo-path", str(temp_markdown_repo),
        "--output", str(output),
        "--skip-embed"
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert (output / "manifest.json").is_file()
    assert (output / "chunks.parquet").is_file()
```

## Test Download (`test_download.py`)

```python
def test_resolve_release(mock_github_api):
    release = resolve_release("test/repo", None)
    assert release["tag_name"] == "v1.0.0"

def test_find_bundle_asset():
    release = {"assets": [{"name": "bundle.tar.zst"}, {"name": "notes.txt"}]}
    asset = find_bundle_asset(release)
    assert asset["name"] == "bundle.tar.zst"

def test_extract_tar_gz(tmp_path):
    import tarfile
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add("test_file = tf.gettarinfo("test.txt")
        tf.addfile(tf_info, io.BytesIO(b"content"))
    
    target = tmp_path / "out"
    _extract(archive, target)
    assert (target / "test.txt").read_text() == "content"

def test_download_verify_sha(tmp_path, mock_github_api, monkeypatch):
    # Mock _http_download to write a known file
    # Mock _http_json to return release with asset
    # Verify SHA check passes/fails correctly
    pass
```

## Smoke Test (`atlas/smoke_test.py`)

```bash
# Run manually
atlas-smoke --bundle ./data/rag-bundle --repo ./data/servicenow-docs/ServiceNowDocs-australia

# What it tests:
# 1. FS server: list_publications, list_files, read_file, search, get_release_info
# 2. RAG server: search_docs, search_code, get_chunk, get_bundle_info
# 3. Both servers return valid JSON, no errors
# 4. Exit code 0 = all pass, non-zero = any failure
```

## Evaluation (`atlas/evaluate.py`)

```bash
# Run evaluation against golden set
atlas-evaluate --bundle ./data/rag-bundle --golden ./eval/golden.jsonl

# Golden set format (JSONL):
{"query": "how to create incident", "expected_files": ["it-service-management/incident-management.md"]}
{"query": "glideRecord query", "expected_files": ["scripting/glide-record.md", "scripting/glide-aggregate.md"]}

# Metrics:
# - Precision@10: fraction of top-10 results that are in expected_files
# - MRR: mean reciprocal rank of first expected file
# - Outputs mean ± std over all queries
```

## CI Integration

```yaml
# .github/workflows/build-bundle.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Sync deps
        run: uv sync --extra build --extra dev
      - name: Run unit tests
        run: uv run pytest tests/ -v --tb=short
      - name: Smoke test
        run: |
          # Create minimal test repo
          mkdir -p test-repo/markdown/test
          echo -e '---\ntitle: Test\n---\n## Hello\nWorld' > test-repo/markdown/test/doc.md
          # Build test bundle
          atlas-build --repo-path test-repo --output test-bundle --skip-embed
          # Run smoke test
          atlas-smoke --bundle test-bundle --repo test-repo
```

## Debugging Test Failures

| Failure | Debug Command |
|---------|---------------|
| Chunker produces wrong chunks | `python -m atlas.chunk < test.md` |
| Embedder shape mismatch | `uv run pytest tests/test_embed.py::test_onnx_embedder_shape -v -s` |
| FS server tool error | `atlas-fs --repo ./test-repo` then call tools via MCP client |
| RAG server no results | `atlas-doctor --bundle ./test-bundle` |
| Bundle SHA mismatch | `sha256sum bundle/chunks.parquet` vs `cat bundle/manifest.json | jq .artifacts.chunks_sha256` |

## Test Data Management

- **No committed test bundles** — generated on-the-fly in `tmp_path`
- **No network in unit tests** — mock GitHub API, HF model downloads
- **Real embeddings only in integration tests** — use `--skip-embed` for speed
- **Golden eval set** — committed in `eval/golden.jsonl` (small, curated)