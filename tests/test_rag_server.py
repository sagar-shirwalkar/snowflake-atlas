"""Tests for the RAG MCP server (atlas/rag_server.py)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atlas.rag_server import Bundle, _bundle_cache

# ---------------------------------------------------------------------------
# Stub embedder that doesn't need a real ONNX model
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Duck-typed embedder that returns random unit vectors."""
    backend = "test"
    active_provider = "stub"
    model_id = "test-model"
    resolved_dir = Path("/tmp")
    dim = 768

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.random.RandomState(42).randn(len(texts), 768).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out / norms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Create a minimal bundle directory with a few chunks."""
    bdir = tmp_path / "bundle"
    bdir.mkdir()

    manifest = {
        "schema_version": 1,
        "source_repo": "https://docs.snowflake.com",
        "source_branch": "crawl-2025-01-01",
        "source_sha": "abc123",
        "built_at": "2025-01-01T00:00:00Z",
        "chunk_count": 4,
        "embedding_model": "Xenova/bge-base-en-v1.5",
        "embedding_dim": 768,
    }
    (bdir / "manifest.json").write_text(json.dumps(manifest))

    table = pa.table(
        {
            "id": ["chunk-0", "chunk-1", "chunk-2", "chunk-3"],
            "text": [
                "Welcome to Snowflake documentation.",
                "CREATE TABLE users (id INT, name TEXT)",
                "Snowflake supports semi-structured data like JSON.",
                "SELECT * FROM orders WHERE status = 'active'",
            ],
            "publication": ["user-guide", "sql-reference", "user-guide", "sql-reference"],
            "file": ["intro.md", "commands/create-table.md", "semi-structured.md", "commands/select.md"],
            "heading": ["Overview", "Syntax", "Overview", "Examples"],
            "is_code": [False, True, False, True],
            "title": ["Introduction", "CREATE TABLE", "Semi-Structured Data", "SELECT"],
            "product_area": ["Core", "SQL", "Core", "SQL"],
            "last_updated": ["", "", "", ""],
            "canonical_url": [
                "https://docs.snowflake.com/en/user-guide/intro",
                "https://docs.snowflake.com/en/sql-reference/create-table",
                "https://docs.snowflake.com/en/user-guide/semi-structured",
                "https://docs.snowflake.com/en/sql-reference/select",
            ],
        }
    )
    pq.write_table(table, bdir / "chunks.parquet")

    rng = np.random.RandomState(42)
    embeddings = rng.randn(4, 768).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    np.save(bdir / "embeddings.f16.npy", embeddings.astype(np.float16))
    np.save(bdir / "norms.f32.npy", norms.flatten().astype(np.float32))

    return bdir


@pytest.fixture
def bundle(bundle_dir: Path) -> Bundle:
    """Create a Bundle with a stubbed embedder (avoids real ONNX model)."""
    with patch("atlas.rag_server.get_embedder", return_value=StubEmbedder()):
        yield Bundle(bundle_dir, prefer="cpu")


@pytest.fixture
def bundle_real_embed(bundle_dir: Path) -> Bundle:
    """Create a Bundle with the real embedder (will download model)."""
    return Bundle(bundle_dir, prefer="cpu")


# ---------------------------------------------------------------------------
# Bundle.__init__
# ---------------------------------------------------------------------------

class TestBundleInit:
    def test_missing_manifest(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Bundle manifest missing"):
            Bundle(tmp_path)

    def test_missing_chunks(self, tmp_path: Path):
        bdir = tmp_path / "bundle"
        bdir.mkdir()
        (bdir / "manifest.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="Bundle chunks missing"):
            Bundle(bdir)

    def test_loaded_correctly(self, bundle):
        assert bundle._n == 4
        assert bundle.manifest["chunk_count"] == 4

    def test_embeddings_loaded(self, bundle):
        assert bundle.embeddings.shape == (4, 768)
        assert bundle.embeddings.dtype == np.float32


# ---------------------------------------------------------------------------
# Bundle.search
# ---------------------------------------------------------------------------

class TestBundleSearch:
    def test_basic_search(self, bundle):
        results = bundle.search("Snowflake documentation", top_k=2)
        assert len(results) == 2
        assert all(r["score"] >= -1.0 for r in results)
        assert all(r["score"] <= 1.0 for r in results)

    def test_search_publication_filter(self, bundle):
        results = bundle.search("table", top_k=5, publication="sql-reference")
        assert len(results) >= 1
        assert all(r["publication"] == "sql-reference" for r in results)

    def test_search_product_area_filter(self, bundle):
        results = bundle.search("data", top_k=5, product_area="Core")
        assert all(r["product_area"] == "Core" for r in results)

    def test_search_code_filter(self, bundle):
        results = bundle.search("SELECT", top_k=5, is_code=True)
        assert all(r["is_code"] is True for r in results)

    def test_search_non_code_filter(self, bundle):
        results = bundle.search("Snowflake", top_k=5, is_code=False)
        assert all(r["is_code"] is False for r in results)

    def test_search_empty_query(self, bundle):
        results = bundle.search("", top_k=3)
        assert len(results) == 3

    def test_search_mode_vector(self, bundle):
        results = bundle.search("create table", top_k=5, mode="vector")
        assert len(results) >= 1

    def test_search_mode_keyword(self, bundle):
        results = bundle.search("CREATE TABLE", top_k=5, mode="keyword")
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("CREATE TABLE" in t for t in texts)

    def test_search_mode_hybrid(self, bundle):
        results = bundle.search("create table", top_k=5, mode="hybrid")
        assert len(results) >= 1

    def test_search_min_score_filter(self, bundle):
        results = bundle.search("xyznonexistentquery", top_k=5, min_score=0.9)
        assert results == []

    def test_search_candidate_k(self, bundle):
        results = bundle.search("data", top_k=2, candidate_k=2)
        assert len(results) == 2

    def test_search_no_results_for_exclusion(self, bundle):
        results = bundle.search("Snowflake", top_k=5, publication="nonexistent")
        assert results == []

    def test_search_result_structure(self, bundle):
        results = bundle.search("documentation", top_k=1)
        assert len(results) == 1
        r = results[0]
        expected_keys = {
            "id", "score", "publication", "file", "heading",
            "title", "product_area", "last_updated", "canonical_url",
            "is_code", "text",
        }
        assert set(r.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Bundle.get_chunk
# ---------------------------------------------------------------------------

class TestBundleGetChunk:
    def test_get_existing_chunk(self, bundle):
        chunk = bundle.get_chunk("chunk-0")
        assert chunk is not None
        assert chunk["id"] == "chunk-0"
        assert "Snowflake" in chunk["text"]

    def test_get_nonexistent_chunk(self, bundle):
        chunk = bundle.get_chunk("nonexistent-id")
        assert chunk is None

    def test_chunk_has_all_keys(self, bundle):
        chunk = bundle.get_chunk("chunk-1")
        expected_keys = {
            "id", "publication", "file", "heading",
            "title", "product_area", "last_updated", "canonical_url",
            "is_code", "text",
        }
        assert set(chunk.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Bundle._title_boost
# ---------------------------------------------------------------------------

class TestTitleBoost:
    def test_boost_matching_titles(self, bundle):
        titles = pa.array(["CREATE TABLE", "SELECT", "Introduction"])
        # pc.match_substring is case-sensitive; use exact-case tokens
        tokens = {"TABLE", "SELECT"}
        boost = Bundle._title_boost(titles, tokens)
        assert boost[0] > 0.0  # "CREATE TABLE" matches "TABLE"
        assert boost[1] > 0.0  # "SELECT" matches "SELECT"
        assert boost[2] == 0.0  # "Introduction" doesn't match either

    def test_boost_empty_tokens(self, bundle):
        titles = pa.array(["a", "b"])
        boost = Bundle._title_boost(titles, set())
        assert np.allclose(boost, 0.0)

    def test_boost_empty_titles(self, bundle):
        boost = Bundle._title_boost(pa.array([], type=pa.string()), {"test"})
        assert boost.shape == (0,)


# ---------------------------------------------------------------------------
# _bundle_cache lazy initialization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bundle_cache_lazy(bundle_dir: Path):
    """The cache should load a bundle on first access."""
    import atlas.rag_server as rag

    rag._bundle_instance = None
    rag._bundle_lock = __import__("asyncio").Lock()
    with patch("atlas.rag_server.get_embedder", return_value=StubEmbedder()):
        bundle = await _bundle_cache(str(bundle_dir), "cpu")
    assert isinstance(bundle, Bundle)
    assert bundle._n == 4

    # Second call should return cached instance
    bundle2 = await _bundle_cache(str(bundle_dir), "cpu")
    assert bundle2 is bundle
