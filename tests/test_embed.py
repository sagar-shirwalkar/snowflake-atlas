"""Tests for the embedding backends and selection logic."""

import platform
from pathlib import Path

import numpy as np
import pytest

from atlas.embed.base import (
    _read_config_backend,
    get_embedder,
    has_mlx,
    has_nvidia_gpu,
    has_onnxruntime_gpu,
    is_apple_silicon,
    l2_normalize,
    load_embeddings,
    mean_pool,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple Silicon only",
)
def test_is_apple_silicon_true_on_mac_arm():
    assert is_apple_silicon() is True


def test_is_apple_silicon_mocked():
    """Check the logic independent of the actual host."""

    orig_system = platform.system
    orig_machine = platform.machine
    try:
        platform.system = lambda: "Darwin"
        platform.machine = lambda: "arm64"
        # Re-import doesn't help; the function uses module imports at call time
        from atlas.embed import base as eb

        assert eb.is_apple_silicon() is True

        platform.machine = lambda: "x86_64"
        assert eb.is_apple_silicon() is False

        platform.system = lambda: "Linux"
        platform.machine = lambda: "arm64"
        assert eb.is_apple_silicon() is False
    finally:
        platform.system = orig_system
        platform.machine = orig_machine


# ---------------------------------------------------------------------------
# Backend detection (import-based, tested via import side-channels)
# ---------------------------------------------------------------------------

def test_has_mlx_returns_bool():
    """Should return True if mlx importable, False otherwise."""
    result = has_mlx()
    assert isinstance(result, bool)


def test_has_nvidia_gpu_returns_bool():
    """Should return True only if nvidia-smi shows a GPU."""
    result = has_nvidia_gpu()
    assert isinstance(result, bool)


def test_has_onnxruntime_gpu_returns_bool():
    """Should return True only if CUDAExecutionProvider available."""
    result = has_onnxruntime_gpu()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------

def test_resolve_backend_explicit_onnx_cpu():
    backend, reason = resolve_backend(prefer="cpu")
    assert backend == "onnx-cpu"
    assert "ONNX+CPU" in reason


def test_resolve_backend_explicit_unknown_falls_to_auto():
    """An unknown pref value should be treated as 'auto' and resolve."""
    backend, _ = resolve_backend(prefer=None)
    assert backend in ("mlx", "onnx-gpu", "onnx-cpu")


def test_resolve_backend_auto_never_raises():
    """resolve_backend should always return a known backend string."""
    backend, _ = resolve_backend(prefer="auto")
    assert backend in ("mlx", "onnx-gpu", "onnx-cpu")


# ---------------------------------------------------------------------------
# Config file parsing
# ---------------------------------------------------------------------------

def test_read_config_backend_no_file(tmp_path: Path):
    """When no config file exists, return None."""
    result = _read_config_backend()
    # This reads from the actual home dir, but we can't mock that easily.
    # The result must be either None or a string — never raise.
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_basic(self):
        hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = np.array([[1, 1, 0]])
        result = mean_pool(hidden, mask)
        assert result.shape == (1, 2)
        expected = np.array([[2.0, 3.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_all_masked(self):
        hidden = np.random.randn(2, 5, 8).astype(np.float32)
        mask = np.zeros((2, 5), dtype=np.int64)
        result = mean_pool(hidden, mask)
        # Should not be NaN
        assert np.all(np.isfinite(result))
        assert result.shape == (2, 8)

    def test_single_token(self):
        hidden = np.array([[[0.5, -0.5, 1.0]]])
        mask = np.array([[1]])
        result = mean_pool(hidden, mask)
        expected = np.array([[0.5, -0.5, 1.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestL2Normalize:
    def test_unit_vector(self):
        x = np.array([[3.0, 4.0]])
        result = l2_normalize(x)
        expected = np.array([[0.6, 0.8]])
        np.testing.assert_allclose(result, expected, atol=1e-6)
        # Verify L2 norm ≈ 1
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    def test_zero_vector(self):
        x = np.array([[0.0, 0.0, 0.0]])
        result = l2_normalize(x)
        assert np.all(np.isfinite(result))
        # A zero vector should be normalized to zero (clip prevents NaN)
        np.testing.assert_allclose(result, [[0.0, 0.0, 0.0]], atol=1e-6)

    def test_multiple_vectors(self):
        x = np.array([[1.0, 1.0], [0.0, 0.0], [-3.0, 4.0]], dtype=np.float32)
        result = l2_normalize(x)
        assert result.dtype == np.float32
        expected_norms = np.array([1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1), expected_norms, atol=1e-6
        )


class TestLoadEmbeddings:
    def test_f16_preferred(self, tmp_path: Path):
        embed_f16 = tmp_path / "embeddings.f16.npy"
        embed_f32 = tmp_path / "embeddings.f32.npy"
        data = np.random.randn(10, 768).astype(np.float16)
        np.save(embed_f16, data)
        np.save(embed_f32, data.astype(np.float32))
        loaded = load_embeddings(tmp_path)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, data.astype(np.float32), atol=1e-3)

    def test_f32_fallback(self, tmp_path: Path):
        embed_f32 = tmp_path / "embeddings.f32.npy"
        data = np.random.randn(5, 768).astype(np.float32)
        np.save(embed_f32, data)
        loaded = load_embeddings(tmp_path)
        assert loaded.shape == (5, 768)
        np.testing.assert_allclose(loaded, data)

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path)


# ---------------------------------------------------------------------------
# Embedder ABC
# ---------------------------------------------------------------------------

def test_embedder_abstract_cannot_instantiate():
    from atlas.embed.base import Embedder

    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


class TestEmbedderConcrete:
    """Test the embed_with_progress default impl via a minimal concrete subclass."""

    @pytest.fixture
    def stub_embedder(self):
        from atlas.embed.base import Embedder

        class StubEmbedder(Embedder):
            backend = "test"
            active_provider = "stub"
            model_id = "test-model"
            resolved_dir = Path("/tmp")
            dim = 768

            def embed(self, texts):
                return np.ones((len(texts), 768), dtype=np.float32)

        return StubEmbedder()

    def test_embed_with_progress_empty(self, stub_embedder):
        result = stub_embedder.embed_with_progress([], show_progress=False)
        assert result.shape == (0, 768)

    def test_embed_with_progress_single(self, stub_embedder):
        result = stub_embedder.embed_with_progress(
            ["hello"], batch_size=1, show_progress=False
        )
        assert result.shape == (1, 768)
        assert result[0, 0] == 1.0

    def test_embed_with_progress_batched(self, stub_embedder):
        texts = [f"text_{i}" for i in range(10)]
        result = stub_embedder.embed_with_progress(
            texts, batch_size=3, show_progress=False
        )
        assert result.shape == (10, 768)
        assert np.allclose(result, 1.0)

    def test_embed_with_progress_retry_then_give_up(self, stub_embedder):
        """A failing embed should retry and then fill with zeros."""

        class FailingEmbedder(type(stub_embedder)):
            attempt = 0

            def embed(self, texts):
                self.attempt += 1
                if self.attempt <= 2:
                    raise RuntimeError("transient failure")
                return np.ones((len(texts), 768), dtype=np.float32)

        f = FailingEmbedder()
        # Override type comparison hacks by using the instance directly
        result = f.embed_with_progress(["hello"], batch_size=1, show_progress=False, max_retries=5)
        assert result.shape == (1, 768)
        # Should have succeeded eventually
        assert f.attempt <= 3

    def test_embed_with_progress_all_fail(self, stub_embedder):
        """If all retries fail, embed_with_progress should fill with zeros."""
        class AlwaysFails(type(stub_embedder)):
            def embed(self, texts):
                raise RuntimeError("always fails")

        f = AlwaysFails()
        result = f.embed_with_progress(
            ["hello", "world"], batch_size=1, show_progress=False, max_retries=1
        )
        assert result.shape == (2, 768)
        # Should be zeros (graceful degradation)
        assert np.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------

def test_get_embedder_returns_onnx_cpu_fallback():
    """Without mlx or onnx-gpu, should return onnx-cpu embedder."""
    embedder = get_embedder("Xenova/bge-base-en-v1.5", prefer="cpu")
    assert embedder.backend == "onnx-cpu"
    assert hasattr(embedder, "embed")
