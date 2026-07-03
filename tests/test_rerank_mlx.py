"""Tests for the MLX cross-encoder reranker (atlas/rerank_mlx.py).

These tests require MLX (Apple Silicon). On non-Apple hardware, the tests
are skipped with an appropriate message.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check if MLX is available on this platform
try:
    from atlas.embed.mlx import _import_mlx

    _import_mlx()
    HAS_MLX = True
except (ImportError, RuntimeError):
    HAS_MLX = False


pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available on this platform")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """Create a temporary weights directory with minimal MLX weight files."""
    wdir = tmp_path / "weights"
    wdir.mkdir()
    # MLX reranker needs at least classifier.weight and classifier.bias
    np.save(wdir / "classifier.weight.npy", np.zeros((1, 768), dtype=np.float32))
    np.save(wdir / "classifier.bias.npy", np.zeros(1, dtype=np.float32))
    return wdir


# ---------------------------------------------------------------------------
# Tests for BertClassifier
# ---------------------------------------------------------------------------

class TestBertClassifier:
    def test_import(self):
        """BertClassifier should import correctly when MLX is available."""
        from atlas.rerank_mlx import BertClassifier

        model = BertClassifier(vocab_size=100, dim=64, n_layers=2, n_heads=2, ff_dim=128, max_seq=64)
        assert model._dim == 64
        assert model.classifier is not None

    def test_call_shape(self):
        """Forward pass should return (batch, 1) logits."""
        import mlx.core as mx

        from atlas.rerank_mlx import BertClassifier

        model = BertClassifier(vocab_size=100, dim=64, n_layers=2, n_heads=2, ff_dim=128, max_seq=64)
        input_ids = mx.ones((3, 16), dtype=mx.int32)
        attention_mask = mx.ones((3, 16), dtype=mx.int32)
        logits = model(input_ids, attention_mask)
        assert logits.shape == (3, 1)


# ---------------------------------------------------------------------------
# Tests for MlxCrossEncoderReranker
# ---------------------------------------------------------------------------

class TestMlxCrossEncoderReranker:
    def test_init_no_weights(self):
        """Raises FileNotFoundError when no weight directory is given and defaults missing."""
        from atlas.rerank_mlx import MlxCrossEncoderReranker

        with pytest.raises(FileNotFoundError, match="MLX reranker weights not found"):
            MlxCrossEncoderReranker(model_dir="/nonexistent/path")

    def test_init_with_weights_dir(self, weights_dir):
        """Should initialise when weights exist."""
        from atlas.rerank_mlx import MlxCrossEncoderReranker

        with patch("atlas.rerank_mlx.AutoTokenizer") as mock_tok:
            tokenizer = MagicMock()
            mock_tok.from_pretrained.return_value = tokenizer
            reranker = MlxCrossEncoderReranker(model_dir=str(weights_dir))
            assert reranker.weights_path == weights_dir

    def test_rerank_empty_results(self, weights_dir):
        """rerank([]) should return empty list."""
        from atlas.rerank_mlx import MlxCrossEncoderReranker

        with patch("atlas.rerank_mlx.AutoTokenizer") as mock_tok:
            tokenizer = MagicMock()
            mock_tok.from_pretrained.return_value = tokenizer
            reranker = MlxCrossEncoderReranker(model_dir=str(weights_dir))
            result = reranker.rerank("test query", [], top_k=5)
            assert result == []

    def test_rerank_maintains_result_keys(self, weights_dir):
        """Reranked results should keep all original keys plus reranked flag."""
        from atlas.rerank_mlx import MlxCrossEncoderReranker

        results = [
            {"id": "1", "text": "doc one", "score": 0.5},
            {"id": "2", "text": "doc two", "score": 0.4},
        ]

        with patch("atlas.rerank_mlx.AutoTokenizer") as mock_tok:
            tokenizer = MagicMock()
            mock_tok.from_pretrained.return_value = tokenizer
            reranker = MlxCrossEncoderReranker(model_dir=str(weights_dir))
            reranked = reranker.rerank("query", results, top_k=2)
            assert len(reranked) == 2
            assert reranked[0]["reranked"] is True
            assert "text" in reranked[0]
            assert "id" in reranked[0]

    def test_rerank_top_k_limits(self, weights_dir):
        """Rerank should respect top_k limit."""
        from atlas.rerank_mlx import MlxCrossEncoderReranker

        results = [{"id": str(i), "text": f"doc {i}", "score": 0.5} for i in range(10)]

        with patch("atlas.rerank_mlx.AutoTokenizer") as mock_tok:
            tokenizer = MagicMock()
            mock_tok.from_pretrained.return_value = tokenizer
            reranker = MlxCrossEncoderReranker(model_dir=str(weights_dir))
            reranked = reranker.rerank("query", results, top_k=3)
            assert len(reranked) == 3


# ---------------------------------------------------------------------------
# Import-level test: module importable on any platform
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HAS_MLX, reason="Only test on non-MLX platforms")
def test_module_importable_without_mlx():
    """atlas.rerank_mlx should be importable even without MLX (ImportError guard)."""
    # The ImportError guard at module level should gracefully produce a
    # module that at least exposes the class names
    import atlas.rerank_mlx

    assert hasattr(atlas.rerank_mlx, "MlxCrossEncoderReranker")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
