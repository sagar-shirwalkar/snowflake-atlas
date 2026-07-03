"""MLX cross-encoder re-ranker (bge-reranker-v2-base).

Provides :class:`MlxCrossEncoderReranker` which re-scores top-k
results using a BERT-based cross-encoder accelerated via MLX on
Apple Silicon.

Model architecture: BERT-base (12 layers, 768 hidden, 12 heads) +
a linear classification head on the ``[CLS]`` token.  Same encoder
architecture as the embedder, making weight conversion straightforward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .embed.mlx import BgeModel, _import_mlx, _load_mlx_weights

_DEFAULT_CACHE = Path.home() / ".cache" / "atlas" / "models" / "bge-reranker-v2-base-mlx"
_MAX_SEQ = 512
_DIM = 768


class BertClassifier:
    """BERT encoder + a linear classification head on the [CLS] token.

    Architecture matches Hugging Face ``AutoModelForSequenceClassification``
    with ``num_labels=1``:

      1. BERT encoder produces hidden states (same as BgeModel).
      2. ``[CLS]`` token (index 0) is extracted.
      3. A single linear layer projects ``(768,) → (1,)`` — a logit.

    The logit can be interpreted as an unnormalised relevance score
    (higher = more relevant).
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        dim: int = _DIM,
        n_layers: int = 12,
        n_heads: int = 12,
        ff_dim: int = 3072,
        max_seq: int = _MAX_SEQ,
    ) -> None:
        """Initialize the BERT classifier with configurable architecture."""
        _, nn = _import_mlx()
        self._dim = dim
        self.encoder = BgeModel(vocab_size, dim, n_layers, n_heads, ff_dim, max_seq)
        self.classifier = nn.Linear(dim, 1)

    def __call__(self, input_ids, attention_mask, token_type_ids=None) -> Any:
        """Forward pass.  Returns logits, shape ``(batch, 1)``."""
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        cls_token = hidden[:, 0, :]  # (batch, dim)
        return self.classifier(cls_token)  # (batch, 1)


def _load_reranker_weights(model: BertClassifier, weights_path: Path) -> None:
    """Load MLX-format weights into a BertClassifier.

    The weight directory must contain:

    - BERT encoder weights (same layout as ``BgeModel``) — 197 ``.npy``
      files with keys like ``encoder.layer.0.attention.q_proj.weight`` etc.
    - ``classifier.weight.npy`` — shape ``(1, 768)``
    - ``classifier.bias.npy`` — shape ``(1,)``
    """
    mx, _ = _import_mlx()

    def _npy(name: str) -> mx.array:
        arr = np.load(weights_path / f"{name}.npy")
        return mx.array(arr)

    # Load BERT encoder weights (delegates to BgeModel's loader)
    _load_mlx_weights(model.encoder, weights_path)

    # Load classification head
    model.classifier.weight = _npy("classifier.weight")
    model.classifier.bias = _npy("classifier.bias")


class MlxCrossEncoderReranker:
    """MLX-accelerated cross-encoder re-ranker for Apple Silicon.

    Loads a BERT-based cross-encoder (``BAAI/bge-reranker-v2-base``)
    converted to MLX weight files.  Scores each ``(query, doc)`` pair
    independently and re-ranks by the resulting logit.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Initialize the MLX cross-encoder reranker.

        Args:
            model_dir: Local directory with MLX weight files. If None,
                       resolves via ``DEFAULT_MLX_CACHE``.

        """
        mx, _ = _import_mlx()

        # Resolve weight directory
        if model_dir is not None:
            weights_path = Path(model_dir)
        elif _DEFAULT_CACHE.is_dir() and (_DEFAULT_CACHE / "classifier.weight.npy").is_file():
            weights_path = _DEFAULT_CACHE
        else:
            raise FileNotFoundError(
                f"MLX reranker weights not found. Looked in {_DEFAULT_CACHE}. "
                f"Run `uv run python tools/convert_reranker_to_mlx.py` to convert "
                f"the Hugging Face checkpoint first."
            )

        self.weights_path = weights_path
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-base")
        self.model = BertClassifier()
        _load_reranker_weights(self.model, weights_path)
        self._warmup_done = False

    def _ensure_warmup(self) -> None:
        if self._warmup_done:
            return
        mx, _ = _import_mlx()
        _ = self._score(["warmup query"], ["warmup passage"])
        mx.eval(_)
        self._warmup_done = True

    def _score(
        self, queries: list[str], docs: list[str]
    ) -> Any:
        """Score ``(query, doc)`` pairs.  Returns logits as an MLX array."""
        mx, _ = _import_mlx()
        encoded = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=_MAX_SEQ,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"].astype(np.int32))
        attention_mask = mx.array(encoded["attention_mask"].astype(np.int32))
        token_type_ids = mx.array(encoded["token_type_ids"].astype(np.int32))

        logits = self.model(input_ids, attention_mask, token_type_ids)
        return logits

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank results using cross-encoder logits.

        Args:
            query: The original search query.
            results: List of result dicts from vector search (must have
                     ``'text'`` field).
            top_k: Number of results to return after re-ranking.

        Returns:
            Re-ranked list of results with updated ``'score'`` field
            (cross-encoder logit) and ``'reranked': True``.

        """
        if not results:
            return results

        self._ensure_warmup()

        # Batch inference over all (query, doc) pairs
        queries = [query] * len(results)
        docs = [r["text"] for r in results]
        logits = self._score(queries, docs)

        mx, _ = _import_mlx()
        mx.eval(logits)
        logit_vals = np.array(logits).flatten()  # (n_results,)

        # Attach scores and sort
        scored = list(zip(results, logit_vals, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            result = result.copy()
            result["score"] = float(score)
            result["reranked"] = True
            reranked.append(result)

        return reranked
