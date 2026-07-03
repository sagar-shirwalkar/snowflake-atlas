"""ONNX Runtime embedding backend (CPU + CUDA).

Provides :class:`OnnxEmbedder` which loads a BGE-base model exported
to ONNX format and runs inference via ONNX Runtime. Supports both
CPU and CUDA execution providers.

The model is expected to be in the Xenova layout:
- ``onnx/model.onnx`` — the ONNX graph
- ``tokenizer.json``, ``tokenizer_config.json``, etc. — tokenizer files

At runtime, the embedder can load from a local directory (the bundle's
``model/`` dir) or from the Hugging Face cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .base import (
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    Embedder,
    l2_normalize,
    mean_pool,
)


class OnnxEmbedder(Embedder):
    """ONNX Runtime embedder for BGE-base (CPU or CUDA)."""

    backend = "onnx-cpu"  # overridden in __init__ if GPU

    def __init__(
        self,
        model_id: str | Path,
        prefer_gpu: bool = False,
    ) -> None:
        """Initialize the ONNX embedder.

        Args:
            model_id: Local directory path or Hugging Face model ID.
            prefer_gpu: If True, try CUDA execution provider first.

        """
        self.model_id = str(model_id)
        self.prefer_gpu = prefer_gpu

        # Resolve model directory
        self.resolved_dir = self._resolve_model_dir(model_id)

        # Load tokenizer
        tokenizer_path = self.resolved_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found in {self.resolved_dir}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self.tokenizer.enable_padding(length=MAX_SEQ_LENGTH)

        # Create ONNX session
        model_path = self.resolved_dir / "onnx" / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"model.onnx not found at {model_path}")

        providers = ["CPUExecutionProvider"]
        if prefer_gpu:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )
        self.backend = "onnx-gpu" if "CUDAExecutionProvider" in self.session.get_providers() else "onnx-cpu"
        self.active_provider = self.session.get_providers()[0]
        self.dim = EMBEDDING_DIM

    def _resolve_model_dir(self, model_id: str | Path) -> Path:
        """Resolve model_id to a local directory containing tokenizer + onnx/model.onnx."""
        path = Path(model_id)
        if path.is_dir():
            # Local directory (e.g., bundle's model/ dir)
            return path.resolve()

        # Hugging Face model ID — use transformers cache
        from huggingface_hub import snapshot_download

        cache_dir = snapshot_download(
            repo_id=str(model_id),
            allow_patterns=["tokenizer*", "onnx/*", "*.json", "vocab.txt"],
        )
        return Path(cache_dir)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts via ONNX Runtime."""
        # Tokenize
        encodings = [self.tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # ONNX inference
        # The Xenova ONNX export requires all three inputs (token_type_ids is
        # the segment/types embedding — zero-fill it since BGE-base doesn't use
        # token types).
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
        outputs = self.session.run(None, inputs)
        last_hidden = outputs[0]  # shape: (batch, seq_len, hidden)

        # Mean pool + L2 normalize
        pooled = mean_pool(last_hidden, attention_mask)
        return l2_normalize(pooled)
