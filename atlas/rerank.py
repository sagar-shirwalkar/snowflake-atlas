"""Cross-encoder re-ranker (MiniLM-L6-v2 ONNX).

Provides :class:`CrossEncoderReranker` which re-scores the top-k
vector search results using a joint query+passage model. Improves
precision by eliminating false positives that rank high on cosine
similarity alone.

Model: ``cross-encoder/ms-marco-MiniLM-L6-v2`` (ONNX export from
Hugging Face). 22.7 M params, ~74 NDCG@10 on MS MARCO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class CrossEncoderReranker:
    """ONNX cross-encoder re-ranker for query-passage pairs."""

    def __init__(self, model_dir: Path | None = None) -> None:
        if model_dir is None:
            # Default to HF cache
            from huggingface_hub import snapshot_download

            model_dir = Path(
                snapshot_download(
                    repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
                    allow_patterns=["*.onnx", "tokenizer*", "*.json"],
                )
            )
        self.model_dir = Path(model_dir)

        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(length=512)

        # Load ONNX model
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.is_file():
            # Try common alternative names
            for name in ["onnx/model.onnx", "model_quantized.onnx"]:
                alt = self.model_dir / name
                if alt.is_file():
                    onnx_path = alt
                    break
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found in {self.model_dir}")

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank results using cross-encoder scores.

        Args:
            query: The original search query
            results: List of result dicts from vector search (must have 'text' field)
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked list of results with updated 'score' field (cross-encoder logits)
        """
        if not results:
            return results

        # Prepare query-passage pairs
        pairs = [(query, r["text"]) for r in results]

        # Tokenize all pairs
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        for q, p in pairs:
            enc = self.tokenizer.encode(q + " [SEP] " + p)
            input_ids_list.append(enc.ids)
            attention_mask_list.append(enc.attention_mask)
            token_type_ids_list.append(enc.type_ids)

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = np.array(
            [ids + [0] * (max_len - len(ids)) for ids in input_ids_list],
            dtype=np.int64,
        )
        attention_mask = np.array(
            [mask + [0] * (max_len - len(mask)) for mask in attention_mask_list],
            dtype=np.int64,
        )
        token_type_ids = np.array(
            [tids + [0] * (max_len - len(tids)) for tids in token_type_ids_list],
            dtype=np.int64,
        )

        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        logits = outputs[0].flatten()  # shape: (n_pairs,)

        # Attach scores and sort
        scored = list(zip(results, logits, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with updated scores
        reranked = []
        for result, score in scored[:top_k]:
            result = result.copy()
            result["score"] = float(score)
            result["reranked"] = True
            reranked.append(result)

        return reranked
