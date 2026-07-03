"""Convert Hugging Face BGE reranker weights to MLX-loadable .npy files.

One-time helper.  Reads the ``BAAI/bge-reranker-v2-base`` PyTorch state
dict and writes BERT encoder weights (same layout as the embedder) plus
the classifier head to ``~/.cache/atlas/models/bge-reranker-v2-base-mlx/``.

Run this once per machine (or once per maintainer) to populate the MLX
weights cache for the cross-encoder re-ranker.

Usage::

    uv run python tools/convert_reranker_to_mlx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The BERT encoder uses the same key mapping as the embedding model
# (``convert_bge_to_mlx.py``).  The reranker additionally has a
# ``classifier.{weight,bias}`` head.
# HF state-dict keys for ``AutoModelForSequenceClassification`` follow:
#   bert.encoder.layer.0.attention.self.query.weight
#   ...
#   classifier.weight
#   classifier.bias


def _hf_to_mlx(hf_key: str) -> str | None:
    """Map a HuggingFace state-dict key to an MLX weight-file name.

    Returns ``None`` for keys that should be skipped (non-parameter
    buffers, the final encoder LayerNorm that our model doesn't use).
    """
    rest = hf_key
    if rest.startswith("bert."):
        rest = rest[len("bert.") :]

    # Embeddings
    if rest == "embeddings.word_embeddings.weight":
        return "word_embeddings.weight"
    if rest == "embeddings.position_embeddings.weight":
        return "position_embeddings.weight"
    if rest == "embeddings.token_type_embeddings.weight":
        return "token_type_embeddings.weight"
    if rest == "embeddings.LayerNorm.weight":
        return "embeddings.LayerNorm.weight"
    if rest == "embeddings.LayerNorm.bias":
        return "embeddings.LayerNorm.bias"

    # Final encoder norm (BGE drops this for retrieval, but the HF
    # state dict still contains it; skip it — our model doesn't use it)
    if rest in ("encoder.LayerNorm.weight", "encoder.LayerNorm.bias"):
        return None

    # Encoder layers
    if rest.startswith("encoder.layer."):
        parts = rest.split(".")
        i = parts[2]
        block = parts[3]
        if block == "attention":
            if len(parts) >= 5 and parts[4] == "self":
                kind = parts[5]
                if kind in ("query", "key", "value"):
                    bias = len(parts) == 7 and parts[6] == "bias"
                    suffix = ".bias" if bias else ".weight"
                    proj = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[kind]
                    return f"encoder.layer.{i}.attention.{proj}{suffix}"
            if len(parts) >= 5 and parts[4] == "output":
                if parts[5] == "dense":
                    bias = len(parts) == 7 and parts[6] == "bias"
                    suffix = ".bias" if bias else ".weight"
                    return f"encoder.layer.{i}.attention.o_proj{suffix}"
                if parts[5] == "LayerNorm":
                    suffix = "." + parts[6]
                    return f"encoder.layer.{i}.attention.LayerNorm{suffix}"
        if block == "intermediate" and parts[4] == "dense":
            bias = len(parts) == 6 and parts[5] == "bias"
            suffix = ".bias" if bias else ".weight"
            return f"encoder.layer.{i}.ffn.lin1{suffix}"
        if block == "output" and parts[4] == "dense":
            bias = len(parts) == 6 and parts[5] == "bias"
            suffix = ".bias" if bias else ".weight"
            return f"encoder.layer.{i}.ffn.lin2{suffix}"
        if block == "output" and parts[4] == "LayerNorm":
            suffix = "." + parts[5]
            return f"encoder.layer.{i}.ffn.LayerNorm{suffix}"

    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Convert BGE reranker weights to MLX format")
    p.add_argument(
        "--src",
        default="BAAI/bge-reranker-v2-base",
        help="Hugging Face model id (BERT-base cross-encoder)",
    )
    p.add_argument(
        "--dst",
        default=str(Path.home() / ".cache" / "atlas" / "models" / "bge-reranker-v2-base-mlx"),
        help="Output directory for .npy weight files",
    )
    args = p.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  PyTorch is required for weight conversion. Install with:", file=sys.stderr)
        print("    uv pip install torch", file=sys.stderr)
        return 1

    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        print("  transformers is required. Install with:", file=sys.stderr)
        print("    uv pip install transformers", file=sys.stderr)
        return 1

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {args.src} (PyTorch state dict)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.src, num_labels=1
    )
    state = model.state_dict()

    converted = 0
    skipped = 0

    for hf_key, tensor in state.items():
        # Classifier head (not part of BERT encoder)
        if hf_key == "classifier.weight":
            mlx_key = "classifier.weight"
        elif hf_key == "classifier.bias":
            mlx_key = "classifier.bias"
        else:
            mlx_key = _hf_to_mlx(hf_key)

        if mlx_key is None:
            skipped += 1
            continue
        arr = tensor.detach().cpu().numpy()
        np.save(dst / f"{mlx_key}.npy", arr)
        converted += 1

    # Save config for verification
    config = {
        "source_model": args.src,
        "format": "mlx-weights-v1",
        "n_tensors": converted,
        "n_skipped": skipped,
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "ff_dim": 3072,
        "max_seq": 512,
        "vocab_size": 30522,
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Wrote {converted} tensors to {dst}")
    print(f"  Skipped {skipped} non-parameter buffers")
    print(f"  Wrote {dst / 'config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
