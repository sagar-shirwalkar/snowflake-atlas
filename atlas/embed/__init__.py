"""Embedding backends package."""

from .base import (
    DEFAULT_MODEL_ID,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    Backend,
    Embedder,
    get_embedder,
    l2_normalize,
    load_embeddings,
    mean_pool,
    resolve_backend,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "EMBEDDING_DIM",
    "MAX_SEQ_LENGTH",
    "Backend",
    "Embedder",
    "get_embedder",
    "load_embeddings",
    "l2_normalize",
    "mean_pool",
    "resolve_backend",
]
