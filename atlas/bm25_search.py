"""BM25 keyword search index for hybrid retrieval.

Wraps ``rank_bm25`` behind a simple interface. The index is built
from chunk texts at bundle-build time, saved as a pickle dump, and
rehydrated by the RAG server at startup. Query-time BM25 scoring runs
in pure Python — no GPU, no external service.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

# Lazy import so the RAG server doesn't need rank_bm25 at import time
# if only vector mode is used.
_BM25_OKAPI: Any = None


def _get_bm25_class():
    global _BM25_OKAPI
    if _BM25_OKAPI is None:
        try:
            from rank_bm25 import BM25Okapi

            _BM25_OKAPI = BM25Okapi
        except ImportError as err:
            raise ImportError(
                "rank_bm25 is required for BM25 search. Install with: "
                "uv sync --extra build"
            ) from err
    return _BM25_OKAPI


# Tokenizer: split on non-alphanumeric chars (preserves SQL/CODE
# identifiers like "TO_DATE", "LATERAL_FLATTEN" as single tokens).
_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_#]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that preserves SQL/CODE identifiers."""
    return [t for t in _TOKEN_RE.split(text.lower()) if t]


# ── Build-time API (used by make_bundle.py) ──────────────────────────


def build_index(texts: list[str]) -> Any:
    """Build a ``BM25Okapi`` index from a list of document texts.

    Args:
        texts: Raw chunk text strings.

    Returns:
        A fully initialized ``BM25Okapi`` instance.

    """
    BM25Okapi = _get_bm25_class()
    if not texts:
        # rank_bm25's __init__ divides by corpus_size, which is 0 for
        # empty corpuses.  Bypass it and return a minimal valid instance.
        obj = object.__new__(BM25Okapi)
        obj.corpus_size = 0
        obj.avgdl = 0.0
        obj.doc_freqs = []
        obj.doc_len = []
        obj.idf = {}
        obj.k1 = 1.5
        obj.b = 0.75
        obj.epsilon = 0.25
        return obj
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def save_index(index: Any, path: Path) -> None:
    """Pickle a BM25Okapi index's internal state to disk.

    Only the minimal state (corpus stats, term frequencies, IDF) is
    saved — not the tokenized corpus itself.  This keeps the file small
    (a few MB for 250k docs).
    """
    data = {
        "corpus_size": index.corpus_size,
        "avgdl": index.avgdl,
        "doc_freqs": index.doc_freqs,
        "doc_len": index.doc_len,
        "idf": index.idf,
        "k1": index.k1,
        "b": index.b,
        "epsilon": index.epsilon,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ── Query-time API (used by rag_server.py) ───────────────────────────


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows types used in BM25 index state.

    Prevents arbitrary code execution during deserialization by blocking
    class instantiation outside of basic Python types and numpy arrays.
    The BM25 save/restore format only uses dicts, lists, floats, ints,
    strings, and bytes — no custom classes.
    """

    _SAFE = {
        "builtins": {"dict", "list", "tuple", "set", "str", "int",
                     "float", "bool", "bytes", "frozenset", "NoneType"},
        "numpy": {"ndarray", "float32", "float64", "int32", "int64"},
    }

    def find_class(self, module: str, name: str) -> type:
        if module in self._SAFE and name in self._SAFE[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refused to unpickle {module}.{name} — not in BM25 safe types"
        )


def rehydrate(path: Path) -> Any:
    """Load a BM25Okapi index from a saved pickle.

    Deserializes the internal state saved by :func:`save_index` and
    constructs a fully functional ``BM25Okapi`` instance via low-level
    attribute assignment (avoiding the costly ``__init__`` corpus scan).

    Uses a restricted unpickler that only allows basic Python types
    and numpy arrays — preventing arbitrary code execution from a
    tampered index file.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        A ``BM25Okapi`` instance ready for scoring.

    """
    BM25Okapi = _get_bm25_class()
    with open(path, "rb") as f:
        data = _RestrictedUnpickler(f).load()
    obj = object.__new__(BM25Okapi)
    obj.corpus_size = data["corpus_size"]
    obj.avgdl = data["avgdl"]
    obj.doc_freqs = data["doc_freqs"]
    obj.doc_len = data["doc_len"]
    obj.idf = data["idf"]
    obj.k1 = data.get("k1", 1.5)
    obj.b = data.get("b", 0.75)
    obj.epsilon = data.get("epsilon", 0.25)
    return obj


def score_index(index: Any, query: str) -> np.ndarray:
    """Score all documents against a query using BM25.

    Args:
        index: A rehydrated ``BM25Okapi`` instance.
        query: Raw query string.

    Returns:
        Float32 array of BM25 scores, one per document in corpus order.

    """
    tokens = tokenize(query)
    if not tokens or index.corpus_size == 0:
        return np.zeros(index.corpus_size, dtype=np.float32)
    scores = index.get_scores(tokens)
    return np.array(scores, dtype=np.float32)
