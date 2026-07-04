"""BM25 keyword search index for hybrid retrieval.

Wraps ``rank_bm25`` behind a simple interface. The index is built
from chunk texts at bundle-build time, saved as a pickle dump, and
rehydrated by the RAG server at startup. Query-time BM25 scoring runs
in pure Python — no GPU, no external service.

Supports two index types:

- **Single-field index** (legacy): built from body text only via
  :func:`build_index`.  Persisted as version 1.
- **Fielded index** (preferred): built from multiple fields (text,
  title, heading) with per-field weights via :func:`build_fielded_index`.
  Persisted as version 2.  The :class:`FieldedBM25Index` wrapper
  exposes the same duck-typed interface (``get_scores``, ``corpus_size``)
  so callers like ``rag_server.py`` work unchanged.
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


_DEFAULT_K1 = 2.0
_DEFAULT_B = 0.7


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


# ── Fielded BM25 index ────────────────────────────────────────────────


class FieldedBM25Index:
    """Multi-field BM25 index with per-field weighting.

    Builds a separate ``BM25Okapi`` instance for each field and combines
    their scores at query time as a weighted sum:

        final_score = sum(weight_f * bm25_f(tokens) for f in fields)

    Exposes ``get_scores(tokens)`` and ``corpus_size`` so that existing
    callers (``score_index``, ``rag_server.py``) work unchanged.
    """

    def __init__(
        self,
        field_texts: dict[str, list[str]],
        field_weights: dict[str, float],
    ) -> None:
        BM25Okapi = _get_bm25_class()
        self.field_weights = dict(field_weights)
        self.corpus_size = 0
        self._indexes: dict[str, Any] = {}

        for field_name, texts in field_texts.items():
            if texts:
                tokenized = [tokenize(t) for t in texts]
                self._indexes[field_name] = BM25Okapi(
                    tokenized, k1=_DEFAULT_K1, b=_DEFAULT_B
                )
                self.corpus_size = len(texts)
            else:
                obj = object.__new__(BM25Okapi)
                obj.corpus_size = 0
                obj.avgdl = 0.0
                obj.doc_freqs = []
                obj.doc_len = []
                obj.idf = {}
                obj.k1 = _DEFAULT_K1
                obj.b = _DEFAULT_B
                obj.epsilon = 0.25
                self._indexes[field_name] = obj

    def get_scores(self, tokens: list[str]) -> list[float]:
        """Weighted sum of per-field BM25 scores."""
        if not self._indexes:
            return []
        combined: np.ndarray | None = None
        for field_name, bm25 in self._indexes.items():
            weight = self.field_weights.get(field_name, 1.0)
            field_scores = np.array(bm25.get_scores(tokens), dtype=np.float32)
            if combined is None:
                combined = weight * field_scores
            else:
                combined += weight * field_scores
        return combined.tolist()  # type: ignore[union-attr]

    def to_save_dict(self) -> dict[str, Any]:
        """Serialize internal state for pickling."""
        fields: dict[str, dict[str, Any]] = {}
        for field_name, bm25 in self._indexes.items():
            fields[field_name] = {
                "corpus_size": bm25.corpus_size,
                "avgdl": bm25.avgdl,
                "doc_freqs": bm25.doc_freqs,
                "doc_len": bm25.doc_len,
                "idf": bm25.idf,
                "k1": bm25.k1,
                "b": bm25.b,
                "epsilon": bm25.epsilon,
            }
        return {
            "version": 2,
            "corpus_size": self.corpus_size,
            "field_weights": dict(self.field_weights),
            "fields": fields,
        }

    @classmethod
    def from_save_dict(cls, data: dict[str, Any]) -> FieldedBM25Index:
        """Deserialize internal state from a save dict."""
        BM25Okapi = _get_bm25_class()
        obj = object.__new__(cls)
        obj.field_weights = dict(data["field_weights"])
        obj.corpus_size = data["corpus_size"]
        obj._indexes = {}
        for field_name, fd in data["fields"].items():
            bm25 = object.__new__(BM25Okapi)
            bm25.corpus_size = fd["corpus_size"]
            bm25.avgdl = fd["avgdl"]
            bm25.doc_freqs = fd["doc_freqs"]
            bm25.doc_len = fd["doc_len"]
            bm25.idf = fd["idf"]
            bm25.k1 = fd.get("k1", _DEFAULT_K1)
            bm25.b = fd.get("b", _DEFAULT_B)
            bm25.epsilon = fd.get("epsilon", 0.25)
            obj._indexes[field_name] = bm25
        return obj


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
        obj.k1 = _DEFAULT_K1
        obj.b = _DEFAULT_B
        obj.epsilon = 0.25
        return obj
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized, k1=_DEFAULT_K1, b=_DEFAULT_B)


def build_fielded_index(
    field_texts: dict[str, list[str]],
    field_weights: dict[str, float] | None = None,
) -> FieldedBM25Index:
    """Build a field-weighted BM25 index.

    Creates separate ``BM25Okapi`` instances for each field (e.g.
    ``text``, ``title``, ``heading``) and wraps them in a
    :class:`FieldedBM25Index` that combines scores at query time.

    All field lists must have the same length (one entry per document).

    Args:
        field_texts: Mapping of ``field_name -> list[document_text]``.
        field_weights: Per-field score weight (default: ``{"text": 1.0,
            "title": 3.0, "heading": 2.0}``).

    Returns:
        A :class:`FieldedBM25Index` instance.

    """
    if field_weights is None:
        field_weights = {"text": 1.0, "title": 3.0, "heading": 2.0}
    return FieldedBM25Index(field_texts, field_weights)


def save_index(index: Any, path: Path) -> None:
    """Pickle a BM25 index's internal state to disk.

    Handles both :class:`FieldedBM25Index` (version 2 format) and plain
    ``BM25Okapi`` (version 1 / legacy format).  Only the minimal state
    (corpus stats, term frequencies, IDF) is saved — not the tokenized
    corpus itself.  This keeps the file small (a few MB for 250k docs).
    """
    if isinstance(index, FieldedBM25Index):
        data = index.to_save_dict()
    else:
        data = {
            "version": 1,
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
    """Load a BM25 index from a saved pickle.

    Deserializes the internal state saved by :func:`save_index` and
    constructs a fully functional ``BM25Okapi`` or
    :class:`FieldedBM25Index` instance via low-level attribute
    assignment (avoiding the costly ``__init__`` corpus scan).

    Uses a restricted unpickler that only allows basic Python types
    and numpy arrays — preventing arbitrary code execution from a
    tampered index file.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        A ``BM25Okapi`` or :class:`FieldedBM25Index` instance ready for scoring.

    """
    BM25Okapi = _get_bm25_class()
    with open(path, "rb") as f:
        data = _RestrictedUnpickler(f).load()

    # Version 2: fielded BM25 index
    if isinstance(data, dict) and data.get("version", 1) >= 2:
        return FieldedBM25Index.from_save_dict(data)

    # Version 1 (or legacy without version): single-field BM25Okapi
    obj = object.__new__(BM25Okapi)
    obj.corpus_size = data["corpus_size"]
    obj.avgdl = data["avgdl"]
    obj.doc_freqs = data["doc_freqs"]
    obj.doc_len = data["doc_len"]
    obj.idf = data["idf"]
    obj.k1 = data.get("k1", _DEFAULT_K1)
    obj.b = data.get("b", _DEFAULT_B)
    obj.epsilon = data.get("epsilon", 0.25)
    return obj


def score_index(index: Any, query: str) -> np.ndarray:
    """Score all documents against a query using BM25.

    Args:
        index: A rehydrated ``BM25Okapi`` or :class:`FieldedBM25Index` instance.
        query: Raw query string.

    Returns:
        Float32 array of BM25 scores, one per document in corpus order.

    """
    tokens = tokenize(query)
    if not tokens or index.corpus_size == 0:
        return np.zeros(index.corpus_size, dtype=np.float32)
    scores = index.get_scores(tokens)
    return np.array(scores, dtype=np.float32)
