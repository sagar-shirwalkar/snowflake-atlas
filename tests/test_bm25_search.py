"""Tests for the BM25 keyword search index (atlas/bm25_search.py)."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pytest

from atlas.bm25_search import (
    _RestrictedUnpickler,
    build_index,
    rehydrate,
    save_index,
    score_index,
    tokenize,
)

# ── tokenize ─────────────────────────────────────────────────────────


class TestTokenize:
    def test_normal_text(self):
        """Standard text splits on non-alphanumeric boundaries."""
        tokens = tokenize("Hello World! This is a TEST.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_preserves_sql_identifiers(self):
        """SQL keywords with underscores are preserved as single tokens."""
        tokens = tokenize("SELECT * FROM TO_DATE('2024-01-01')")
        assert "to_date" in tokens
        assert "select" in tokens

    def test_preserves_code_identifiers(self):
        """Code identifiers like function_names are preserved."""
        tokens = tokenize("my_function() var_name_2")
        assert "my_function" in tokens
        assert "var_name_2" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_punctuation(self):
        assert tokenize("!!! ???") == []

    def test_lowercases(self):
        tokens = tokenize("UPPER lower MiXeD")
        assert tokens == ["upper", "lower", "mixed"]


# ── build_index ──────────────────────────────────────────────────────


class TestBuildIndex:
    def test_build_with_texts(self):
        texts = [
            "CREATE TABLE users (id INT)",
            "SELECT * FROM users WHERE id = 1",
            "DROP TABLE users",
        ]
        index = build_index(texts)
        assert index.corpus_size == 3
        assert index.avgdl > 0
        assert len(index.doc_len) == 3
        assert len(index.doc_freqs) == 3

    def test_empty_corpus(self):
        """BM25Okapi handles empty corpus gracefully."""
        index = build_index([])
        assert index.corpus_size == 0

    def test_single_document(self):
        index = build_index(["just one document"])
        assert index.corpus_size == 1

    def test_reproducible(self):
        """Building twice from the same texts produces the same IDF."""
        texts = ["apple banana", "banana cherry", "cherry date"]
        i1 = build_index(texts)
        i2 = build_index(texts)
        assert i1.idf == i2.idf


# ── save + rehydrate round-trip ──────────────────────────────────────


class TestSaveRehydrate:
    def test_round_trip(self, tmp_path: Path):
        texts = [
            "CREATE TABLE snowflake_warehouses",
            "ALTER WAREHOUSE SET WAREHOUSE_SIZE = XLARGE",
            "DROP WAREHOUSE IF EXISTS my_wh",
        ]
        index = build_index(texts)
        pkl = tmp_path / "bm25.pkl"
        save_index(index, pkl)
        assert pkl.exists()
        assert pkl.stat().st_size > 0

        loaded = rehydrate(pkl)
        assert loaded.corpus_size == index.corpus_size
        assert loaded.avgdl == index.avgdl
        assert loaded.doc_len == index.doc_len
        assert loaded.idf == index.idf
        assert loaded.k1 == index.k1
        assert loaded.b == index.b
        assert loaded.epsilon == index.epsilon

    def test_round_trip_empty_corpus(self, tmp_path: Path):
        index = build_index([])
        pkl = tmp_path / "empty.pkl"
        save_index(index, pkl)
        loaded = rehydrate(pkl)
        assert loaded.corpus_size == 0
        assert loaded.k1 == 1.5  # default preserved

    def test_pickle_protocol(self, tmp_path: Path):
        """Saved pickle should be readable by Python 3.11+."""
        texts = ["test doc one", "test doc two"]
        index = build_index(texts)
        pkl = tmp_path / "bm25.pkl"
        save_index(index, pkl)
        # Re-read with raw pickle to verify
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        assert "corpus_size" in data
        assert "avgdl" in data
        assert "doc_freqs" in data
        assert "k1" in data
        assert "b" in data
        assert "epsilon" in data


# ── score_index ──────────────────────────────────────────────────────


class TestScoreIndex:
    def test_scores_all_docs(self):
        texts = [
            "CREATE TABLE users (id INT PRIMARY KEY)",
            "SELECT * FROM orders JOIN users ON orders.user_id = users.id",
            "ALTER TABLE users ADD COLUMN email VARCHAR",
        ]
        index = build_index(texts)
        scores = score_index(index, "users table")
        assert len(scores) == 3
        assert scores.dtype == np.float32

    def test_relevant_doc_ranks_higher(self):
        texts = [
            "apple pie recipes with fresh apples",
            "configuring Snowflake virtual warehouse compute",
            "Snowflake data loading and some data pipelines",
            "building large scale data systems with Apache Spark",
            "orchestrating ETL workflows with Airflow",
        ]
        index = build_index(texts)
        scores = score_index(index, "snowflake warehouse")
        # Doc 1 mentions both snowflake and warehouse;
        # doc 2 mentions snowflake but not warehouse;
        # docs 0, 3, 4 mention neither
        assert scores[1] > scores[0]
        assert scores[1] > scores[4]
        assert scores[2] > scores[0]

    def test_empty_query_returns_zeros(self):
        texts = ["doc one", "doc two"]
        index = build_index(texts)
        scores = score_index(index, "")
        assert np.all(scores == 0.0)

    def test_query_no_matches(self):
        texts = ["apples and bananas", "cherries and dates"]
        index = build_index(texts)
        scores = score_index(index, "xyznonexistent")
        assert np.all(scores == 0.0)

    def test_partial_match(self):
        """A doc matching some query terms scores higher than one matching none."""
        texts = [
            "Snowflake data cloud platform and Snowflake compute",
            "Completely unrelated unrelated content here",
            "Some other discussion about cloud infrastructure",
        ]
        index = build_index(texts)
        scores = score_index(index, "snowflake cloud")
        # Doc 0 matches both terms; docs 1 and 2 match none/fewer
        assert scores[0] > scores[1]


# ── Integration: build → save → rehydrate → score ────────────────────


class TestIntegration:
    def test_full_pipeline(self, tmp_path: Path):
        texts = [
            "CREATE VIRTUAL WAREHOUSE my_wh WITH WAREHOUSE_SIZE = 'XSMALL'",
            "SELECT CURRENT_TIMESTAMP",
            "ALTER SESSION SET TIMEZONE = 'UTC'",
        ]
        # Build
        index = build_index(texts)
        # Save
        pkl = tmp_path / "bm25.pkl"
        save_index(index, pkl)
        # Rehydrate
        loaded = rehydrate(pkl)
        # Score
        scores = score_index(loaded, "warehouse virtual")
        assert len(scores) == 3
        assert scores.dtype == np.float32
        # Doc 0 should be most relevant
        assert scores[0] >= scores[1]
        assert scores[0] >= scores[2]

    def test_rehydrated_scores_match_original(self, tmp_path: Path):
        texts = [
            "The cat sat on the mat",
            "The dog chased the cat",
            "Birds fly in the sky",
        ]
        index = build_index(texts)
        original_scores = score_index(index, "cat dog")
        pkl = tmp_path / "bm25.pkl"
        save_index(index, pkl)
        loaded = rehydrate(pkl)
        loaded_scores = score_index(loaded, "cat dog")
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)


# ── Restricted unpickler ─────────────────────────────────────────────


class _EvilDummy:
    """A dummy class used to test that restricted unpickler blocks it."""

    def __init__(self):
        pass


class TestRestrictedUnpickler:
    def test_accepts_bm25_state(self):
        """Unpickler should accept dicts with basic Python types (BM25 state)."""
        data = {
            "corpus_size": 3,
            "avgdl": 5.0,
            "doc_freqs": [{"hello": 1}, {"world": 1}],
            "doc_len": [2, 2],
            "idf": {"hello": 0.5, "world": 0.5},
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
        }
        obj = _RestrictedUnpickler(io.BytesIO(pickle.dumps(data))).load()
        assert obj["corpus_size"] == 3
        assert obj["k1"] == 1.5

    def test_rejects_custom_class(self):
        """Unpickler should refuse to load custom classes outside the safe list."""
        evil = pickle.dumps(_EvilDummy)
        with pytest.raises(pickle.UnpicklingError, match="Refused to unpickle"):
            _RestrictedUnpickler(io.BytesIO(evil)).load()

    def test_rejects_builtin_type(self):
        """Unpickler should refuse arbitrary builtin types not in the safe list."""
        # Fractions are not in the safe type list
        from fractions import Fraction
        evil = pickle.dumps(Fraction)
        with pytest.raises(pickle.UnpicklingError, match="Refused to unpickle"):
            _RestrictedUnpickler(io.BytesIO(evil)).load()

    def test_accepts_nested_builtins(self):
        """Unpickler should accept deeply nested builtin structures."""
        data = {
            "levels": [
                {"a": [1, 2, 3], "b": (4, 5)},
                {"c": {"nested": {"key": b"bytes"}}},
                [10.5, 20.5, 30.5],
            ],
            "flags": {True, False, None},
        }
        obj = _RestrictedUnpickler(io.BytesIO(pickle.dumps(data))).load()
        assert obj["levels"][0]["a"] == [1, 2, 3]
        assert obj["levels"][2][0] == 10.5
        assert None in obj["flags"]
