"""MCP server exposing the pre-built RAG bundle.

Four tools: `search_docs`, `search_code`, `get_chunk`,
`get_bundle_info`. Loads the bundle once at startup into memory,
answers queries with cosine similarity over a single matrix
multiply. Returns chunks with file paths, headings, similarity
scores, and provenance.

Reads the portable bundle produced by ``make_bundle.py`` and answers
semantic-search queries by:
  1. Embedding the query with the embedder resolved at startup
     (Apple MLX on M-series, ONNX+CUDA on NVIDIA Linux boxes,
     ONNX+CPU everywhere else; see ``atlas/embed/base.py``).
  2. Computing cosine similarity against the precomputed matrix.
  3. Returning the top-k chunks with their metadata.

The bundle is loaded once at startup. Bundle format (see
``make_bundle.py``) is platform-agnostic: only the inference runtime
differs between backends, not the vectors. See ``atlas-doctor`` for
which backend was selected at build and at run time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .bm25_search import rehydrate as load_bm25_index
from .bm25_search import score_index as bm25_score
from .embed import (
    DEFAULT_MODEL_ID,
    get_embedder,
    load_embeddings,
    resolve_backend,
)
from .log import configure_logging, get_logger
from .rerank import CrossEncoderReranker

# MLX reranker is optional — falls back to ONNX if not available
try:
    from .rerank_mlx import MlxCrossEncoderReranker as _MlxReranker

    _HAS_MLX_RERANKER = True
except ImportError:
    _MlxReranker = None  # type: ignore[assignment]
    _HAS_MLX_RERANKER = False

app = Server("snowflake-rag")
logger = get_logger()


class Bundle:
    """In-memory RAG bundle: embeddings, BM25 index, metadata.

    Loads a pre-built bundle from disk and exposes ``search()`` and
    ``get_chunk()`` methods for the RAG MCP server.  Supports vector,
    keyword, and hybrid search modes.
    """

    def __init__(self, bundle_dir: Path | str, prefer: str = "auto") -> None:
        """Load a bundle from disk, resolving embedder and BM25 index."""
        self.bundle_dir = Path(bundle_dir).resolve()
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Bundle manifest missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        chunks_path = self.bundle_dir / "chunks.parquet"
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Bundle chunks missing: {chunks_path}")
        table = pq.read_table(chunks_path)
        self.embeddings = load_embeddings(self.bundle_dir)
        norms_path = self.bundle_dir / "norms.f32.npy"
        if norms_path.is_file():
            self.norms = np.load(norms_path)
        else:
            self.norms = np.linalg.norm(self.embeddings, axis=1).astype(np.float32)
        bundled_model = self.bundle_dir / "model" / "onnx" / "model.onnx"
        if bundled_model.is_file():
            model_id: str | Path = self.bundle_dir / "model"
        else:
            model_id = DEFAULT_MODEL_ID
        try:
            self.embedder = get_embedder(model_id, prefer=prefer)
        except Exception as e:
            logger.warning(
                "Preferred embedder failed, falling back to ONNX+CPU",
                model_id=str(model_id),
                error=str(e),
            )
            self.embedder = get_embedder(DEFAULT_MODEL_ID, prefer="cpu")
        self._norms_safe = self.norms.clip(min=1e-9)

        self._n = len(table)
        self._pub = table.column("publication").to_numpy(zero_copy_only=False)
        self._area = table.column("product_area").to_numpy(zero_copy_only=False)
        self._code = table.column("is_code").to_numpy().astype(bool)
        self._texts = table.column("text").to_pylist()
        self._titles_pa = table.column("title")
        ids = table.column("id").to_pylist()
        self._id_to_idx = {cid: i for i, cid in enumerate(ids)}
        self._chunk_ids = ids
        self._files = table.column("file").to_pylist()
        self._headings = table.column("heading").to_pylist()
        self._product_areas = table.column("product_area").to_pylist()
        self._last_updated = table.column("last_updated").to_pylist()
        self._canonical_urls = table.column("canonical_url").to_pylist()

        # Hierarchical cluster tags (sibling stems for path-aware boost)
        if "cluster_tags" in table.column_names:
            self._cluster_tags_pa = table.column("cluster_tags")
        else:
            self._cluster_tags_pa = None

        # Load BM25 index for keyword / hybrid search modes
        bm25_path = self.bundle_dir / "bm25.pkl"
        if bm25_path.is_file():
            self._bm25 = load_bm25_index(bm25_path)
            logger.info("Loaded BM25 index", path=str(bm25_path))
        else:
            self._bm25 = None
            logger.info("No BM25 index found — keyword/hybrid fallback to title boost only")

    @staticmethod
    def _title_boost(titles: list[str], query_tokens: set[str]) -> np.ndarray:
        boost = np.zeros(len(titles), dtype=np.float32)
        for t in query_tokens:
            matches = pc.match_substring(titles, t)
            boost += matches.cast("float32").to_numpy()
        return boost * 0.05

    def search(
        self,
        query: str,
        top_k: int = 5,
        publication: str | None = None,
        product_area: str | None = None,
        is_code: bool | None = None,
        min_score: float = 0.0,
        mode: str = "hybrid",
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the bundle using vector, keyword, or hybrid mode.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            publication: Filter by publication name.
            product_area: Filter by product area.
            is_code: Filter to code chunks only.
            min_score: Minimum similarity score threshold.
            mode: ``"vector"``, ``"keyword"``, or ``"hybrid"`` (default).
            candidate_k: Internal candidate pool size (for reranking).

        Returns:
            List of result dicts with chunk metadata and scores.

        """
        q = self.embedder.embed([query])[0]
        vec_scores = (self.embeddings @ q).flatten() / self._norms_safe
        query_tokens = set(query.lower().split())
        tb = self._title_boost(self._titles_pa, query_tokens) if query_tokens else 0.0
        boosted = vec_scores + tb

        # Hierarchical cluster boost: sibling stem tokens in the same dir
        if query_tokens and self._cluster_tags_pa is not None:
            sb = np.zeros(self._n, dtype=np.float32)
            for t in query_tokens:
                matches = pc.match_substring(self._cluster_tags_pa, t)
                sb += matches.cast("float32").to_numpy()
            boosted += sb * 0.03

        mask = np.ones(self._n, dtype=bool)
        if publication:
            mask &= self._pub == publication
        if product_area:
            mask &= self._area == product_area
        if is_code is not None:
            mask &= self._code == bool(is_code)

        if mode == "vector":
            if candidate_k is not None:
                masked = np.where(mask, boosted, -np.inf)
                pool = min(candidate_k, int(mask.sum()))
                if pool == 0:
                    return []
                pool_idx = np.argpartition(-masked, pool - 1)[:pool]
                order = np.argsort(-boosted[pool_idx])
                pool_idx = pool_idx[order]
                return self._collect(min_score, boosted[pool_idx], boosted[pool_idx], pool, idx_map=pool_idx)
            scores = np.where(mask, boosted, -np.inf)
            order = np.argsort(-scores)
            return self._collect(min_score, scores[order], scores[order], top_k, idx_map=order)

        # ── Interleaved candidate pool (vector + BM25) ────────────────
        candidate_pool = max(top_k * 20, 50)
        masked = np.where(mask, boosted, -np.inf)
        pool_size = min(candidate_pool, int(mask.sum()))
        if pool_size == 0:
            return []

        # Compute BM25 early (needed for interleaved pool and hybrid fusion)
        if self._bm25 is not None:
            all_kw = bm25_score(self._bm25, query)
        elif query_tokens:
            all_kw = np.array(
                [sum(t in txt.lower() for t in query_tokens) for txt in self._texts],
                dtype=np.float32,
            )
        else:
            all_kw = None

        # Vector-selected pool (always)
        vec_pool = np.argpartition(-masked, pool_size - 1)[:pool_size]

        # BM25-selected pool interleaved for keyword-aware modes
        if mode in ("keyword", "hybrid") and all_kw is not None:
            kw_masked = np.where(mask, all_kw, -np.inf)
            kw_pool = np.argpartition(-kw_masked, pool_size - 1)[:pool_size]
            pool_idx = np.unique(np.concatenate([vec_pool, kw_pool]))
        else:
            pool_idx = vec_pool

        if not query_tokens:
            order = np.argsort(-boosted[pool_idx])
            pool_idx = pool_idx[order]
            return self._collect(min_score, boosted[pool_idx], boosted[pool_idx], top_k)

        # BM25 scores for the selected pool
        if all_kw is not None:
            kw_raw = all_kw[pool_idx]
        else:
            kw_raw = np.array(
                [sum(t in self._texts[i].lower() for t in query_tokens) for i in pool_idx],
                dtype=np.float32,
            )

        # ── Fusion ───────────────────────────────────────────────────
        if mode == "keyword":
            scores = kw_raw + tb[pool_idx]
            vec_ref = kw_raw
        else:
            kw_max = kw_raw.max()
            kw_norm = kw_raw / kw_max if kw_max > 0 else kw_raw
            scores = 0.6 * boosted[pool_idx] + 0.4 * kw_norm
            vec_ref = boosted[pool_idx]

        order = np.argsort(-scores)
        pool_idx = pool_idx[order]
        scores = scores[order]
        vec_ref = vec_ref[order]

        # ── AdaGReS diversity for hybrid mode ────────────────────────
        if mode == "hybrid" and top_k > 1 and len(pool_idx) > top_k:
            sel = self._adagres_select(scores, pool_idx, top_k)
            pool_idx = pool_idx[sel]
            scores = scores[sel]
            vec_ref = vec_ref[sel]

        return self._collect(min_score, scores, vec_ref, top_k, idx_map=pool_idx)

    def _collect(
        self,
        min_score: float,
        scores: np.ndarray,
        vec_ref: np.ndarray,
        top_k: int,
        idx_map: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rank in range(min(top_k, len(scores))):
            score = float(scores[rank])
            if score < min_score:
                continue
            i = idx_map[rank] if idx_map is not None else rank
            results.append(
                {
                    "id": self._chunk_ids[i],
                    "score": score,
                    "publication": self._pub[i],
                    "file": self._files[i],
                    "heading": self._headings[i],
                    "title": self._titles_pa[i].as_py(),
                    "product_area": self._product_areas[i],
                    "last_updated": self._last_updated[i],
                    "canonical_url": self._canonical_urls[i],
                    "is_code": bool(self._code[i]),
                    "text": self._texts[i],
                }
            )
        return results

    def _adagres_select(
        self,
        pool_scores: np.ndarray,
        pool_idx: np.ndarray,
        top_k: int,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Heading-aware AdaGReS diversity ranking.

        Adaptively trades relevance for diversity as rank increases.
        At k=1 the top score item is always kept.  At each subsequent
        step an adaptive lambda penalises candidates whose file+heading
        has already been selected.

        Similarity rules:
        - Same file + same heading: fully redundant (sim=1.0)
        - Same file + different heading: partial (sim=0.5)
        - Different file: no penalty (sim=0.0)

        Args:
            pool_scores: Hybrid fusion scores, sorted descending.
            pool_idx: Corpus index for each pool entry.
            top_k: Number of items to select.
            alpha: AdaGReS decay rate (default 0.5).

        Returns:
            Indices into *pool_scores* / *pool_idx* for the selected
            items, preserving original score order.

        """
        n = len(pool_scores)
        if n <= top_k:
            return np.arange(n, dtype=np.intp)

        pool_files = [self._files[i] for i in pool_idx]
        pool_headings = [self._headings[i] for i in pool_idx]

        selected = [0]  # always keep the top-scoring item
        chosen: set[tuple[str, str]] = {(pool_files[0], pool_headings[0])}
        remaining = set(range(1, n))

        while len(selected) < top_k and remaining:
            k = len(selected) + 1
            lambda_k = math.exp(-alpha * (k - 1))

            best_j: int | None = None
            best_score = -1.0

            for j in remaining:
                # Maximum similarity to any already-selected item
                max_sim = 0.0
                fj = pool_files[j]
                hj = pool_headings[j]
                for i in selected:
                    if fj == pool_files[i]:
                        if hj == pool_headings[i]:
                            max_sim = 1.0  # same file + heading
                            break
                        max_sim = max(max_sim, 0.5)  # same file, diff heading
                    # else sim = 0.0 (different file)

                score = lambda_k * float(pool_scores[j]) + (1.0 - lambda_k) * (1.0 - max_sim)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                selected.append(best_j)
                chosen.add((pool_files[best_j], pool_headings[best_j]))
                remaining.remove(best_j)
            else:
                break

        return np.array(selected, dtype=np.intp)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a single chunk by its ID, or None if not found."""
        i = self._id_to_idx.get(chunk_id)
        if i is None:
            return None
        return {
            "id": self._chunk_ids[i],
            "publication": self._pub[i],
            "file": self._files[i],
            "heading": self._headings[i],
            "title": self._titles_pa[i].as_py(),
            "product_area": self._product_areas[i],
            "last_updated": self._last_updated[i],
            "canonical_url": self._canonical_urls[i],
            "is_code": bool(self._code[i]),
            "text": self._texts[i],
        }


def _result(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the RAG server tools (MCP list_tools handler)."""
    return [
        Tool(
            name="search_docs",
            description="Semantic search across documentation. Best for conceptual or 'how do I' queries. Returns top-k chunks with file paths, headings, and similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "publication": {"type": "string"},
                    "product_area": {"type": "string"},
                    "min_score": {"type": "number", "default": 0.0, "minimum": -1.0, "maximum": 1.0},
                    "mode": {"type": "string", "enum": ["vector", "hybrid", "keyword"], "default": "hybrid"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_code",
            description="Semantic search restricted to code examples in the docs. Useful for 'show me a script that does X'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "publication": {"type": "string"},
                    "min_score": {"type": "number", "default": 0.0},
                    "mode": {"type": "string", "enum": ["vector", "hybrid", "keyword"], "default": "hybrid"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_chunk",
            description="Fetch a single chunk by its ID. Use after a search_docs call to get the full content of a specific hit.",
            inputSchema={
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="get_bundle_info",
            description="Return the bundle manifest: source repo/branch/SHA, build date, chunk count, embedding model. Use to cite freshness.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


_bundle_instance: Bundle | None = None
_reranker: CrossEncoderReranker | Any | None = None
_bundle_lock = asyncio.Lock()
_reranker_lock = asyncio.Lock()


async def _bundle_cache(bundle_arg: str, prefer: str) -> Bundle:
    """Lazily initialise and cache the :class:`Bundle` singleton."""
    global _bundle_instance
    if _bundle_instance is not None:
        return _bundle_instance
    async with _bundle_lock:
        if _bundle_instance is not None:
            return _bundle_instance
        bundle_path = Path(bundle_arg).expanduser()
        if not bundle_path.is_absolute():
            bundle_path = bundle_path.resolve()
        backend, reason = resolve_backend(prefer)
        logger.info("Backend resolved", backend=backend, reason=reason)
        _bundle_instance = Bundle(bundle_path, prefer=prefer)
    return _bundle_instance


async def _get_reranker() -> Any | None:
    """Lazily load and cache the cross-encoder reranker (MLX or ONNX)."""
    global _reranker
    if _reranker is not None or not _get_args().rerank:
        return _reranker
    async with _reranker_lock:
        if _reranker is not None:
            return _reranker
        # Try MLX first (Apple Silicon), fall back to ONNX
        if _HAS_MLX_RERANKER:
            try:
                _reranker = _MlxReranker()
                logger.info("Loaded MLX cross-encoder reranker (ANE/GPU)")
                return _reranker
            except (ImportError, FileNotFoundError, RuntimeError) as e:
                logger.info("MLX reranker unavailable, falling back to ONNX", error=str(e))
        _reranker = CrossEncoderReranker()
        logger.info("Loaded ONNX cross-encoder reranker (CPU)")
    return _reranker


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RAG MCP server."""
    p = argparse.ArgumentParser(description="Atlas RAG MCP server")
    p.add_argument(
        "--bundle",
        default="./data/rag-bundle",
        help="Path to a pre-built RAG bundle directory",
    )
    p.add_argument(
        "--prefer",
        choices=["auto", "apple", "nvidia", "cpu"],
        default="auto",
        help="Embedding backend preference: apple=MLX, nvidia=CUDA, cpu=ONNX+CPU, auto=probe",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Load a cross-encoder re-ranker (MiniLM-L6-v2) to refine top-100 vector results. Adds ~45 MB RAM and ~5 ms/query.",
    )
    return p.parse_args()


_ARGS: argparse.Namespace | None = None


def _get_args() -> argparse.Namespace:
    global _ARGS
    if _ARGS is None:
        _ARGS = parse_args()
    return _ARGS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to search/code/chunk/info handlers (MCP call_tool handler)."""
    cid = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    _start = time.perf_counter()
    logger.info("Tool called", tool=name)

    try:
        args = _get_args()
        bundle = await _bundle_cache(args.bundle, args.prefer)
    except FileNotFoundError as e:
        logger.warning("Bundle not found", error=str(e))
        return _result({"error": str(e)})

    try:
        if name == "get_bundle_info":
            return _result(bundle.manifest)

        top_k = arguments.get("top_k", 5)
        reranker = await _get_reranker()
        search_kw: dict[str, Any] = {
            "publication": arguments.get("publication"),
            "product_area": arguments.get("product_area"),
            "min_score": arguments.get("min_score", 0.0),
            "mode": arguments.get("mode", "hybrid"),
        }

        if name == "search_docs":
            results = bundle.search(
                arguments["query"],
                top_k=100 if reranker else top_k,
                candidate_k=100 if reranker else None,
                **search_kw,
            )
            if reranker and results:
                results = reranker.rerank(arguments["query"], results, top_k=top_k)
            return _result(results)

        if name == "search_code":
            results = bundle.search(
                arguments["query"],
                top_k=100 if reranker else top_k,
                is_code=True,
                candidate_k=100 if reranker else None,
                **search_kw,
            )
            if reranker and results:
                results = reranker.rerank(arguments["query"], results, top_k=top_k)
            return _result(results)

        if name == "get_chunk":
            chunk = bundle.get_chunk(arguments["chunk_id"])
            if chunk is None:
                return _result({"error": f"chunk_id not found: {arguments['chunk_id']}"})
            return _result(chunk)

        raise ValueError(f"Unknown tool: {name}")
    finally:
        elapsed = (time.perf_counter() - _start) * 1000
        logger.info("Tool finished", tool=name, duration_ms=round(elapsed, 2))


async def serve() -> None:
    """Run the RAG server over stdio (MCP transport)."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point: configure logging, parse args, and run the server."""
    configure_logging()
    # Ensure args are parsed before serving (triggers lazy init)
    _ = _get_args()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
