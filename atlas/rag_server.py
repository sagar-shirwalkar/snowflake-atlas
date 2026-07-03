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

from .embed import (
    DEFAULT_MODEL_ID,
    get_embedder,
    load_embeddings,
    resolve_backend,
)
from .log import configure_logging, get_logger
from .rerank import CrossEncoderReranker

app = Server("snowflake-rag")
logger = get_logger()


class Bundle:
    def __init__(self, bundle_dir: Path | str, prefer: str = "auto") -> None:
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
        mode: str = "vector",
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        q = self.embedder.embed([query])[0]
        vec_scores = (self.embeddings @ q).flatten() / self._norms_safe
        query_tokens = set(query.lower().split())
        tb = self._title_boost(self._titles_pa, query_tokens) if query_tokens else 0.0
        boosted = vec_scores + tb

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

        candidate_pool = max(top_k * 20, 50)
        masked = np.where(mask, boosted, -np.inf)
        pool_size = min(candidate_pool, int(mask.sum()))
        if pool_size == 0:
            return []
        pool_idx = np.argpartition(-masked, pool_size - 1)[:pool_size]

        if not query_tokens:
            order = np.argsort(-boosted[pool_idx])
            pool_idx = pool_idx[order]
            return self._collect(min_score, boosted[pool_idx], boosted[pool_idx], top_k)

        kw_raw = np.array(
            [sum(t in self._texts[i].lower() for t in query_tokens) for i in pool_idx],
            dtype=np.float32,
        )

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

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
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
                    "mode": {"type": "string", "enum": ["vector", "hybrid", "keyword"], "default": "vector"},
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
                    "mode": {"type": "string", "enum": ["vector", "hybrid", "keyword"], "default": "vector"},
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
_reranker: CrossEncoderReranker | None = None
_bundle_lock = asyncio.Lock()
_reranker_lock = asyncio.Lock()


async def _bundle_cache(bundle_arg: str, prefer: str) -> Bundle:
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


async def _get_reranker() -> CrossEncoderReranker | None:
    global _reranker
    if _reranker is not None or not _get_args().rerank:
        return _reranker
    async with _reranker_lock:
        if _reranker is not None:
            return _reranker
        _reranker = CrossEncoderReranker()
    return _reranker


def parse_args() -> argparse.Namespace:
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
            "mode": arguments.get("mode", "vector"),
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
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    configure_logging()
    # Ensure args are parsed before serving (triggers lazy init)
    _ = _get_args()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
