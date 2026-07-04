"""RAG quality evaluation (Precision@10, MRR).

``atlas-evaluate`` console script. Loads a bundle, runs queries from
a golden set, and computes retrieval metrics against expected files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .bm25_search import score_index as bm25_score
from .embed import get_embedder, load_embeddings
from .rerank import CrossEncoderReranker


def load_bundle(bundle_dir: Path, prefer: str = "auto"):
    """Load bundle components for evaluation."""
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    table = pq.read_table(bundle_dir / "chunks.parquet")
    embeddings = load_embeddings(bundle_dir)
    norms_path = bundle_dir / "norms.f32.npy"
    norms = np.load(norms_path) if norms_path.is_file() else np.linalg.norm(embeddings, axis=1).astype(np.float32)

    bundled_model = bundle_dir / "model" / "onnx" / "model.onnx"
    model_id = bundle_dir / "model" if bundled_model.is_file() else manifest["embedding_model"]
    embedder = get_embedder(model_id, prefer=prefer)

    # Load BM25 index for keyword / hybrid evaluation
    bm25_path = bundle_dir / "bm25.pkl"
    bm25_index = None
    if bm25_path.is_file():
        from .bm25_search import rehydrate as _load_bm25
        bm25_index = _load_bm25(bm25_path)

    # Extract columns
    pub_col = table.column("publication").to_numpy(zero_copy_only=False)
    file_col = table.column("file").to_pylist()
    id_col = table.column("id").to_pylist()
    text_col = table.column("text").to_pylist()
    title_col = table.column("title").to_pylist()

    return {
        "manifest": manifest,
        "embeddings": embeddings,
        "norms": norms,
        "embedder": embedder,
        "bm25": bm25_index,
        "pub": pub_col,
        "file": file_col,
        "id": id_col,
        "text": text_col,
        "title": title_col,
        "n": len(table),
    }


def search_bundle(bundle: dict, query: str, top_k: int = 10, mode: str = "vector") -> list[dict[str, Any]]:
    """Search the bundle in vector, keyword, or hybrid mode."""
    q = bundle["embedder"].embed([query])[0]
    vec_scores = (bundle["embeddings"] @ q).flatten() / bundle["norms"].clip(min=1e-9)

    query_tokens = set(query.lower().split())
    if query_tokens:
        import pyarrow.compute as pc
        tb = np.zeros(bundle["n"], dtype=np.float32)
        for t in query_tokens:
            matches = pc.match_substring(bundle["title"], t)
            tb += matches.cast("float32").to_numpy()
        boosted = vec_scores + tb * 0.05
    else:
        boosted = vec_scores

    if mode == "keyword":
        bm25 = bundle.get("bm25")
        if bm25 is None:
            raise ValueError("BM25 index required for keyword mode (re-build bundle with --prefer auto)")
        kw_scores = bm25_score(bm25, query)
        scores = kw_scores + (tb * 0.05 if query_tokens else 0.0)
    elif mode == "hybrid":
        bm25 = bundle.get("bm25")
        if bm25 is None:
            raise ValueError("BM25 index required for hybrid mode (re-build bundle with --prefer auto)")
        kw_scores = bm25_score(bm25, query)
        kw_max = kw_scores.max()
        kw_norm = kw_scores / kw_max if kw_max > 0 else kw_scores
        scores = 0.6 * boosted + 0.4 * kw_norm
    else:
        scores = boosted

    order = np.argsort(-scores)
    results = []
    for i in order[:top_k]:
        results.append({
            "id": bundle["id"][i],
            "score": float(scores[i]),
            "publication": bundle["pub"][i],
            "file": bundle["file"][i],
            "title": bundle["title"][i],
            "text": bundle["text"][i],
        })
    return results


def load_golden(golden_path: Path) -> list[dict[str, Any]]:
    """Load golden evaluation set (JSONL format)."""
    queries = []
    with golden_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def evaluate(bundle_dir: Path, golden_path: Path, top_k: int = 10, prefer: str = "auto", rerank: bool = False, mode: str = "vector") -> dict[str, Any]:
    """Run evaluation and return metrics."""
    print(f"Loading bundle from {bundle_dir}...")
    bundle = load_bundle(bundle_dir, prefer)

    reranker = None
    if rerank:
        print("Loading cross-encoder re-ranker...")
        reranker = CrossEncoderReranker()

    print(f"Loading golden set from {golden_path}...")
    golden = load_golden(golden_path)
    print(f"Evaluating {len(golden)} queries ({mode=})...")

    precisions = []
    mrrs = []

    for i, item in enumerate(golden):
        query = item["query"]
        expected_files = set(item.get("expected_files", []))
        expected_publications = set(item.get("expected_publications", []))

        results = search_bundle(bundle, query, top_k=100 if reranker else top_k, mode=mode)

        results = reranker.rerank(query, results, top_k=top_k) if reranker and results else results[:top_k]

        # Precision@k
        retrieved_files = {r['file'] for r in results}
        # Check file-level matches
        hits = retrieved_files & expected_files
        precision = len(hits) / top_k if top_k > 0 else 0
        precisions.append(precision)

        # MRR - first relevant result
        mrr = 0.0
        for rank, r in enumerate(results, 1):
            if r['file'] in expected_files:
                mrr = 1.0 / rank
                break
            # Also check publication-level match
            if r['publication'] in expected_publications:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(golden)} queries")

    mean_precision = np.mean(precisions) if precisions else 0
    std_precision = np.std(precisions) if precisions else 0
    mean_mrr = np.mean(mrrs) if mrrs else 0
    std_mrr = np.std(mrrs) if mrrs else 0

    return {
        "num_queries": len(golden),
        "top_k": top_k,
        "mode": mode,
        "mean_precision": float(mean_precision),
        "std_precision": float(std_precision),
        "mean_mrr": float(mean_mrr),
        "std_mrr": float(std_mrr),
        "reranked": rerank,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluate command."""
    p = argparse.ArgumentParser(description="Evaluate RAG quality (Precision@10, MRR)")
    p.add_argument("--bundle", type=Path, required=True, help="Path to RAG bundle")
    p.add_argument("--golden", type=Path, required=True, help="Path to golden set (JSONL)")
    p.add_argument("--top-k", type=int, default=10, help="Top-k for evaluation")
    p.add_argument("--prefer", choices=["auto", "apple", "nvidia", "cpu"], default="auto")
    p.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="vector",
                    help="Search mode for retrieval (default: vector)")
    p.add_argument("--rerank", action="store_true", help="Use cross-encoder re-ranker")
    p.add_argument("--output", type=Path, help="Output JSON file for results")
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    results = evaluate(
        args.bundle,
        args.golden,
        top_k=args.top_k,
        prefer=args.prefer,
        rerank=args.rerank,
        mode=args.mode,
    )

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Queries evaluated: {results['num_queries']}")
    print(f"  Top-k:             {results['top_k']}")
    print(f"  Mode:              {results['mode']}")
    print(f"  Re-ranked:         {results['reranked']}")
    print()
    print(f"  Precision@{results['top_k']}: {results['mean_precision']:.4f} \u00b1 {results['std_precision']:.4f}")
    print(f"  MRR:                {results['mean_mrr']:.4f} \u00b1 {results['std_mrr']:.4f}")
    print("=" * 50)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output}")

    return 0


def main() -> None:
    """Entry point: run evaluation and exit."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
