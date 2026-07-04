"""Build a portable RAG bundle from a markdown documentation source.

End-to-end bundle build. ``git pull`` the docs, walk every ``.md``,
chunk, embed, write ``chunks.parquet`` + ``embeddings.f16.npy`` +
``norms.f32.npy`` + ``model/`` + ``manifest.json``. Records the pinned
source SHA so re-runs are reproducible. Prints the chosen embedding
backend and the reason for the choice. Accepts
``--prefer {auto,apple,nvidia,cpu}``.

When the backend is ONNX+CPU (the CI path), the embedding step is
automatically parallelized across all available CPU cores. Each
worker loads its own ONNX session and processes a shard of the
chunks. This is transparent to the caller — ``_run()`` calls
:func:`embed_chunks` which dispatches to the parallel or sequential
path based on the backend.

Pipeline:
  1. ``git fetch`` the pinned docs branch (default:
     ``main``) into ``--repo-path`` (for git source).
  2. Walk every ``markdown/**/*.md`` file and H2-chunk it.
  3. Embed all chunks with ``Xenova/bge-base-en-v1.5`` (ONNX) in
     batches with progress and retries.
  4. Persist:
       - chunks.parquet   (text + metadata)
       - embeddings.f16.npy
       - norms.f32.npy
       - model/           (ONNX model + tokenizer)
       - manifest.json    (provenance + integrity metadata)
  5. Print a summary and a SHA256 of each artifact.

The output is a self-contained directory that any MCP client can
point ``snowflake-rag --bundle <dir>`` at. No GPU, no Ollama,
no torch required at runtime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .bm25_search import build_fielded_index as build_bm25_index
from .bm25_search import save_index as save_bm25_index
from .chunk import chunk_file
from .embed import (
    DEFAULT_MODEL_ID,
    get_embedder,
    resolve_backend,
)
from .log import configure_logging, get_logger
from .sources import GitSource, LocalSource, MarkdownSource, WebCrawlSource

logger = get_logger()

# ---------------------------------------------------------------------------
# Build-time constants
# ---------------------------------------------------------------------------

_CLUSTER_BOOST_WEIGHT = 0.03  # sibling/path boost at query time

DEFAULT_BRANCH = "main"
DEFAULT_LOCAL_PATH = "./data/snowflake-docs"

BUNDLE_SCHEMA_VERSION = 1


def _embed_shard(model_id: str, prefer: str, texts: list[str], batch_size: int) -> np.ndarray:
    """Embed a shard of texts in a worker process.

    Each worker loads its own ONNX session from the same cached model
    files. Called by :func:`embed_chunks` via ``ProcessPoolExecutor``.
    """
    embedder = get_embedder(model_id, prefer=prefer)
    return embedder.embed_with_progress(texts, batch_size=batch_size, show_progress=False)


def embed_chunks(texts: list[str], model_id: str, prefer: str, backend: str, batch_size: int) -> np.ndarray:
    """Embed all chunks, parallelizing ONNX+CPU across CPU cores.

    Sequential backends (MLX, CUDA) run as-is. ONNX+CPU splits the
    texts into ``os.cpu_count()`` shards and embeds each in a
    separate process, concatenating the results.
    """
    if backend != "onnx-cpu":
        embedder = get_embedder(model_id, prefer=prefer)
        return embedder.embed_with_progress(texts, batch_size=batch_size)

    n_workers = min(os.cpu_count() or 2, 4)
    logger.info("Parallel embedding across CPU workers", workers=n_workers)
    n = len(texts)
    if n_workers <= 1 or n < n_workers * batch_size:
        embedder = get_embedder(model_id, prefer=prefer)
        return embedder.embed_with_progress(texts, batch_size=batch_size)

    shard_size = math.ceil(n / n_workers)
    shards = [texts[i * shard_size : (i + 1) * shard_size] for i in range(n_workers)]
    shards = [s for s in shards if s]

    results: list[np.ndarray] = [None] * len(shards)  # type: ignore[list-item]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(shards)) as pool:
        futs = {
            pool.submit(_embed_shard, model_id, prefer, s, batch_size): i
            for i, s in enumerate(shards)
        }
        for future in concurrent.futures.as_completed(futs):
            idx = futs[future]
            results[idx] = future.result()
            done = sum(1 for r in results if r is not None)
            logger.info("Shard complete", done=done, total=len(shards))

    if any(r is None for r in results):
        raise RuntimeError(f"Parallel embedding failed for {sum(1 for r in results if r is None)} shard(s)")
    return np.vstack(results)


def _git_retry(cmd: list[str], max_attempts: int = 3) -> None:
    """Run a subprocess with retry + exponential backoff + timeout.

    Git network operations (fetch, clone) fail transiently. This
    helper retries up to ``max_attempts`` times with 2^attempt
    backoff and caps each attempt at 120 seconds.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, check=True, timeout=120)
            return
        except (subprocess.CalledProcessError, OSError, TimeoutError) as e:
            if attempt == max_attempts:
                raise
            wait = 2.0**attempt
            logger.warning("Git command failed, retrying", attempt=attempt, max_attempts=max_attempts, error=str(e), wait=wait)
            time.sleep(wait)


def ensure_repo(repo_path: Path, repo_url: str, branch: str) -> Path:
    """Clone or update the docs repo at the pinned branch."""
    if repo_path.exists() and (repo_path / ".git").is_dir():
        logger.info("Fetching latest branch", branch=branch, path=str(repo_path))
        _git_retry(["git", "-C", str(repo_path), "fetch", "origin", branch])
        subprocess.run(
            ["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch}"],
            check=True,
            timeout=30,
        )
    else:
        logger.info("Cloning repo", url=repo_url, branch=branch, path=str(repo_path))
        _git_retry(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                str(repo_path),
            ]
        )
    return repo_path.resolve()


def current_sha(repo_path: Path) -> str:
    """Return the current git HEAD SHA for a repository."""
    return subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def create_source(args: argparse.Namespace) -> MarkdownSource:
    """Create a source adapter from CLI arguments."""
    if args.source_type == "git":
        repo_path = ensure_repo(Path(args.repo_path), args.repo_url, args.branch)
        sha = current_sha(repo_path)
        logger.info("Pinned to SHA", sha=sha)
        return GitSource(repo_path, args.repo_url, args.branch)
    elif args.source_type == "web-crawl":
        return WebCrawlSource(Path(args.mirror_path), Path(args.crawl_meta) if args.crawl_meta else None)
    elif args.source_type == "local":
        return LocalSource(Path(args.mirror_path), args.repo_url or "local://docs", args.branch or "local")
    else:
        raise ValueError(f"Unknown source type: {args.source_type}")


def walk_markdown(source: MarkdownSource) -> list[Path]:
    """Walk the source and return all markdown file paths."""
    return sorted(source.walk_markdown())


def build_chunk_table(files: list[Path], source: MarkdownSource) -> pa.Table:
    """Build the chunk table with metadata, including hierarchical cluster tags.

    ``cluster_tags`` is a space-separated string of sibling document stems
    in the same directory (excluding the document itself).  At query time
    the RAG server does substring matching on this field to apply a small
    relevance boost — a lightweight form of path-aware retrieval that
    surfaces related docs from the same "neighbourhood".
    """
    repo_root = source.mirror_root if hasattr(source, 'mirror_root') else source.repo_path

    # Pre-compute sibling map: dir_key -> set of .md stem names
    sibling_map: dict[str, set[str]] = {}
    for path in files:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 2 or parts[0] != "markdown":
            continue
        # dir_key = path components between markdown/ and the filename
        dir_key = "/".join(parts[1:-1]) if len(parts) > 2 else ""
        sib_set = sibling_map.get(dir_key)
        if sib_set is None:
            sib_set = set()
            sibling_map[dir_key] = sib_set
        sib_set.add(path.stem)

    cols: dict[str, list] = {
        "id": [],
        "text": [],
        "publication": [],
        "file": [],
        "heading": [],
        "is_code": [],
        "title": [],
        "product_area": [],
        "last_updated": [],
        "canonical_url": [],
        "cluster_tags": [],
    }
    for i, path in enumerate(files, 1):
        try:
            chunks = chunk_file(path, repo_root)
        except Exception as e:
            logger.warning("Chunking failed", file=str(path), error=str(e))
            continue

        # Compute cluster_tags for this file from the sibling map
        try:
            rel = path.relative_to(repo_root)
            parts = rel.parts
            dir_key = "/".join(parts[1:-1]) if len(parts) > 2 else ""
            siblings = sibling_map.get(dir_key, set())
            sibling_stems = sorted(s for s in siblings if s != path.stem)
            cluster_tags = " ".join(sibling_stems)
        except (ValueError, KeyError):
            cluster_tags = ""

        for j, c in enumerate(chunks):
            text = c["text"]
            if not text.strip():
                continue
            meta = source.get_metadata(path)
            cols["id"].append(f"{meta['publication']}/{meta['file']}#{j}")
            cols["text"].append(text)
            cols["publication"].append(meta["publication"])
            cols["file"].append(meta["file"])
            cols["heading"].append(c["heading"])
            cols["is_code"].append(c["is_code"])
            cols["title"].append(c["frontmatter"].get("title", ""))
            cols["product_area"].append(c["frontmatter"].get("product_area", ""))
            cols["last_updated"].append(str(c["frontmatter"].get("last_updated", "")))
            cols["canonical_url"].append(c["frontmatter"].get("canonical_url", ""))
            cols["cluster_tags"].append(cluster_tags)
        if i % 500 == 0:
            logger.info("Chunking progress", files=i, total=len(files), chunks=len(cols["text"]))
    return pa.table(cols)


def stage_model(model_dir: Path, bundle_dir: Path) -> Path:
    """Copy the ONNX model + tokenizer into the bundle.

    The Xenova repo stores the model under ``onnx/`` and tokenizer
    files at the root. We mirror that layout so the runtime can
    load it the same way it would load from Hugging Face.
    """
    target = bundle_dir / "model"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    onnx_src = model_dir / "onnx"
    if onnx_src.is_dir():
        shutil.copytree(onnx_src, target / "onnx")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
    return target


def save_bundle_artifacts(
    embeddings: np.ndarray,
    output_dir: Path,
    dtype: str = "float16",
) -> tuple[Path, Path]:
    """Write embeddings and precomputed norms to disk.

    Float16 halves the on-disk size (~360MB -> ~180MB for 250k
    chunks) with negligible effect on retrieval quality because
    cosine similarity is rank-preserving under half precision.
    Norms are stored separately in float32 for accurate
    cosine-at-query-time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if dtype == "float16":
        emb_path = output_dir / "embeddings.f16.npy"
        emb_to_save = embeddings.astype(np.float16)
    elif dtype == "float32":
        emb_path = output_dir / "embeddings.f32.npy"
        emb_to_save = embeddings.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    np.save(emb_path, emb_to_save)
    norms = np.linalg.norm(emb_to_save.astype(np.float32), axis=1).astype(np.float32)
    norms_path = output_dir / "norms.f32.npy"
    np.save(norms_path, norms)
    return emb_path, norms_path


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    bundle_dir: Path,
    source: MarkdownSource,
    chunk_count: int,
    model_id: str,
    embedding_dim: int = 768,
    embedding_backend: str = "",
    embedding_active_provider: str = "",
) -> Path:
    """Write the bundle manifest.json with metadata and artifact SHA256s."""
    release_info = source.get_release_info()
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_repo": release_info["repo_url"],
        "source_branch": release_info["branch"],
        "source_sha": release_info["sha"],
        "source_published": None,
        "built_at": datetime.now(UTC).isoformat(),
        "chunk_count": chunk_count,
        "embedding_model": model_id,
        "embedding_dim": embedding_dim,
        "artifacts": {
            "chunks": "chunks.parquet",
            "embeddings": "embeddings.f16.npy",
            "norms": "norms.f32.npy",
            "model_dir": "model/",
            "bm25_index": "bm25.pkl",
        },
    }
    if embedding_backend:
        manifest["embedding_backend"] = embedding_backend
    if embedding_active_provider:
        manifest["embedding_active_provider"] = embedding_active_provider
    for key, rel in list(manifest["artifacts"].items()):
        if key == "model_dir":
            continue
        p = bundle_dir / rel
        if p.exists():
            manifest["artifacts"][f"{key}_sha256"] = sha256_file(p)
    out = bundle_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def _check_manifest_source(manifest_path: Path) -> None:
    """Verify the persisted manifest points to Snowflake docs."""
    try:
        manifest = json.loads(manifest_path.read_text())
        source_repo = manifest.get("source_repo", "")
        expected = "https://docs.snowflake.com"
        if source_repo != expected:
            logger.warning(
                "Manifest source_repo does not match expected Snowflake docs URL",
                expected=expected,
                got=source_repo,
                hint="Check that --mirror-path or --repo-url points to the correct docs source",
            )
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not verify manifest source", error=str(e))


def _print_summary(title: str, output_dir: Path, chunk_count: int, dim: int | None = None) -> None:
    """Print the final build summary banner."""
    manifest_path = output_dir / "manifest.json"
    source_repo = "?"
    if manifest_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            source_repo = json.loads(manifest_path.read_text()).get("source_repo", "?")
    print("\n" + "=" * 60)
    print(f"  {title}")
    print(f"    source   : {source_repo}")
    print(f"    chunks   : {chunk_count}")
    if dim is not None:
        print(f"    dim      : {dim}")
    print(f"    path     : {output_dir.resolve()}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the bundle build command."""
    p = argparse.ArgumentParser(description="Build the Atlas RAG bundle")
    p.add_argument("--source-type", choices=["git", "web-crawl", "local"], default="web-crawl", help="Source type")

    # Git source args
    p.add_argument("--repo-path", default=DEFAULT_LOCAL_PATH, help="Local path to git repo (git source)")
    p.add_argument("--repo-url", default="", help="Git repo URL (git source)")
    p.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch (git source)")

    # Web-crawl/local source args
    p.add_argument("--mirror-path", help="Path to local mirror (web-crawl/local source)")
    p.add_argument("--crawl-meta", help="Path to crawl_meta.json (web-crawl source)")

    # Shared args
    p.add_argument("--output", required=True, type=Path, help="Output bundle directory")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id or local path")
    p.add_argument("--limit", type=int, default=0, help="Limit number of files (for smoke tests)")
    p.add_argument("--skip-embed", action="store_true", help="Skip embedding (for chunk-only smoke tests)")
    p.add_argument(
        "--prefer",
        choices=["auto", "apple", "nvidia", "cpu"],
        default="auto",
        help="Embedding backend preference: apple=MLX, nvidia=CUDA, cpu=ONNX+CPU, auto=probe",
    )
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    configure_logging()

    source = create_source(args)
    files = walk_markdown(source)
    if args.limit:
        files = files[: args.limit]
    logger.info("Found markdown files", count=len(files))

    table = build_chunk_table(files, source)
    logger.info("Built chunks", count=len(table))

    args.output.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output / "chunks.parquet"
    pq.write_table(table, chunks_path)
    logger.info("Wrote chunks", path=str(chunks_path))

    # Build fielded BM25 keyword index from chunk texts + metadata
    texts = table.column("text").to_pylist()
    titles = [t or "" for t in table.column("title").to_pylist()]
    headings = [h or "" for h in table.column("heading").to_pylist()]
    # File path as a BM25 field — stems like "create-warehouse" are strong relevance signals
    file_paths = [f.replace(".md", "") if f else "" for f in table.column("file").to_pylist()]
    field_texts = {"text": texts, "title": titles, "heading": headings, "file_path": file_paths}
    field_weights = {"text": 1.0, "title": 3.0, "heading": 2.0, "file_path": 2.5}
    logger.info("Building fielded BM25 index", count=len(texts), fields=list(field_texts.keys()))
    bm25_index = build_bm25_index(field_texts, field_weights)
    bm25_path = args.output / "bm25.pkl"
    save_bm25_index(bm25_index, bm25_path)
    logger.info("Wrote BM25 index", path=str(bm25_path), size=f"{bm25_path.stat().st_size / 1024:.1f} KB")

    if args.skip_embed:
        logger.info("Skipping embedding (--skip-embed set)")
        write_manifest(
            args.output,
            source,
            len(table),
            args.model,
            embedding_dim=768,
        )
        _check_manifest_source(args.output / "manifest.json")
        _print_summary("Bundle ready (--skip-embed).", args.output, len(table))
        return 0

    backend, reason = resolve_backend(args.prefer)
    batch_size = 256 if backend == "onnx-cpu" else 32
    logger.info("Embedding backend selected", backend=backend, reason=reason)
    logger.info("Loading model", model=args.model)
    embedder = get_embedder(args.model, prefer=args.prefer)
    logger.info("Active provider", provider=embedder.active_provider)
    logger.info("Embedding dim", dim=embedder.dim)
    embeddings = embed_chunks(
        table.column("text").to_pylist(),
        args.model,
        args.prefer,
        backend,
        batch_size,
    )

    emb_path, norms_path = save_bundle_artifacts(embeddings, args.output)
    logger.info("Wrote embeddings", path=str(emb_path))
    logger.info("Wrote norms", path=str(norms_path))

    stage_model(embedder.resolved_dir, args.output)
    logger.info("Staged model files", path=str(args.output / "model"))

    manifest_path = write_manifest(
        args.output,
        source,
        len(table),
        args.model,
        embedding_dim=embedder.dim,
        embedding_backend=backend,
        embedding_active_provider=str(embedder.active_provider),
    )
    logger.info("Wrote manifest", path=str(manifest_path))

    _check_manifest_source(manifest_path)
    _print_summary("Bundle ready.", args.output, len(table), dim=embeddings.shape[1])
    return 0


def main() -> None:
    """Script entry point.

    Calls ``sys.exit(_run())`` so the return code propagates through
    both ``python -m`` and the console-script entry points defined
    in ``pyproject.toml``. Console scripts do not propagate return
    values on their own, so the ``sys.exit`` has to live here.
    """
    sys.exit(_run())


if __name__ == "__main__":
    main()
