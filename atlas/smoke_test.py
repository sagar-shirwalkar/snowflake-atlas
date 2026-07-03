"""End-to-end smoke test for Atlas MCP servers.

``atlas-smoke`` console script. Runs a quick validation of both
the filesystem and RAG servers against a real bundle and repo.
Exits non-zero on any failure. Designed to run in ~1-2 minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .fs_server import set_source
from .rag_server import _bundle_cache
from .sources import GitSource, LocalSource, WebCrawlSource


async def test_fs_server(source, verbose: bool = False) -> list[str]:
    """Test filesystem server tools."""
    errors = []
    set_source(source)

    # Test list_publications
    try:
        from .fs_server import list_publications
        pubs = list_publications(source)
        if verbose:
            print(f"  list_publications: {len(pubs)} publications")
        assert len(pubs) > 0, "No publications found"
    except Exception as e:
        errors.append(f"list_publications failed: {e}")

    # Test list_files on first publication
    try:
        from .fs_server import list_publication_files
        if pubs:
            pub_name = pubs[0]["name"]
            files = list_publication_files(source, pub_name)
            if verbose:
                print(f"  list_files({pub_name}): {len(files)} files")
            assert len(files) > 0, f"No files in {pub_name}"
    except Exception as e:
        errors.append(f"list_files failed: {e}")

    # Test read_file on first file
    try:
        from .fs_server import read_publication_file
        if pubs and files:
            file_name = files[0]["file"]
            result = read_publication_file(source, pubs[0]["name"], file_name)
            if verbose:
                print(f"  read_file: {len(result['content'])} chars")
            assert "content" in result
    except Exception as e:
        errors.append(f"read_file failed: {e}")

    # Test search
    try:
        from .fs_server import full_text_search
        results = full_text_search(source, "SELECT", max_results=5)
        if verbose:
            print(f"  search: {len(results)} hits")
    except Exception as e:
        errors.append(f"search failed: {e}")

    # Test get_release_info
    try:
        from .fs_server import get_release_info
        info = get_release_info(source)
        if verbose:
            print(f"  release_info: {info.get('branch', '?')} @ {info.get('sha', '?')[:8]}")
    except Exception as e:
        errors.append(f"get_release_info failed: {e}")

    return errors


async def test_rag_server(bundle_path: Path, verbose: bool = False) -> list[str]:
    """Test RAG server tools."""
    errors = []

    try:
        bundle = await _bundle_cache(str(bundle_path), "auto")
    except Exception as e:
        errors.append(f"Bundle load failed: {e}")
        return errors

    # Test search_docs
    try:
        results = bundle.search("how to create table", top_k=5)
        if verbose:
            print(f"  search_docs: {len(results)} results")
        assert len(results) > 0, "No results for 'how to create table'"
    except Exception as e:
        errors.append(f"search_docs failed: {e}")

    # Test search_code
    try:
        results = bundle.search("CREATE TABLE", top_k=5, is_code=True)
        if verbose:
            print(f"  search_code: {len(results)} results")
    except Exception as e:
        errors.append(f"search_code failed: {e}")

    # Test get_chunk
    try:
        if results:
            chunk = bundle.get_chunk(results[0]["id"])
            if verbose:
                print(f"  get_chunk: {chunk['id'][:50]}...")
            assert chunk is not None
    except Exception as e:
        errors.append(f"get_chunk failed: {e}")

    # Test get_bundle_info
    try:
        info = bundle.manifest
        if verbose:
            print(f"  bundle_info: {info.get('chunk_count')} chunks")
        assert info.get("chunk_count", 0) > 0
    except Exception as e:
        errors.append(f"get_bundle_info failed: {e}")

    return errors


async def _run_smoke_test(
    bundle_path: Path,
    repo_path: Path | None = None,
    source_type: str = "git",
    repo_url: str | None = None,
    branch: str | None = None,
    mirror_path: Path | None = None,
    crawl_meta: Path | None = None,
    verbose: bool = False,
) -> int:
    print("=" * 60)
    print("  Atlas Smoke Test")
    print("=" * 60)

    # Create source
    if source_type == "git":
        if not repo_path or not repo_url or not branch:
            print("  ERROR: git source requires --repo-path, --repo-url, --branch")
            return 1
        source = GitSource(repo_path, repo_url, branch)
    elif source_type == "web-crawl":
        if not mirror_path:
            print("  ERROR: web-crawl source requires --mirror-path")
            return 1
        source = WebCrawlSource(mirror_path, crawl_meta)
    elif source_type == "local":
        if not mirror_path:
            print("  ERROR: local source requires --mirror-path")
            return 1
        source = LocalSource(mirror_path)
    else:
        print(f"  ERROR: unknown source type: {source_type}")
        return 1

    all_errors = []

    print("\n[1/2] Testing filesystem server...")
    fs_errors = await test_fs_server(source, verbose)
    all_errors.extend([f"FS: {e}" for e in fs_errors])
    if fs_errors:
        print(f"  FAILED: {len(fs_errors)} error(s)")
    else:
        print("  PASSED")

    print("\n[2/2] Testing RAG server...")
    rag_errors = await test_rag_server(bundle_path, verbose)
    all_errors.extend([f"RAG: {e}" for e in rag_errors])
    if rag_errors:
        print(f"  FAILED: {len(rag_errors)} error(s)")
    else:
        print("  PASSED")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"  SMOKE TEST FAILED: {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"    - {e}")
        return 1
    else:
        print("  SMOKE TEST PASSED")
        return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke test command."""
    p = argparse.ArgumentParser(description="Run Atlas smoke tests")
    p.add_argument("--bundle", type=Path, required=True, help="Path to RAG bundle")
    p.add_argument("--source-type", choices=["git", "web-crawl", "local"], default="git")
    p.add_argument("--repo-path", type=Path, help="Path to git repo (git source)")
    p.add_argument("--repo-url", help="Git repo URL (git source)")
    p.add_argument("--branch", help="Git branch (git source)")
    p.add_argument("--mirror-path", type=Path, help="Path to local mirror (web-crawl/local source)")
    p.add_argument("--crawl-meta", type=Path, help="Path to crawl_meta.json (web-crawl source)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    return asyncio.run(_run_smoke_test(
        bundle_path=args.bundle,
        source_type=args.source_type,
        repo_path=args.repo_path,
        repo_url=args.repo_url,
        branch=args.branch,
        mirror_path=args.mirror_path,
        crawl_meta=args.crawl_meta,
        verbose=args.verbose,
    ))


def main() -> None:
    """Entry point: run smoke tests and exit with status code."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
