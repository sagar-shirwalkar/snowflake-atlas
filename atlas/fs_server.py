"""MCP server exposing a markdown documentation repo as a navigable filesystem.

Five tools: `list_publications`, `list_files`, `read_file`, `search`,
`get_release_info`. All deterministic, all backed by file I/O and
`ripgrep`. No model. No state. Drop-in for any markdown repo.

Five tools, no models, no embeddings. Backed by a local markdown repo
plus ripgrep for full-text search. This is the
"give the model the link" surface - deterministic, zero-infra,
reproducible across every client that speaks MCP.

The same server works for any markdown repo - swap the source adapter
and it'll happily expose a different docs site.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .log import configure_logging, get_logger
from .sources import GitSource, LocalSource, MarkdownSource, WebCrawlSource

app = Server("snowflake-fs")
logger = get_logger()


def create_source(
    source_type: str,
    repo_path: str | None = None,
    repo_url: str | None = None,
    branch: str | None = None,
    mirror_path: str | None = None,
    crawl_meta: str | None = None,
) -> MarkdownSource:
    """Create a :class:`MarkdownSource` adapter from CLI parameters."""
    if source_type == "git":
        if not repo_path or not repo_url or not branch:
            raise ValueError("git source requires --repo-path, --repo-url, --branch")
        return GitSource(Path(repo_path), repo_url, branch)
    elif source_type == "web-crawl":
        if not mirror_path:
            raise ValueError("web-crawl source requires --mirror-path")
        return WebCrawlSource(Path(mirror_path), Path(crawl_meta) if crawl_meta else None)
    elif source_type == "local":
        if not mirror_path:
            raise ValueError("local source requires --mirror-path")
        return LocalSource(Path(mirror_path), repo_url or "local://docs", branch or "local")
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def _parse_md(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1]) or {}
                if isinstance(loaded, dict):
                    meta = loaded
            except yaml.YAMLError:
                pass
            body = parts[2].strip()
    return {"frontmatter": meta, "content": body}


def list_publications(source: MarkdownSource) -> list[dict[str, Any]]:
    """List all documentation publications with file counts."""
    pubs: dict[str, int] = {}
    for f in source.walk_markdown():
        rel = f.relative_to(source.mirror_root if hasattr(source, 'mirror_root') else source.repo_path / "markdown")
        pub = rel.parts[0] if rel.parts else "unknown"
        pubs[pub] = pubs.get(pub, 0) + 1
    return [{"name": k, "file_count": v} for k, v in sorted(pubs.items())]


def list_publication_files(source: MarkdownSource, publication: str) -> list[dict[str, Any]]:
    """List all files in a publication with frontmatter metadata."""
    pub_dir = None
    if hasattr(source, 'mirror_root'):
        pub_dir = source.mirror_root / publication
    elif hasattr(source, 'repo_path'):
        pub_dir = source.repo_path / "markdown" / publication

    if not pub_dir or not pub_dir.is_dir():
        raise FileNotFoundError(f"Publication not found: {publication}")

    out: list[dict[str, Any]] = []
    for f in sorted(pub_dir.rglob("*.md")):
        parsed = _parse_md(f)
        meta = parsed["frontmatter"]
        try:
            rel = f.relative_to(pub_dir)
        except ValueError:
            rel = f
        out.append(
            {
                "file": str(rel),
                "title": meta.get("title", ""),
                "product_area": meta.get("product_area", ""),
                "last_updated": str(meta.get("last_updated", "")),
            }
        )
    return out


def read_publication_file(source: MarkdownSource, publication: str, file: str, max_chars: int = 50_000) -> dict[str, Any]:
    """Read a markdown file from a publication, with path traversal protection."""
    if hasattr(source, 'mirror_root'):
        base = source.mirror_root
    elif hasattr(source, 'repo_path'):
        base = source.repo_path / "markdown"
    else:
        raise ValueError("Unknown source type")

    # Resolve the base directory first so traversal can't escape it
    base = base.resolve()

    # Reject path traversal in publication: no ".." components, no absolute paths
    if ".." in publication.split("/") or publication.startswith("/"):
        raise ValueError("Invalid publication name")

    pub_root = (base / publication).resolve()
    # Verify pub_root is within the intended base directory
    if not str(pub_root).startswith(str(base) + "/"):
        raise ValueError("Path traversal detected in publication")

    target = (pub_root / file).resolve()
    # Verify target is within the publication root
    if not str(target).startswith(str(pub_root) + "/"):
        raise ValueError("Path traversal blocked")
    if not target.is_file():
        raise FileNotFoundError(f"{publication}/{file} not found")
    parsed = _parse_md(target)
    content = parsed["content"]
    truncated = False
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    return {
        "publication": publication,
        "file": file,
        "frontmatter": parsed["frontmatter"],
        "content": content,
        "truncated": truncated,
    }


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def get_release_info(source: MarkdownSource) -> dict[str, Any]:
    """Return the source release metadata (branch, SHA, file count)."""
    return source.get_release_info()


def full_text_search(
    source: MarkdownSource,
    query: str,
    scope: str | None = None,
    regex: bool = False,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Full-text search over the documentation using ripgrep.

    Args:
        source: The markdown source to search.
        query: Search query (literal or regex depending on ``regex`` flag).
        scope: Optional publication folder to restrict search to.
        regex: If True, treat ``query`` as a regex pattern.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts with ``file``, ``line``, and ``preview`` keys.

    """
    if not shutil.which("rg"):
        raise RuntimeError("ripgrep (rg) not installed. Install via `brew install ripgrep`.")

    if hasattr(source, 'mirror_root'):
        search_root = source.mirror_root
    elif hasattr(source, 'repo_path'):
        search_root = source.repo_path / "markdown"
    else:
        raise ValueError("Unknown source type")

    if scope:
        # Resolve base first, then verify scope stays within the doc tree
        search_root = search_root.resolve()
        candidate = (search_root / scope).resolve()
        if not str(candidate).startswith(str(search_root) + "/"):
            raise ValueError("Scope path traversal blocked")
        search_root = candidate

    if not search_root.is_dir():
        raise FileNotFoundError(f"Scope not found: {scope}")

    cmd = ["rg", "--no-heading", "--line-number", "--color", "never"]
    if not regex:
        cmd.append("--fixed-strings")
    # Use -- to prevent query from being interpreted as rg flags
    cmd.extend(["--max-count", "1", "--", query, str(search_root)])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    hits: list[dict[str, Any]] = []

    base_path = source.mirror_root if hasattr(source, 'mirror_root') else source.repo_path / "markdown"

    for line in proc.stdout.splitlines()[:max_results]:
        if ":" not in line:
            continue
        path_part, _, rest = line.partition(":")
        line_no_str, _, _ = rest.partition(":")
        try:
            line_no = int(line_no_str)
        except ValueError:
            continue
        file_path = Path(path_part)
        try:
            rel = file_path.relative_to(base_path)
        except ValueError:
            rel = file_path
        hits.append(
            {
                "file": str(rel),
                "line": line_no,
                "preview": _preview_line(file_path, line_no),
            }
        )
    return hits


def _preview_line(path: Path, line_no: int, context: int = 1) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(0, line_no - 1 - context)
        end = min(len(lines), line_no + context)
        return "".join(lines[start:end]).strip()
    except OSError:
        return ""


def _result(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Declare the five filesystem server tools (MCP list_tools handler)."""
    return [
        Tool(
            name="list_publications",
            description="List every documentation publication (top-level folder) available in the local mirror.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_files",
            description="List all files in a single publication, with title/product_area/last_updated from YAML frontmatter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "publication": {
                        "type": "string",
                        "description": "Publication folder name, e.g. 'user-guide'",
                    }
                },
                "required": ["publication"],
            },
        ),
        Tool(
            name="read_file",
            description="Read a single markdown file. Returns parsed YAML frontmatter and the body content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "publication": {"type": "string"},
                    "file": {
                        "type": "string",
                        "description": "Path relative to the publication folder, e.g. 'data-loading.md'",
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 50000,
                        "description": "Truncate the body to this many characters.",
                    },
                },
                "required": ["publication", "file"],
            },
        ),
        Tool(
            name="search",
            description="Full-text search (ripgrep) over the docs. Fast, deterministic, regex-capable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "description": "Optional: restrict to one publication folder.",
                    },
                    "regex": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_release_info",
            description="Return the current docs release metadata: branch, commit SHA, last commit date, total file count.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


_source: MarkdownSource | None = None


def set_source(src: MarkdownSource) -> None:
    """Set the global source adapter instance."""
    global _source
    _source = src


def get_source() -> MarkdownSource:
    """Return the global source adapter, raising if not set."""
    if _source is None:
        raise RuntimeError("Source not initialized. Call set_source() first.")
    return _source


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to the appropriate handler (MCP call_tool handler)."""
    cid = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    _start = time.perf_counter()
    logger.info("Tool called", tool=name)

    source = get_source()
    try:
        if name == "list_publications":
            return _result(list_publications(source))
        if name == "list_files":
            return _result(list_publication_files(source, arguments["publication"]))
        if name == "read_file":
            return _result(
                read_publication_file(
                    source,
                    arguments["publication"],
                    arguments["file"],
                    arguments.get("max_chars", 50_000),
                )
            )
        if name == "search":
            return _result(
                full_text_search(
                    source,
                    arguments["query"],
                    arguments.get("scope"),
                    arguments.get("regex", False),
                    arguments.get("max_results", 50),
                )
            )
        if name == "get_release_info":
            return _result(get_release_info(source))
    except (FileNotFoundError, ValueError) as e:
        logger.error("Tool failed", tool=name, error=str(e))
        return _result({"error": str(e)})
    finally:
        elapsed = (time.perf_counter() - _start) * 1000
        logger.info("Tool finished", tool=name, duration_ms=round(elapsed, 2))
    raise ValueError(f"Unknown tool: {name}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the filesystem MCP server."""
    p = argparse.ArgumentParser(description="Atlas filesystem MCP server")
    p.add_argument("--source-type", choices=["git", "web-crawl", "local"], default="git")
    p.add_argument("--repo-path", help="Path to git repo (git source)")
    p.add_argument("--repo-url", help="Git repo URL (git source)")
    p.add_argument("--branch", help="Git branch (git source)")
    p.add_argument("--mirror-path", help="Path to local mirror (web-crawl/local source)")
    p.add_argument("--crawl-meta", help="Path to crawl_meta.json (web-crawl source)")
    return p.parse_args()


async def serve() -> None:
    """Run the filesystem server over stdio (MCP transport)."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point: configure, parse args, and run the server."""
    configure_logging()
    args = parse_args()
    source = create_source(
        args.source_type,
        repo_path=args.repo_path,
        repo_url=args.repo_url,
        branch=args.branch,
        mirror_path=args.mirror_path,
        crawl_meta=args.crawl_meta,
    )
    set_source(source)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
