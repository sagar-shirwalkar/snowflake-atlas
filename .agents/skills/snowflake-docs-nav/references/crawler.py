# Snowflake Docs Crawler

Python script to mirror all Snowflake documentation locally as markdown files, preserving the folder structure for use with `atlas-build`.

## crawler.py

```python
#!/usr/bin/env python3
"""
Snowflake Documentation Crawler

Mirrors all docs from https://docs.snowflake.com/ to a local markdown/ directory
compatible with atlas-build (expects markdown/<publication>/*.md structure).

Usage:
    python crawler.py --output ./data/snowflake-docs --max-pages 100
    python crawler.py --output ./data/snowflake-docs --sections cortex-ai,sql-functions
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiofiles
import aiohttp
from bs4 import BeautifulSoup


# Configuration
ROOT_LLMSTXT = "https://docs.snowflake.com/llms.txt"
BASE_URL = "https://docs.snowflake.com/en/"
USER_AGENT = "SnowflakeAtlasBot/1.0 (+https://github.com/yourorg/snowflake-atlas)"
DEFAULT_DELAY = 0.5  # seconds between requests
MAX_CONCURRENT = 5
TIMEOUT = 30


# Section mapping from root llms.txt (see references/section-map.md)
SECTIONS = {
    "general": "reference.md",
    "user-guide": "user-guide/llms.txt",
    "loading-data": "user-guide/data-integration/llms.txt",
    "cortex-ai": "user-guide/snowflake-cortex/llms.txt",
    "cortex-code": "user-guide/cortex-code/llms.txt",
    "clean-rooms": "user-guide/cleanrooms/llms.txt",
    "snowsight": "user-guide/ui-snowsight/llms.txt",
    "snowflake-postgres": "user-guide/snowflake-postgres/llms.txt",
    "sql-functions": "sql-reference/functions/llms.txt",
    "sql-commands": "sql-reference/sql/llms.txt",
    "account-usage": "sql-reference/account-usage/llms.txt",
    "org-usage": "sql-reference/organization-usage/llms.txt",
    "info-schema": "sql-reference/info-schema/llms.txt",
    "sql-classes": "sql-reference/classes/llms.txt",
    "sql-general": "sql-reference/llms.txt",
    "connectors": "connectors/llms.txt",
    "collaboration": "collaboration/llms.txt",
    "migrations": "migrations/llms.txt",
    "release-notes": "release-notes/llms.txt",
    "programmatic-access": "progaccess/llms.txt",
    "developer-guide": "developer-guide/llms.txt",
    "snowpark": "developer-guide/snowpark/llms.txt",
    "snowflake-ml": "developer-guide/snowflake-ml/llms.txt",
    "native-apps": "developer-guide/native-apps/llms.txt",
    "streamlit": "developer-guide/streamlit/llms.txt",
    "snowflake-cli": "developer-guide/snowflake-cli/llms.txt",
    "snowpark-containers": "developer-guide/snowpark-container-services/llms.txt",
    "rest-api": "developer-guide/snowflake-rest-api/llms.txt",
}


class SnowflakeCrawler:
    def __init__(
        self,
        output_dir: Path,
        sections: Optional[list[str]] = None,
        max_pages: Optional[int] = None,
        delay: float = DEFAULT_DELAY,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.sections = sections or list(SECTIONS.keys())
        self.max_pages = max_pages
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {"fetched": 0, "skipped": 0, "errors": 0, "bytes": 0}
        self._last_request = 0.0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": USER_AGENT},
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    async def _fetch(self, url: str) -> Optional[str]:
        await self._rate_limit()
        async with self.semaphore:
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 404:
                        print(f"  404: {url}")
                    else:
                        print(f"  HTTP {resp.status}: {url}")
            except Exception as e:
                print(f"  Error fetching {url}: {e}")
        return None

    def _parse_llms_txt(self, content: str, base_url: str) -> list[str]:
        """Extract .md page URLs from llms.txt content."""
        urls = []
        for line in content.splitlines():
            line = line.strip()
            # Match markdown links: - [Title](url)
            match = re.match(r"- \[.*?\]\((.*?)\)", line)
            if match:
                url = match.group(1)
                # Convert to absolute URL
                absolute = urljoin(base_url, url)
                # Only keep .md files (skip other llms.txt references)
                if absolute.endswith(".md"):
                    urls.append(absolute)
        return urls

    def _extract_frontmatter_and_body(self, content: str) -> tuple[dict, str]:
        """Parse YAML frontmatter from markdown."""
        frontmatter = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    if not isinstance(frontmatter, dict):
                        frontmatter = {}
                except Exception:
                    frontmatter = {}
                body = parts[2].strip()
        return frontmatter, body

    def _local_path(self, url: str) -> Path:
        """Convert docs URL to local file path under markdown/."""
        # Remove base URL
        parsed = urlparse(url)
        path = parsed.path
        # Remove /en/ prefix
        if path.startswith("/en/"):
            path = path[4:]
        # Remove leading slash
        path = path.lstrip("/")
        # Determine publication (first path component)
        parts = path.split("/")
        if len(parts) >= 2:
            publication = parts[0]
            file_path = "/".join(parts[1:])
        else:
            publication = "general"
            file_path = path
        return self.output_dir / "markdown" / publication / file_path

    async def _fetch_and_save_page(self, url: str) -> bool:
        """Fetch a single page and save to local mirror."""
        local_path = self._local_path(url)
        if local_path.exists():
            self.stats["skipped"] += 1
            return True

        content = await self._fetch(url)
        if content is None:
            self.stats["errors"] += 1
            return False

        # Ensure directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        async with aiofiles.open(local_path, "w", encoding="utf-8") as f:
            await f.write(content)

        self.stats["fetched"] += 1
        self.stats["bytes"] += len(content.encode("utf-8"))

        # Progress
        if self.stats["fetched"] % 50 == 0:
            print(f"  Fetched {self.stats['fetched']} pages...")

        return True

    async def crawl_section(self, section_key: str) -> int:
        """Crawl all pages in a section."""
        if section_key not in SECTIONS:
            print(f"  Unknown section: {section_key}")
            return 0

        section_path = SECTIONS[section_key]
        section_url = urljoin(BASE_URL, section_path)

        print(f"\n=== Crawling section: {section_key} ===")
        print(f"  Index: {section_url}")

        # Fetch section llms.txt
        index_content = await self._fetch(section_url)
        if index_content is None:
            print(f"  Failed to fetch section index")
            return 0

        # Parse page URLs
        page_urls = self._parse_llms_txt(index_content, section_url)
        print(f"  Found {len(page_urls)} pages")

        if self.max_pages:
            page_urls = page_urls[: self.max_pages]
            print(f"  Limited to {len(page_urls)} pages")

        # Fetch all pages concurrently
        tasks = [self._fetch_and_save_page(url) for url in page_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(1 for r in results if r is True)
        errors = sum(1 for r in results if r is False or isinstance(r, Exception))
        print(f"  Completed: {success} fetched, {errors} errors")

        return success

    async def crawl_all(self) -> dict:
        """Crawl all selected sections."""
        print(f"Starting crawl to {self.output_dir}")
        print(f"Sections: {', '.join(self.sections)}")

        total_fetched = 0
        for section in self.sections:
            if self.max_pages and total_fetched >= self.max_pages:
                break
            fetched = await self.crawl_section(section)
            total_fetched += fetched

        print(f"\n=== Crawl Complete ===")
        print(f"  Fetched: {self.stats['fetched']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"  Errors:  {self.stats['errors']}")
        print(f"  Bytes:   {self.stats['bytes'] / 1e6:.1f} MB")

        return self.stats


def main():
    parser = argparse.ArgumentParser(description="Crawl Snowflake documentation")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--sections", help="Comma-separated list of sections to crawl")
    parser.add_argument("--max-pages", type=int, help="Maximum pages per section")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between requests (seconds)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent requests")
    args = parser.parse_args()

    sections = args.sections.split(",") if args.sections else None

    async def run():
        async with SnowflakeCrawler(
            output_dir=args.output,
            sections=sections,
            max_pages=args.max_pages,
            delay=args.delay,
            max_concurrent=args.max_concurrent,
        ) as crawler:
            await crawler.crawl_all()

    asyncio.run(run())


if __name__ == "__main__":
    main()
```

## Usage Examples

```bash
# Crawl all docs (will take a while - 6800+ pages)
python crawler.py --output ./data/snowflake-docs

# Crawl only Cortex AI and SQL Functions (for testing)
python crawler.py --output ./data/snowflake-docs --sections cortex-ai,sql-functions --max-pages 50

# Faster crawl with more concurrency (be respectful!)
python crawler.py --output ./data/snowflake-docs --max-concurrent 10 --delay 0.2
```

## After Crawling

The output structure will be:
```
data/snowflake-docs/
└── markdown/
    ├── user-guide/
    │   ├── getting-started-for-users.md
    │   ├── concepts-for-administrators.md
    │   └── ...
    ├── sql-reference/
    │   ├── functions/
    │   │   ├── abs.md
    │   │   ├── ai_complete.md
    │   │   └── ...
    │   └── sql/
    │       ├── create-table.md
    │       └── ...
    └── ...
```

Then run the standard atlas build:
```bash
atlas-build --repo-path ./data/snowflake-docs --output ./data/snowflake-rag-bundle
```

## Dependencies

```bash
uv add aiohttp aiofiles pyyaml beautifulsoup4
```
