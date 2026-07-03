#!/usr/bin/env python3
"""Snowflake Documentation Crawler — Stealth Edition

Mirrors all docs from https://docs.snowflake.com/ to a local markdown/ directory
compatible with atlas-build (expects markdown/<publication>/*.md structure).

Two engines:

  aiohttp (default)  — async HTTP, optionally with rotating User-Agent pool,
                       jittered delays, and realistic browser headers.

  camoufox           — full Firefox browser via Camoufox with anti-detection
                       patches. Slower (~3-5 s/page) but virtually undetectable.

Usage:
    python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs
    python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs --stealth
    python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs --engine camoufox
    python -m snowflake_docs_nav.crawler --output ./data/snowflake-docs --sections cortex-ai,sql-functions --max-pages 50
"""

import argparse
import asyncio
import datetime
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiofiles
import aiohttp
import yaml
from bs4 import BeautifulSoup

# =========================================================================
# Browser profiles for stealth mode
# =========================================================================
# Each profile bundles a realistic User-Agent with internally consistent
# headers (Sec-CH-UA, Accept, Accept-Language, etc.) that match the
# claimed browser brand, version, and operating system.  Accept-Encoding,
# Sec-Fetch-*, DNT, and Upgrade-Insecure-Requests are applied uniformly
# in _build_request_headers().
# =========================================================================

_BROWSER_PROFILES = [
    # Chrome 126 on macOS
    {
        "name": "chrome-mac-126",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-CH-UA": (
                '"Google Chrome";v="126", "Chromium";v="126", '
                '"Not.A/Brand";v="24"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"macOS"',
        },
    },
    # Chrome 125 on Windows 11
    {
        "name": "chrome-win-125",
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-CH-UA": (
                '"Google Chrome";v="125", "Chromium";v="125", '
                '"Not.A/Brand";v="24"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
        },
    },
    # Chrome 126 on Linux
    {
        "name": "chrome-linux-126",
        "user_agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-CH-UA": (
                '"Google Chrome";v="126", "Chromium";v="126", '
                '"Not.A/Brand";v="24"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Linux"',
        },
    },
    # Firefox 128 on macOS
    {
        "name": "firefox-mac-128",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) "
            "Gecko/20100101 Firefox/128.0"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            # Firefox does not send Sec-CH-UA
        },
    },
    # Firefox 127 on Windows 11
    {
        "name": "firefox-win-127",
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) "
            "Gecko/20100101 Firefox/127.0"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
        },
    },
    # Safari 17.6 on macOS
    {
        "name": "safari-mac-17_6",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.6 Safari/605.1.15"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            # Safari does not send Sec-CH-UA
        },
    },
    # Edge 126 on macOS
    {
        "name": "edge-mac-126",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36 "
            "Edg/126.0.0.0"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-CH-UA": (
                '"Microsoft Edge";v="126", "Chromium";v="126", '
                '"Not.A/Brand";v="24"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"macOS"',
        },
    },
    # Chrome 126 on macOS (alternate minor version → variety)
    {
        "name": "chrome-mac-126-b",
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.6478.127 Safari/537.36"
        ),
        "headers": {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-CH-UA": (
                '"Google Chrome";v="126", "Chromium";v="126", '
                '"Not.A/Brand";v="24"'
            ),
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"macOS"',
        },
    },
]


# =========================================================================
# Configuration
# =========================================================================

POLITE_USER_AGENT = (
    "SnowflakeAtlasCrawler/1.0 "
    "(+https://github.com/yourorg/snowflake-atlas; "
    "educational research project; polite crawler)"
)

ROOT_LLMSTXT = "https://docs.snowflake.com/llms.txt"
BASE_URL = "https://docs.snowflake.com/en/"

# Default jitter range (seconds)
DEFAULT_DELAY_MIN = 0.3
DEFAULT_DELAY_MAX = 1.5
DEFAULT_DELAY = 0.5  # fallback when no jitter range given
MAX_CONCURRENT = 5
TIMEOUT = 30

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


# =========================================================================
# Helpers
# =========================================================================


def _pick_browser_profile() -> dict:
    """Return a random browser-profile dict."""
    return random.choice(_BROWSER_PROFILES)


def _build_request_headers(stealth: bool) -> dict:
    """Assemble a header dict that avoids default Python library fingerprints.

    When *stealth* is True the returned dict includes a randomly-chosen
    browser profile.  When False only a polite User-Agent is set.
    """
    if not stealth:
        return {"User-Agent": POLITE_USER_AGENT}

    profile = _pick_browser_profile()
    headers = dict(profile["headers"])
    headers["User-Agent"] = profile["user_agent"]

    # Common modern-browser headers applied uniformly across profiles
    headers["Accept-Encoding"] = "gzip, deflate, br"
    headers["DNT"] = "1"
    headers["Connection"] = "keep-alive"
    headers["Sec-Fetch-Dest"] = "document"
    headers["Sec-Fetch-Mode"] = "navigate"
    headers["Sec-Fetch-Site"] = "none"
    headers["Sec-Fetch-User"] = "?1"
    headers["Upgrade-Insecure-Requests"] = "1"

    return headers


# =========================================================================
# Fetch engines
# =========================================================================


class BaseEngine:
    """Interface for URL-fetch engines."""

    async def start(self):
        """Acquire resources (session, browser, etc.)."""

    async def stop(self):
        """Release resources."""

    async def fetch(self, url: str) -> str | None:
        """Return response body text, or None on failure."""
        ...


class AiohttpEngine(BaseEngine):
    """Async HTTP fetch engine.

    In *stealth* mode the User-Agent and client-hint headers are rotated
    on every request, and the inter-request delay uses a random jitter.
    """

    def __init__(
        self,
        stealth: bool,
        delay_s: float,
        delay_min: float,
        delay_max: float,
        max_concurrent: int,
    ):
        self.stealth = stealth
        self.delay_s = delay_s
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: aiohttp.ClientSession | None = None
        self._last_request = 0.0

    async def start(self):
        timeout = aiohttp.ClientTimeout(total=TIMEOUT)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            # Default polite headers; rotated per-request in stealth.
            headers=_build_request_headers(stealth=False),
        )

    async def stop(self):
        if self.session:
            await self.session.close()

    async def _rate_limit(self):
        """Sleep for a random interval drawn from the configured range."""
        # If a fixed delay was requested, delay_min == delay_max.
        delay = random.uniform(self.delay_min, self.delay_max)
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request = time.monotonic()

    async def fetch(self, url: str) -> str | None:
        await self._rate_limit()
        async with self.semaphore:
            try:
                # Rotate browser identity on every request in stealth mode.
                req_headers = (
                    _build_request_headers(stealth=True)
                    if self.stealth
                    else None
                )
                async with self.session.get(url, headers=req_headers) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 404:
                        print(f"  404: {url}")
                    else:
                        print(f"  HTTP {resp.status}: {url}")
            except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                print(f"  Error fetching {url}: {exc}")
        return None


class CamoufoxEngine(BaseEngine):
    """Fetch engine backed by a real Firefox browser via Camoufox.

    Camoufox applies anti-detection patches (WebGL, fonts, timezone,
    navigator.*) that mask headless-browser attributes.  This is the
    slowest but most stealthy option (~3-5 s per page).
    """

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._camoufox = None
        self._browser = None

    async def start(self):
        try:
            from camoufox import Camoufox
        except ImportError:
            print(
                "Camoufox is not installed.\n"
                "  pip install camoufox\n"
                "  # or use --engine aiohttp (the default).",
                file=sys.stderr,
            )
            raise

        self._camoufox = Camoufox(
            headless=True,
            humanize=True,       # random mouse-move & wait patterns
            disable_webrtc=True,  # prevent IP leaks
        )
        await self._camoufox.__aenter__()

    async def stop(self):
        if self._camoufox:
            await self._camoufox.__aexit__(None, None, None)

    async def fetch(self, url: str) -> str | None:
        async with self.semaphore:
            try:
                # Camoufox returns a stealth-patched Playwright page.
                page = await self._camoufox.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT * 1000)
                # Short human-like pause before grabbing content
                await asyncio.sleep(random.uniform(0.3, 1.0))
                content = await page.content()
                await page.close()
            except Exception as exc:
                print(f"  Browser error fetching {url}: {exc}")
                return None

        # Extract readable text from the raw HTML.
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        source = main if main else soup
        return source.get_text(separator="\n", strip=True)


# =========================================================================
# Crawler
# =========================================================================


class SnowflakeCrawler:
    """Orchestrate the full crawl: fetch llms.txt trees, save .md files."""

    def __init__(
        self,
        output_dir: Path,
        sections: list[str] | None = None,
        max_pages: int | None = None,
        engine: str = "aiohttp",
        stealth: bool = False,
        delay: float = DEFAULT_DELAY,
        delay_min: float = DEFAULT_DELAY_MIN,
        delay_max: float = DEFAULT_DELAY_MAX,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.sections = sections or list(SECTIONS.keys())
        self.max_pages = max_pages
        self.stealth = stealth
        self.max_concurrent = max_concurrent

        # Build the engine.
        if engine == "camoufox":
            self.engine_impl: BaseEngine = CamoufoxEngine(
                max_concurrent=max_concurrent,
            )
            self._engine_label = "camoufox"
        else:
            self.engine_impl = AiohttpEngine(
                stealth=stealth,
                delay_s=delay,
                delay_min=delay_min,
                delay_max=delay_max,
                max_concurrent=max_concurrent,
            )
            self._engine_label = "aiohttp+stealth" if stealth else "aiohttp"

        self.stats = {"fetched": 0, "skipped": 0, "errors": 0, "bytes": 0}
        self.crawl_meta = {
            "source_url": "https://docs.snowflake.com",
            "crawled_at": "",
            "crawler_sha": "unknown",
            "engine": engine,
            "stealth": stealth,
        }

    # ---- lifecycle -------------------------------------------------------

    async def __aenter__(self):
        await self.engine_impl.start()
        return self

    async def __aexit__(self, *args):
        await self.engine_impl.stop()

    # ---- internal helpers ------------------------------------------------

    async def _fetch(self, url: str) -> str | None:
        return await self.engine_impl.fetch(url)

    @staticmethod
    def _parse_llms_txt(content: str, base_url: str) -> list[str]:
        """Extract ``.md`` page URLs from an ``llms.txt`` index."""
        urls: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            m = re.match(r"- \[.*?\]\((.*?)\)", line)
            if m:
                absolute = urljoin(base_url, m.group(1))
                if absolute.endswith(".md"):
                    urls.append(absolute)
        return urls

    @staticmethod
    def _extract_frontmatter_and_body(content: str) -> tuple[dict, str]:
        frontmatter: dict = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1])
                    frontmatter = fm if isinstance(fm, dict) else {}
                except Exception:
                    frontmatter = {}
                body = parts[2].strip()
        return frontmatter, body

    def _local_path(self, url: str) -> Path:
        """Map a docs URL to the local ``markdown/<publication>/<path>`` file."""
        parsed = urlparse(url)
        path = parsed.path
        if path.startswith("/en/"):
            path = path[4:]
        path = path.lstrip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            publication = parts[0]
            file_path = "/".join(parts[1:])
        else:
            publication = "general"
            file_path = path
        return self.output_dir / "markdown" / publication / file_path

    # ---- page fetching ---------------------------------------------------

    async def _fetch_and_save_page(self, url: str) -> bool:
        local_path = self._local_path(url)
        if local_path.exists():
            self.stats["skipped"] += 1
            return True

        content = await self._fetch(url)
        if content is None:
            self.stats["errors"] += 1
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(local_path, "w", encoding="utf-8") as f:
            await f.write(content)

        self.stats["fetched"] += 1
        self.stats["bytes"] += len(content.encode("utf-8"))

        if self.stats["fetched"] % 50 == 0:
            print(f"  [{self._engine_label}] {self.stats['fetched']} pages fetched...")

        return True

    # ---- section crawling ------------------------------------------------

    async def crawl_section(self, section_key: str) -> int:
        if section_key not in SECTIONS:
            print(f"  Unknown section: {section_key}")
            return 0

        section_path = SECTIONS[section_key]
        section_url = urljoin(BASE_URL, section_path)

        print(f"\n=== {section_key} ===")
        print(f"  Index: {section_url}")

        index_content = await self._fetch(section_url)
        if index_content is None:
            print("  Failed to fetch section index")
            return 0

        page_urls = self._parse_llms_txt(index_content, section_url)
        print(f"  Pages: {len(page_urls)}")

        if self.max_pages:
            page_urls = page_urls[: self.max_pages]
            print(f"  (limited to {len(page_urls)})")

        results = await asyncio.gather(
            *[self._fetch_and_save_page(u) for u in page_urls],
            return_exceptions=True,
        )

        success = sum(1 for r in results if r is True)
        errors = sum(1 for r in results if r is False or isinstance(r, Exception))
        print(f"  Done: {success} ok, {errors} errors")
        return success

    # ---- top-level -------------------------------------------------------

    async def crawl_all(self) -> dict:
        print(f"Output  : {self.output_dir}")
        print(f"Sections: {', '.join(self.sections)}")
        print(f"Engine  : {self._engine_label}")
        if self._engine_label == "aiohttp+stealth":
            print(f"Jitter  : {self.engine_impl.delay_min}-{self.engine_impl.delay_max}s")

        self.crawl_meta["crawled_at"] = datetime.datetime.now(datetime.UTC).isoformat()

        total = 0
        for section in self.sections:
            if self.max_pages and total >= self.max_pages:
                break
            total += await self.crawl_section(section)

        meta_path = self.output_dir / "crawl_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(self.crawl_meta, indent=2))
        print(f"\nMetadata → {meta_path}")

        print(
            f"\nComplete: {self.stats['fetched']} fetched, "
            f"{self.stats['skipped']} skipped, "
            f"{self.stats['errors']} errors, "
            f"{self.stats['bytes']/1e6:.1f} MB"
        )
        return self.stats


# =========================================================================
# CLI
# =========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawl Snowflake documentation to a local mirror.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --output ./data/snowflake-docs\n"
            "  %(prog)s --output ./data/snowflake-docs --stealth\n"
            "  %(prog)s --output ./data/snowflake-docs --engine camoufox\n"
            "  %(prog)s --output ./data/snowflake-docs --sections cortex-ai,sql-functions\n"
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="Output directory")
    p.add_argument("--sections", help="Comma-separated section keys to crawl")
    p.add_argument("--max-pages", type=int, help="Max pages overall")
    p.add_argument(
        "--engine",
        choices=["aiohttp", "camoufox"],
        default="aiohttp",
        help="Fetch engine (default: aiohttp).  'camoufox' is stealthier but slower.",
    )
    p.add_argument(
        "--stealth",
        action="store_true",
        help="Rotate User-Agent, browser headers, and use jittered delays. "
        "Ignored when --engine=camoufox (already stealth).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Fixed delay between requests (seconds).  Sets both min and max "
        "to the same value, disabling jitter.  Overrides --delay-min/--delay-max.",
    )
    p.add_argument(
        "--delay-min",
        type=float,
        default=DEFAULT_DELAY_MIN,
        help=f"Minimum delay (default {DEFAULT_DELAY_MIN}s).  Ignored when --delay is set.",
    )
    p.add_argument(
        "--delay-max",
        type=float,
        default=DEFAULT_DELAY_MAX,
        help=f"Maximum delay (default {DEFAULT_DELAY_MAX}s).  Ignored when --delay is set.",
    )
    p.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help=f"Max concurrent requests (default {MAX_CONCURRENT})")
    return p


def main():
    args = build_parser().parse_args()

    sections = args.sections.split(",") if args.sections else None

    # Resolve delay: --delay sets a fixed (no-jitter) value, else use range.
    if args.delay is not None:
        delay_min = delay_max = args.delay
    else:
        delay_min = args.delay_min
        delay_max = args.delay_max

    async def run():
        async with SnowflakeCrawler(
            output_dir=args.output,
            sections=sections,
            max_pages=args.max_pages,
            engine=args.engine,
            stealth=args.stealth,
            delay=args.delay or DEFAULT_DELAY,
            delay_min=delay_min,
            delay_max=delay_max,
            max_concurrent=args.max_concurrent,
        ) as crawler:
            await crawler.crawl_all()

    asyncio.run(run())


if __name__ == "__main__":
    main()
