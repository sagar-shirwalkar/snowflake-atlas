#!/usr/bin/env bash
set -euo pipefail

# Snowflake Atlas Bundle Publisher
# 
# Builds the bundle locally and publishes to GitHub Releases.
# Run this on a maintainer's machine (not in CI).

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v2026.07"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${REPO_ROOT}/data/snowflake-rag-bundle"
ARCHIVE_NAME="snowflake-atlas-bundle-${VERSION#v}.tar.zst"
ARCHIVE_PATH="${REPO_ROOT}/${ARCHIVE_NAME}"
GITHUB_REPO="${GITHUB_REPO:-your-org/snowflake-atlas-bundles}"

echo "=== Snowflake Atlas Bundle Publisher ==="
echo "Version: ${VERSION}"
echo "Bundle dir: ${BUNDLE_DIR}"
echo "Archive: ${ARCHIVE_NAME}"
echo "GitHub repo: ${GITHUB_REPO}"
echo

# 1. Verify bundle exists
if [[ ! -f "${BUNDLE_DIR}/manifest.json" ]]; then
    echo "ERROR: Bundle not found at ${BUNDLE_DIR}"
    echo "Run atlas-build first:"
    echo "  atlas-build --source-type web-crawl --mirror-path ./data/snowflake-docs --output ./data/snowflake-rag-bundle --prefer auto"
    exit 1
fi

echo "[1/4] Verifying bundle..."
CHUNK_COUNT=$(jq -r '.chunk_count' "${BUNDLE_DIR}/manifest.json")
SOURCE_SHA=$(jq -r '.source_sha' "${BUNDLE_DIR}/manifest.json")
echo "  Chunks: ${CHUNK_COUNT}"
echo "  Source SHA: ${SOURCE_SHA}"

# 2. Create archive
echo "[2/4] Creating archive (zstd)..."
tar --zstd -cf "${ARCHIVE_PATH}" -C "${BUNDLE_DIR}" .
ARCHIVE_SIZE=$(stat -c%s "${ARCHIVE_PATH}" 2>/dev/null || stat -f%z "${ARCHIVE_PATH}")
echo "  Size: $(numfmt --to=iec-i --suffix=B ${ARCHIVE_SIZE})"

# 3. Verify archive
echo "[3/4] Verifying archive..."
tar --zstd -tf "${ARCHIVE_PATH}" > /dev/null
echo "  Archive OK"

# 4. Publish to GitHub Releases
echo "[4/4] Publishing to GitHub Releases..."
if ! command -v gh &> /dev/null; then
    echo "ERROR: gh CLI not installed. Install with: brew install gh"
    exit 1
fi

gh release create "${VERSION}" "${ARCHIVE_PATH}" \
    --repo "${GITHUB_REPO}" \
    --title "Snowflake Atlas Bundle ${VERSION}" \
    --notes "Snowflake documentation RAG bundle built from docs.snowflake.com

**Contents:**
- ${CHUNK_COUNT} chunks
- Embedding model: Xenova/bge-base-en-v1.5 (768 dim)
- Source: docs.snowflake.com (crawled)
- Source SHA: ${SOURCE_SHA}

**Installation:**
\`\`\`bash
uv sync --extra mlx  # or --extra gpu / default
atlas-download --repo ${GITHUB_REPO} --tag ${VERSION} --output ./data/snowflake-rag-bundle
\`\`\`"

echo
echo "=== Done! ==="
echo "Release: https://github.com/${GITHUB_REPO}/releases/tag/${VERSION}"
echo
echo "Users can now install with:"
echo "  uv sync --extra mlx"
echo "  atlas-download --repo ${GITHUB_REPO} --tag ${VERSION} --output ./data/snowflake-rag-bundle"
