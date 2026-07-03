"""Tests for the bundle downloader (atlas/download.py)."""

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas.download import (
    _extract,
    _sha256,
    find_bundle_asset,
    resolve_release,
)


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_compute_hash(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _sha256(f) == expected

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256(f) == expected

    def test_large_file(self, tmp_path: Path):
        f = tmp_path / "large.bin"
        data = b"a" * (2 * 1024 * 1024 + 13)
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(f) == expected


# ---------------------------------------------------------------------------
# find_bundle_asset
# ---------------------------------------------------------------------------

class TestFindBundleAsset:
    def test_finds_tar_zst(self):
        release = {
            "assets": [
                {"name": "bundle.tar.zst"},
                {"name": "README.md"},
            ]
        }
        asset = find_bundle_asset(release)
        assert asset["name"] == "bundle.tar.zst"

    def test_finds_tar_gz(self):
        release = {
            "assets": [
                {"name": "bundle.tar.gz"},
            ]
        }
        asset = find_bundle_asset(release)
        assert asset["name"] == "bundle.tar.gz"

    def test_finds_zip(self):
        release = {
            "assets": [
                {"name": "bundle.zip"},
            ]
        }
        asset = find_bundle_asset(release)
        assert asset["name"] == "bundle.zip"

    def test_no_matching_asset_raises(self):
        release = {"assets": [{"name": "README.md"}, {"name": "notes.txt"}]}
        with pytest.raises(RuntimeError, match="No bundle asset"):
            find_bundle_asset(release)

    def test_empty_assets_raises(self):
        release = {"assets": []}
        with pytest.raises(RuntimeError, match="No bundle asset"):
            find_bundle_asset(release)


# ---------------------------------------------------------------------------
# resolve_release
# ---------------------------------------------------------------------------

class TestResolveRelease:
    @patch("atlas.download._http_json")
    def test_with_tag(self, mock_http):
        mock_http.return_value = {"tag_name": "v1.0", "name": "v1.0"}
        result = resolve_release("org/repo", "v1.0")
        mock_http.assert_called_once_with(
            "https://api.github.com/repos/org/repo/releases/tags/v1.0"
        )
        assert result["tag_name"] == "v1.0"

    @patch("atlas.download._http_json")
    def test_latest(self, mock_http):
        mock_http.return_value = {"tag_name": "v2.0", "name": "Latest"}
        result = resolve_release("org/repo", None)
        mock_http.assert_called_once_with(
            "https://api.github.com/repos/org/repo/releases/latest"
        )
        assert result["tag_name"] == "v2.0"


# ---------------------------------------------------------------------------
# _extract
# ---------------------------------------------------------------------------

class TestExtract:
    def _make_tar_with_prefix(self, archive: Path, inner_prefix: str, files: dict[str, bytes]):
        """Create a tar.gz with an inner prefix dir, like real backup tars do."""
        with tarfile.open(archive, "w:gz") as tf:
            for name, content in files.items():
                info = tarfile.TarInfo(f"{inner_prefix}/{name}")
                info.type = tarfile.REGTYPE
                info.size = len(content)
                tf.addfile(info, fileobj=__import__("io").BytesIO(content))

    def test_tar_gz_with_prefix(self, tmp_path: Path):
        """Real bundle tars have a top-level dir.  _extract strips it via
        --strip-components=1 for tar.zst but for tar.gz it uses
        extractall, which preserves the tar's internal structure."""
        archive = tmp_path / "test.tar.gz"
        self._make_tar_with_prefix(
            archive,
            "bundle",
            {"manifest.json": json.dumps({"key": "value"}).encode(), "readme.txt": b"test"},
        )

        target = tmp_path / "output"
        _extract(archive, target)

        # tar.gz uses extractall() which preserves the top-level dir
        assert (target / "bundle" / "manifest.json").is_file()
        # Files are also flat at the target if there's no prefix stripping
        content = json.loads((target / "bundle" / "manifest.json").read_text())
        assert content["key"] == "value"

    def test_zip(self, tmp_path: Path):
        archive = tmp_path / "test.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"key": "value"}))
            zf.writestr("readme.txt", "test")

        target = tmp_path / "output"
        _extract(archive, target)
        assert (target / "manifest.json").is_file()
        assert json.loads((target / "manifest.json").read_text())["key"] == "value"

    def test_unsupported_format(self, tmp_path: Path):
        archive = tmp_path / "test.rar"
        archive.write_bytes(b"fake rar")
        with pytest.raises(RuntimeError, match="Unsupported archive format"):
            _extract(archive, tmp_path / "out")
