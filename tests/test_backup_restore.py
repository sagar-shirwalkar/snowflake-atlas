"""Tests for bundle snapshot and restore (atlas/backup.py, atlas/restore.py)."""

import json
import tarfile
from pathlib import Path

import pytest

from atlas.backup import create_snapshot
from atlas.restore import restore_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Create a minimal bundle with manifest."""
    bdir = tmp_path / "rag-bundle"
    bdir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "chunk_count": 42,
        "built_at": "2025-01-01T00:00:00Z",
        "embedding_model": "test-model",
    }
    (bdir / "manifest.json").write_text(json.dumps(manifest))
    (bdir / "chunks.parquet").write_bytes(b"fake parquet data")
    return bdir


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------

class TestCreateSnapshot:
    def test_snapshot_creates_tar(self, bundle_dir: Path):
        """Use a backup root outside the bundle dir to avoid tar self-include."""
        snapshot_path = create_snapshot(bundle_dir, bundle_dir.parent / ".backups")
        assert snapshot_path.exists()
        assert snapshot_path.suffix == ".gz"
        assert snapshot_path.name.startswith("snapshot-")
        assert tarfile.is_tarfile(snapshot_path)

    def test_snapshot_not_bundle_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Not a valid bundle"):
            create_snapshot(empty_dir, empty_dir / ".backups")

    def test_snapshot_contains_manifest(self, bundle_dir: Path):
        snapshot_path = create_snapshot(bundle_dir, bundle_dir.parent / ".backups")
        with tarfile.open(snapshot_path, "r:gz") as tf:
            names = tf.getnames()
            # The tar has bundle_dir.name as the top-level directory
            assert any("manifest.json" in n for n in names)


# ---------------------------------------------------------------------------
# restore_snapshot
# ---------------------------------------------------------------------------

class TestRestoreSnapshot:
    def test_restore_from_snapshot(self, bundle_dir: Path, tmp_path: Path):
        """Restore must match bundle_dir name because the tar has a top-level dir."""
        snapshot_path = create_snapshot(bundle_dir, bundle_dir.parent / ".backups")

        # The restore target must have the same *name* as the original bundle dir,
        # because the snapshot tar is created with —-C <parent> <dirname> which
        # preserves <dirname>/ as the top-level entry.
        restore_dir = tmp_path / bundle_dir.name
        restore_snapshot(snapshot_path, restore_dir)

        assert restore_dir.is_dir()
        manifest_path = restore_dir / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["chunk_count"] == 42

    def test_restore_over_existing(self, bundle_dir: Path, tmp_path: Path):
        snapshot_path = create_snapshot(bundle_dir, bundle_dir.parent / ".backups")

        # Use a unique restore dir name that doesn't clash with bundle_dir fixture
        restore_dir = tmp_path / "restored" / bundle_dir.name
        restore_dir.mkdir(parents=True)
        (restore_dir / "stale.txt").write_text("stale")

        restore_snapshot(snapshot_path, restore_dir)
        assert not (restore_dir / "stale.txt").exists()
        assert (restore_dir / "manifest.json").exists()

    def test_restore_missing_snapshot_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Snapshot not found"):
            restore_snapshot(tmp_path / "nonexistent.tar.gz", tmp_path / "out")

    def test_restore_verifies_manifest(self, bundle_dir: Path, tmp_path: Path):
        create_snapshot(bundle_dir, bundle_dir.parent / ".backups")  # noqa: F841

        # Create an invalid snapshot without manifest
        corrupt_path = tmp_path / "corrupt.tar.gz"
        with tarfile.open(corrupt_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="some-file.txt")
            info.size = 4
            info.type = tarfile.REGTYPE
            tf.addfile(info, fileobj=__import__("io").BytesIO(b"test"))

        restore_dir = tmp_path / bundle_dir.name
        with pytest.raises(RuntimeError, match="missing manifest"):
            restore_snapshot(corrupt_path, restore_dir)


# ---------------------------------------------------------------------------
# Round-trip: snapshot then restore
# ---------------------------------------------------------------------------

class TestSnapshotRestoreRoundTrip:
    def test_round_trip(self, bundle_dir: Path, tmp_path: Path):
        snapshot_path = create_snapshot(bundle_dir, bundle_dir.parent / ".backups")

        original_content = (bundle_dir / "chunks.parquet").read_bytes()

        restore_dir = tmp_path / bundle_dir.name
        restore_snapshot(snapshot_path, restore_dir)

        restored_content = (restore_dir / "chunks.parquet").read_bytes()
        assert restored_content == original_content

        restored_manifest = json.loads((restore_dir / "manifest.json").read_text())
        assert restored_manifest["chunk_count"] == 42
