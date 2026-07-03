"""Roll back to a previous bundle snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def restore_snapshot(snapshot_path: Path, target_dir: Path) -> None:
    """Extract a snapshot and verify its manifest."""
    snapshot_path = Path(snapshot_path).resolve()
    target_dir = Path(target_dir).resolve()

    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    if target_dir.exists():
        print(f"  Removing existing bundle at {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {snapshot_path} to {target_dir}")
    subprocess.run(
        ["tar", "-xzf", str(snapshot_path), "-C", str(target_dir.parent)],
        check=True,
        timeout=120,
    )

    # Verify manifest
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Restored bundle is missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    print(f"  Restored bundle: {manifest.get('chunk_count')} chunks, built {manifest.get('built_at')}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the restore command."""
    p = argparse.ArgumentParser(description="Roll back to a previous bundle snapshot")
    p.add_argument("--snapshot", type=Path, required=True, help="Path to snapshot .tar.gz")
    p.add_argument("--target", type=Path, required=True, help="Target directory to restore to")
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    restore_snapshot(args.snapshot, args.target)
    return 0


def main() -> None:
    """Entry point: restore a snapshot and exit."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
