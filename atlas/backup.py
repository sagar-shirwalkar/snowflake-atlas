"""Snapshot the current RAG bundle for rollback."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def create_snapshot(bundle_dir: Path, backup_root: Path) -> Path:
    """Create a timestamped snapshot of the bundle."""
    bundle_dir = Path(bundle_dir).resolve()
    if not (bundle_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Not a valid bundle: {bundle_dir}")

    backup_root.mkdir(parents=True, exist_ok=True)
    ts = subprocess.run(
        ["date", "+%Y%m%dT%H%M%SZ"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    dest = backup_root / f"snapshot-{ts}.tar.gz"
    print(f"  Creating snapshot: {dest}")
    subprocess.run(
        ["tar", "-czf", str(dest), "-C", str(bundle_dir.parent), bundle_dir.name],
        check=True,
        timeout=120,
    )
    return dest


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the backup command."""
    p = argparse.ArgumentParser(description="Snapshot the current RAG bundle")
    p.add_argument("--bundle", type=Path, required=True, help="Path to the bundle directory")
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Where to store the snapshot (default: <bundle>/.backups)",
    )
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    backup_root = args.backup_dir or (args.bundle / ".backups")
    create_snapshot(args.bundle, backup_root)
    return 0


def main() -> None:
    """Entry point: create a bundle snapshot and exit."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
