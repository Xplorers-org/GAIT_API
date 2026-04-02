#!/usr/bin/env python3
"""Cleanup utility for GAIT_API generated files.

This script removes files under the runs directory that are older than a
configurable age (default: 30 minutes). It is designed to be triggered by cron.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup old files from runs directory")
    parser.add_argument(
        "--path",
        default="/app/runs",
        help="Runs directory path (default: /app/runs)",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=30,
        help="Delete files older than this many minutes (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted",
    )
    return parser.parse_args()


def cleanup_runs(root: Path, max_age_minutes: float, dry_run: bool = False) -> tuple[int, int]:
    if not root.exists():
        return 0, 0

    now = time.time()
    cutoff = now - (max_age_minutes * 60)

    deleted_files = 0
    deleted_dirs = 0

    # Delete eligible files first
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            mtime = file_path.stat().st_mtime
            if mtime <= cutoff:
                if dry_run:
                    print(f"[DRY-RUN] file: {file_path}")
                else:
                    file_path.unlink(missing_ok=True)
                deleted_files += 1
        except FileNotFoundError:
            # Might have been removed by another process
            continue
        except PermissionError:
            print(f"[WARN] Permission denied: {file_path}")

    # Remove empty subdirectories bottom-up (excluding root and outputs/)
    preserve_dirs = {root, root / "outputs"}
    for dir_path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        # Skip preserving root structure
        if dir_path in preserve_dirs or dir_path.parent in preserve_dirs:
            continue
        try:
            if any(dir_path.iterdir()):
                continue
            if dry_run:
                print(f"[DRY-RUN] dir:  {dir_path}")
            else:
                dir_path.rmdir()
            deleted_dirs += 1
        except (FileNotFoundError, PermissionError, OSError):
            # OSError when directory isn't empty anymore
            continue

    return deleted_files, deleted_dirs


def main() -> int:
    args = parse_args()
    root = Path(args.path)

    deleted_files, deleted_dirs = cleanup_runs(
        root=root,
        max_age_minutes=args.max_age_minutes,
        dry_run=args.dry_run,
    )

    print(
        f"Cleanup complete for {root} | "
        f"deleted_files={deleted_files} | deleted_dirs={deleted_dirs} | "
        f"max_age_minutes={args.max_age_minutes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
