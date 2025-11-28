#!/usr/bin/env python3
import os
import re
from pathlib import Path
import argparse

PATTERN = re.compile(r"^synthetic_(?:[0-9]|10)_(\d+)\.png$", re.IGNORECASE)

def rename_in_dir(d: Path, dry_run: bool):
    changed, skipped = 0, 0
    for name in sorted(os.listdir(d)):
        m = PATTERN.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        if not (0 <= idx <= 149):
            skipped += 1
            continue

        src = d / name
        dst = d / f"slice_{idx:04d}.png"

        # If the target already exists and isn't the same file, skip to be safe.
        if dst.exists() and src.resolve() != dst.resolve():
            print(f"[SKIP] {src} -> {dst} (target exists)")
            skipped += 1
            continue

        print(f"[RENAME]{' (dry-run)' if dry_run else ''} {src} -> {dst}")
        if not dry_run:
            src.rename(dst)
        changed += 1
    return changed, skipped

def main():
    ap = argparse.ArgumentParser(
        description="Recursively rename synthetic_2_<n>.png to slice_<nnnn>.png (0..149) per folder."
    )
    ap.add_argument("root", type=Path, help="Root directory to traverse")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without renaming")
    args = ap.parse_args()

    total_changed = total_skipped = 0
    for dirpath, dirnames, filenames in os.walk(args.root):
        d = Path(dirpath)
        changed, skipped = rename_in_dir(d, args.dry_run)
        total_changed += changed
        total_skipped += skipped

    print(f"\nDone. Renamed: {total_changed}, Skipped: {total_skipped}")

if __name__ == "__main__":
    main()
