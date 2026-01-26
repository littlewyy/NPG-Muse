#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

# python copy_by_prefix.py /path/src prefix_ /path/dst
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy files with a prefix from source dir to target dir."
    )
    parser.add_argument("source_dir", help="Source directory to scan")
    parser.add_argument("prefix", help="Filename prefix to match")
    parser.add_argument("target_dir", help="Target directory to copy into")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )
    return parser.parse_args()


def iter_files(source_dir: str, recursive: bool):
    if recursive:
        for root, _, files in os.walk(source_dir):
            for name in files:
                yield root, name
    else:
        for name in os.listdir(source_dir):
            full_path = os.path.join(source_dir, name)
            if os.path.isfile(full_path):
                yield source_dir, name


def main() -> int:
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)

    if not os.path.isdir(source_dir):
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 2

    if not args.dry_run:
        os.makedirs(target_dir, exist_ok=True)

    copied = 0
    for root, name in iter_files(source_dir, args.recursive):
        if not name.startswith(args.prefix):
            continue
        src = os.path.join(root, name)
        dst = os.path.join(target_dir, name)
        if args.dry_run:
            print(f"DRY-RUN: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
        copied += 1

    print(f"Total matched files: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
