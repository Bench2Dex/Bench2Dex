#!/usr/bin/env python3
"""Fix absolute texture paths in usd_sdf_linux assets to relative paths.

Some usd_sdf_linux USD files contain hardcoded absolute texture paths.
This script rewrites them to relative paths like:
    @./textures/foo.png@
so the assets work on any OS.

Usage:
    python tools/fix_usd_sdf_linux_paths.py /path/to/dex2bench_dataset
    python tools/fix_usd_sdf_linux_paths.py /path/to/dex2bench_dataset --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def fix_layer(usd_path: str, dry_run: bool = False) -> int:
    """Fix absolute paths in a single USD file. Returns number of fixes."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        return 0

    text = layer.ExportToString()
    usd_dir = os.path.dirname(os.path.abspath(usd_path))

    def _make_relative(match: re.Match) -> str:
        abs_path = match.group(1)
        # The original path points into <dataset-root>/Objects/.../textures/foo.png.
        # The USD file is at   .../Objects/163_baguette/usd_sdf_linux/base3.usd
        # We need to find the common structure suffix and compute relative path.
        #
        # Strategy: find "usd_sdf_linux/" in both the abs path and the USD file path,
        # then compute relative path between the two suffixes.
        norm = abs_path.replace("\\", "/")
        marker = "usd_sdf_linux/"
        idx = norm.find(marker)
        if idx < 0:
            return match.group(0)  # no marker, skip

        # Suffix after the usd_sdf_linux/ directory (e.g. "textures/foo.png")
        ref_suffix = norm[idx + len(marker):]

        # USD file's position relative to its usd_sdf_linux/ root
        usd_abs = os.path.abspath(usd_path).replace("\\", "/")
        usd_idx = usd_abs.find(marker)
        if usd_idx < 0:
            return match.group(0)
        usd_suffix = usd_abs[usd_idx + len(marker):]  # e.g. "base3.usd" or "configuration/mobility_base.usd"

        # Compute relative path from USD file dir to referenced file
        usd_subdir = os.path.dirname(usd_suffix)  # e.g. "" or "configuration"
        if usd_subdir:
            depth = usd_subdir.count("/") + 1
            rel = "../" * depth + ref_suffix
        else:
            rel = "./" + ref_suffix

        return f"@{rel}@"

    # Match both absolute paths (/home/...) and already-broken relative paths
    # that still contain the original absolute directory structure.
    new_text, count = re.subn(r"@([^@]*?/home/[^@]+)@", _make_relative, text)

    if count > 0 and not dry_run:
        # Write back as usda (text format) — Sdf can re-open it fine
        new_layer = Sdf.Layer.CreateAnonymous()
        new_layer.ImportFromString(new_text)
        new_layer.Export(usd_path)

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", help="Path to dex2bench_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify files")
    args = parser.parse_args()

    objects_dir = os.path.join(args.dataset_root, "Objects")
    if not os.path.isdir(objects_dir):
        print(f"Error: {objects_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect all USD files under usd_sdf_linux directories
    patterns = [
        os.path.join(objects_dir, "*", "usd_sdf_linux", "*.usd"),
        os.path.join(objects_dir, "*", "usd_sdf_linux", "**", "*.usd"),
        os.path.join(objects_dir, "*", "*", "*", "*", "usd_sdf_linux", "*.usd"),
        os.path.join(objects_dir, "*", "*", "*", "*", "usd_sdf_linux", "**", "*.usd"),
    ]
    usd_files = set()
    for pat in patterns:
        usd_files.update(glob.glob(pat, recursive=True))

    usd_files = sorted(usd_files)
    print(f"Found {len(usd_files)} USD files under usd_sdf_linux/")

    total_fixes = 0
    fixed_files = 0
    for path in usd_files:
        n = fix_layer(path, dry_run=args.dry_run)
        if n > 0:
            rel = os.path.relpath(path, args.dataset_root)
            print(f"  {'[dry-run] ' if args.dry_run else ''}Fixed {n} paths in {rel}")
            total_fixes += n
            fixed_files += 1

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {total_fixes} absolute paths in {fixed_files}/{len(usd_files)} files.")


if __name__ == "__main__":
    main()
