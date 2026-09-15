#!/usr/bin/env python3
"""Fix Windows backslash paths in USD file references for Linux compatibility.

Scans all .usd/.usda/.usdc files under a given root directory, finds references
containing Windows-style backslashes, and replaces them with forward slashes.
"""

import os
import sys

from pxr import Sdf


def fix_layer(layer_path: str, dry_run: bool = False) -> int:
    """Fix backslash references in a single USD layer. Returns count of fixes."""
    try:
        layer = Sdf.Layer.FindOrOpen(layer_path)
    except Exception as e:
        print(f"  [SKIP] Cannot open: {layer_path} ({e})")
        return 0
    if layer is None:
        print(f"  [SKIP] Cannot open: {layer_path}")
        return 0

    fixes = 0

    def visit(spec):
        nonlocal fixes
        if spec is None:
            return
        # Check references
        ref_list = spec.referenceList
        for accessor_name in ("prependedItems", "appendedItems", "explicitItems"):
            accessor = getattr(ref_list, accessor_name, None)
            if accessor is None:
                continue
            items = list(accessor)
            for i, ref in enumerate(items):
                if '\\' in ref.assetPath:
                    new_path = ref.assetPath.replace('\\', '/')
                    if dry_run:
                        print(f"    Would fix [{accessor_name}]: {ref.assetPath} -> {new_path}")
                    else:
                        print(f"    Fixed [{accessor_name}]: {ref.assetPath} -> {new_path}")
                        accessor[i] = Sdf.Reference(
                            assetPath=new_path,
                            primPath=ref.primPath,
                            layerOffset=ref.layerOffset,
                            customData=ref.customData,
                        )
                    fixes += 1

        # Recurse
        for child in spec.nameChildren:
            visit(child)

    # Visit all root prims
    for prim_spec in layer.rootPrims:
        visit(prim_spec)

    if fixes > 0 and not dry_run:
        layer.Save()
        print(f"    Saved: {layer_path}")

    return fixes


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_usd_paths.py <root_dir> [--dry-run]")
        sys.exit(1)

    root = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[DRY RUN] No files will be modified.\n")

    total_files = 0
    total_fixes = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(('.usd', '.usda', '.usdc')):
                continue
            fpath = os.path.join(dirpath, fname)
            # Quick binary scan for backslash
            try:
                with open(fpath, 'rb') as f:
                    content = f.read()
                if b'\\' not in content:
                    continue
            except Exception:
                continue

            total_files += 1
            print(f"[{total_files}] {fpath}")
            n = fix_layer(fpath, dry_run=dry_run)
            total_fixes += n

    print(f"\nDone: {total_files} files scanned, {total_fixes} references fixed.")


if __name__ == "__main__":
    main()
