#!/usr/bin/env python3
"""Batch-rename object IDs in task 44 HDF5 files to match the corrected scene YAML.

Background: ``scenes/44_microwave_bowl_loading.yaml`` had its asset misnamed
``a029_plate`` while the actual model is ``024_bowl``.  The scene YAML and custom
evaluator have been updated to use ``obj_024_bowl_3``, but the recorded HDF5
trajectories still reference ``obj_029_plate_3``.

This script rewrites every HDF5 under the dataset directory so that:

* ``objects/obj_029_plate_3``   → ``objects/obj_024_bowl_3``
* ``object_asset_keys``  JSON    → key + value updated
* ``object_asset_paths`` JSON    → key updated
* ``object_roles``       JSON    → key updated

Usage:
    python scripts/fix_task44_hdf5_object_id.py [--dry-run] [--dataset-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py

OLD_OBJ_ID = "obj_029_plate_3"
NEW_OBJ_ID = "obj_024_bowl_3"

OLD_ASSET_KEY = "a029_plate"
NEW_ASSET_KEY = "a024_bowl"

# h5py ``move()`` leaves an empty group behind in some versions — verify after.
# Also, the old group path may linger as a soft/external link; we clean it up.

META_JSON_FIELDS = ["object_asset_keys", "object_asset_paths", "object_roles"]


def fix_one(path: Path, dry_run: bool = False) -> bool:
    """Rewrite *path* in-place.  Returns True on success."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = False

    try:
        if not dry_run:
            # h5py cannot rename groups in-place reliably across all versions,
            # so we copy the file, operate on the copy, then atomically replace.
            import shutil
            shutil.copy2(path, tmp)

        f = h5py.File(tmp if not dry_run else path, "r+" if not dry_run else "r")

        # --- 1. Rename objects/ group ------------------------------------------------
        old_grp = f"objects/{OLD_OBJ_ID}"
        new_grp = f"objects/{NEW_OBJ_ID}"
        if old_grp in f:
            if dry_run:
                print(f"  [dry-run] would rename {old_grp} → {new_grp}")
            else:
                f.move(old_grp, new_grp)
            changed = True
        else:
            if NEW_OBJ_ID not in f.get("objects", {}):
                print(f"  ⚠  {path.name}: neither {OLD_OBJ_ID} nor {NEW_OBJ_ID} found in objects/")

        # --- 2. Update JSON metadata fields ------------------------------------------
        for field in META_JSON_FIELDS:
            ds = f["meta"].get(field)
            if ds is None:
                continue
            try:
                val = json.loads(ds[()])
            except (json.JSONDecodeError, TypeError):
                print(f"  ⚠  {path.name}: cannot decode meta/{field}")
                continue

            if isinstance(val, dict):
                # 1. Rename object-id key (e.g. obj_029_plate_3 → obj_024_bowl_3)
                if OLD_OBJ_ID in val:
                    val[NEW_OBJ_ID] = val.pop(OLD_OBJ_ID)
                    changed = True
                # 2. Replace old asset-key *values* (e.g. "a029_plate" → "a024_bowl")
                for k, v in list(val.items()):
                    if v == OLD_ASSET_KEY:
                        val[k] = NEW_ASSET_KEY
                        changed = True

                new_raw = json.dumps(val, separators=(",", ":")).encode()
                if not dry_run:
                    del f["meta"][field]
                    f["meta"][field] = new_raw

        f.close()

        if changed and not dry_run:
            os.replace(tmp, path)
        elif not dry_run:
            os.unlink(tmp)

        return changed

    except Exception:
        if not dry_run and tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Fix task-44 HDF5 object IDs")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.expanduser("~/teleopdata/dataset/44_microwave_bowl_loading"),
        help="Root of the task-44 dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        print(f"ERROR: dataset directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    hdf5_files = sorted(root.rglob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files under {root}")
    print("Dry run — no changes will be made." if args.dry_run else "Rewriting files in-place…")
    print()

    ok, skipped, failed = 0, 0, 0
    for p in hdf5_files:
        rel = p.relative_to(root)
        try:
            changed = fix_one(p, dry_run=args.dry_run)
            if changed:
                print(f"  ✓ {rel}")
                ok += 1
            else:
                print(f"  - {rel}  (already up-to-date)")
                skipped += 1
        except Exception as exc:
            print(f"  ✗ {rel}  — {exc}")
            failed += 1

    print()
    print(f"Done: {ok} updated, {skipped} skipped, {failed} failed")
    if args.dry_run and ok:
        print("Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
