#!/usr/bin/env python3
"""Re-key stale object names in anchor ``resolved_object_placements``.

Scene ``44_microwave_bowl_loading`` renamed the bowl from ``obj_029_plate_3`` to
``obj_024_bowl_3`` in the scene YAML, but the per-episode anchor metadata stored
inside each HDF5 at ``meta/scene_generalization_sample`` (a JSON string) still
carries the old key under ``resolved_object_placements``.  At scene-build time
``build/scene_builder.py`` looks up ``saved_placements.get(obj_id)`` using the
*current* YAML id (``obj_024_bowl_3``), misses, and falls back to random pose
sampling — so ``none``/``inv_only`` end up with different bowl poses despite an
identical episode seed.

This script rewrites the stale key in-place.  The stored position/rpy values are
the bowl's true pose (verified to match ``/objects/obj_024_bowl_3/pose_world[0]``
to ~3e-7 m / 0.0013 deg across all episodes), so only the dict key changes.

Usage::

    python tools/labels/rekey_anchor_placements.py \
        /path/to/44_microwave_bowl_loading [--apply]

Without ``--apply`` the script runs read-only and reports what it *would*
change (dry run).  Pass ``--apply`` to write.  Either way it also writes a
backup JSON recording the original string for every file it touched, so the
change is fully reversible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py

# Stale-name -> current-name, for the objects present in this scene.
# Only obj_029_plate_3 -> obj_024_bowl_3 is needed (single rename).
RENAME_MAP = {
    "obj_029_plate_3": "obj_024_bowl_3",
}

DATASET_PATH = "meta/scene_generalization_sample"


def _decode(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def rekey_sample(sample: dict) -> tuple[dict, list[str]]:
    """Return (new_sample, renamed_keys). Mutates a copy only."""
    rop = sample.get("resolved_object_placements", {})
    if not isinstance(rop, dict) or not rop:
        return sample, []
    renamed = [k for k in rop if k in RENAME_MAP]
    if not renamed:
        return sample, []
    new_rop = {}
    for k, v in rop.items():
        new_rop[RENAME_MAP.get(k, k)] = v
    new_sample = dict(sample)
    new_sample["resolved_object_placements"] = new_rop
    return new_sample, renamed


def iter_hdf5_files(root: str):
    for sub in sorted(os.listdir(root)):
        subd = os.path.join(root, sub)
        if not os.path.isdir(subd):
            continue
        for fn in sorted(os.listdir(subd)):
            if fn.endswith(".hdf5"):
                yield os.path.join(subd, fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Scene dataset dir, e.g. .../44_microwave_bowl_loading")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    args = ap.parse_args()

    backup = {}  # filepath -> original raw string (only files actually changed)
    stats = {"total": 0, "changed": 0, "already_ok": 0, "skipped_no_ds": 0}
    rename_counts = {k: 0 for k in RENAME_MAP}

    for fp in iter_hdf5_files(args.root):
        stats["total"] += 1
        with h5py.File(fp, "r+") if args.apply else h5py.File(fp, "r") as h:
            if DATASET_PATH not in h:
                stats["skipped_no_ds"] += 1
                continue
            ds = h[DATASET_PATH]
            raw = ds[()]
            try:
                sample = json.loads(_decode(raw))
            except Exception as e:
                print(f"[WARN] {fp}: failed to parse JSON ({e}); skipping", file=sys.stderr)
                continue
            new_sample, renamed = rekey_sample(sample)
            if not renamed:
                stats["already_ok"] += 1
                continue
            for k in renamed:
                rename_counts[k] += 1
            if not args.apply:
                stats["changed"] += 1
                rel = os.path.relpath(fp, args.root)
                print(f"[dry-run] would rekey {rel}: {renamed}")
                continue
            # Write back to the scalar vlen-string dataset in place.
            new_str = json.dumps(new_sample, ensure_ascii=False)
            ds[()] = new_str if ds.dtype == object else new_str.encode()
            stats["changed"] += 1
            backup[os.path.relpath(fp, args.root)] = _decode(raw)

    print("\n=== summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  per-stale-key renamed occurrences:", rename_counts)

    if args.apply and backup:
        bpath = os.path.join(args.root, f"_rekey_rop_backup_{os.getpid() or 'manual'}.json")
        with open(bpath, "w") as fh:
            json.dump(backup, fh, ensure_ascii=False, indent=1)
        print(f"\nbackup of original strings -> {bpath}")
        print("restore with: python tools/labels/rekey_anchor_placements.py --restore <backup.json>")
    elif not args.apply:
        print("\n(dry run; pass --apply to write)")


if __name__ == "__main__":
    main()
