#!/usr/bin/env python3
"""Rename stale ``plate`` keys in task-44 HDF5 ``labels/box3d`` and ``metrics``.

Companion to ``tools/fix_task44_hdf5_object_id.py`` (the first migration pass,
which already renamed ``objects/`` and the asset meta JSON).  That pass did NOT
touch ``labels/box3d`` or ``metrics/`` — so origin episodes still carry the old
names there:

* ``labels/box3d/obj_029_plate_3``                       → ``labels/box3d/obj_024_bowl_3``
* ``metrics/timeseries/grasp_<m>_obj_029_plate_3`` (×9)  → ``..._obj_024_bowl_3``
* ``metrics/timeseries/stage_plate_inside_{completed,current}`` → ``stage_bowl_in_mw_{...}``

The target names mirror the rename in commit ``c3f052d`` ("rename plate→bowl"):
object ``obj_029_plate_3``→``obj_024_bowl_3`` and stage id ``plate_inside``→
``bowl_in_mw``.  Values are untouched (only link names change).

Scope is ``origin-generalization`` only by default (replay's ``labels/box3d`` is
already bowl-named; replay ``metrics`` still carries ``plate`` — see notes in
``--help`` / run output).

Usage:
    python tools/labels/rekey_task44_metrics_box3d.py DATASET_ROOT [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py

OLD_OBJ, NEW_OBJ = "obj_029_plate_3", "obj_024_bowl_3"
OLD_STAGE, NEW_STAGE = "plate_inside", "bowl_in_mw"

# grasp metric suffix keys: grasp_<metric>_obj_029_plate_3  → grasp_<metric>_obj_024_bowl_3
GRASP_PREFIX = "grasp_"


def planned_moves(h: h5py.File) -> list[tuple[str, str]]:
    """Build the list of (src, dst) link moves for this file (only existing srcs)."""
    moves: list[tuple[str, str]] = []

    # box3d group (sub-datasets come along for free)
    src = f"labels/box3d/{OLD_OBJ}"
    dst = f"labels/box3d/{NEW_OBJ}"
    if src in h and dst not in h:
        moves.append((src, dst))

    ts = h.get("metrics/timeseries")
    if isinstance(ts, h5py.Group):
        for key in list(ts.keys()):
            if key.startswith(GRASP_PREFIX) and key.endswith(f"_{OLD_OBJ}"):
                metric = key[len(GRASP_PREFIX):-len(f"_{OLD_OBJ}")]
                new_key = f"{GRASP_PREFIX}{metric}_{NEW_OBJ}"
                if new_key not in ts:
                    moves.append((f"metrics/timeseries/{key}", f"metrics/timeseries/{new_key}"))
            elif key == f"stage_{OLD_STAGE}_completed":
                moves.append((f"metrics/timeseries/{key}", f"metrics/timeseries/stage_{NEW_STAGE}_completed"))
            elif key == f"stage_{OLD_STAGE}_current":
                moves.append((f"metrics/timeseries/{key}", f"metrics/timeseries/stage_{NEW_STAGE}_current"))

    return moves


def fix_one(path: Path, apply: bool) -> tuple[bool, list[tuple[str, str]]]:
    """Return (changed, moves). Uses atomic copy→replace when applying."""
    with h5py.File(path, "r") as f:
        moves = planned_moves(f)
    if not moves:
        return False, []

    if not apply:
        return True, moves

    # Atomic: operate on a sibling copy, then os.replace.
    tmp = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(path, tmp)
    try:
        with h5py.File(tmp, "r+") as f:
            for src, dst in moves:
                f.move(src, dst)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return True, moves


def iter_files(root: Path, subdirs):
    for sub in subdirs:
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.hdf5")):
            yield p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Dataset root, e.g. .../44_microwave_bowl_loading")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    ap.add_argument("--subdirs", nargs="*", default=["origin-generalization"],
                    help="Subdirs to process (default: origin-generalization only).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr); sys.exit(1)

    print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}   subdirs: {args.subdirs}")
    summary = {"files": 0, "changed": 0, "skipped": 0}
    manifest = {}
    for p in iter_files(root, args.subdirs):
        summary["files"] += 1
        try:
            changed, moves = fix_one(p, args.apply)
        except Exception as e:
            print(f"  ✗ {p.relative_to(root)}: {e}", file=sys.stderr); continue
        rel = str(p.relative_to(root))
        if changed:
            summary["changed"] += 1
            manifest[rel] = moves
            n = len(moves)
            print(f"  {'✓' if args.apply else '~'} {rel}  ({n} link moves)")
        else:
            summary["skipped"] += 1

    print(f"\nsummary: {summary}")
    if args.apply and manifest:
        bpath = root / f"_rekey_metrics_box3d_backup_{os.getpid() or 'manual'}.json"
        with open(bpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"move manifest -> {bpath}\n(to revert: swap src/dst in each entry and re-move)")


if __name__ == "__main__":
    main()
