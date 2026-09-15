#!/usr/bin/env python3
"""Audit scene-YAML vs HDF5 object/stage name drift across all collected tasks.

For every task that has collected data under ``teleopdata/dataset/<task>`` and a
matching scene YAML under ``scenes/<task>.yaml``, compare the YAML's declared
object ids / asset keys / stage ids against what each HDF5 actually stores in:

* ``/objects/<id>``                            (object pose group)
* ``meta/scene_generalization_sample`` JSON    (``resolved_object_placements`` keys)
* ``labels/box3d/<id>``                        (3D box label group)
* ``metrics/timeseries/grasp_*_obj_<id>``      (grasp metric suffix)
* ``metrics/timeseries/stage_<stage_id>_*``    (stage metric slug)
* ``meta/object_asset_keys/paths/roles`` JSON  (asset key per object)

Distractor/clutter keys (``distractor_*``) are excluded — task objects are the
``obj_*`` ones, which must be stable across episodes.  Any ``obj_*`` id present
in the HDF5 but not in the YAML (or vice-versa) is reported as drift, exactly
like the task-44 ``obj_029_plate_3`` → ``obj_024_bowl_3`` stale key.

Usage:
    python tools/labels/audit_scene_hdf5_name_drift.py [--scenes P] [--data P] [--all]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import h5py
import yaml

SCENES = "./scenes"
DATA = "../teleopdata/dataset"


def task_objs_stages(scene_path: str):
    """Return (obj_ids, {obj_id: asset_key}, stage_ids).

    ``stages:`` lives under ``metrics:`` (not top-level) in these scenes, so we
    look there first and fall back to a top-level ``stages:`` for safety.
    """
    with open(scene_path) as f:
        t = yaml.safe_load(f)
    obj_ids, asset_map = [], {}
    for o in t.get("objects", []) or []:
        if isinstance(o, dict) and "id" in o:
            obj_ids.append(str(o["id"]))
            if "asset" in o:
                asset_map[str(o["id"])] = str(o["asset"])
    stages = (t.get("stages") or
              (t.get("metrics") or {}).get("stages") or
              (t.get("terminal") or {}).get("stages") or
              [])
    stage_ids = [str(s.get("id")) for s in stages if isinstance(s, dict) and "id" in s]
    return obj_ids, asset_map, stage_ids


def is_asset_code(v: str) -> bool:
    """Real asset keys look like ``a024_bowl``; clutter values are bare codes."""
    return isinstance(v, str) and v.startswith("a") and v[1:4].isdigit()


def is_task_obj(key: str) -> bool:
    return key.startswith("obj_") and not key.startswith("distractor_")


def decode(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def scan_file(fp, acc):
    """acc: dict of location -> set(obj_id) ; stage -> set ; asset_key -> set ."""
    try:
        with h5py.File(fp, "r") as h:
            asset_map = acc.setdefault("asset_map", {})  # dict, not the defaultdict(set)
            if "objects" in h:
                for k in h["objects"].keys():
                    if is_task_obj(k):
                        acc["objects"].add(k)
            if "meta/scene_generalization_sample" in h:
                try:
                    s = json.loads(decode(h["meta/scene_generalization_sample"][()]))
                    for k in (s.get("resolved_object_placements") or {}):
                        if is_task_obj(k):
                            acc["resolved"].add(k)
                except Exception:
                    acc["err"].add("scene_generalization_sample parse")
            if "labels/box3d" in h:
                for k in h["labels/box3d"].keys():
                    if is_task_obj(k):
                        acc["box3d"].add(k)
            ts = h.get("metrics/timeseries")
            if isinstance(ts, h5py.Group):
                for k in ts.keys():
                    if k.startswith("grasp_") and "_obj_" in k:
                        oid = k.rsplit("_obj_", 1)[1]
                        if is_task_obj(oid):
                            acc["grasp"].add(oid)
                    elif k.startswith("stage_") and (k.endswith("_completed") or k.endswith("_current")):
                        slug = k[len("stage_"):].rsplit("_", 1)[0]
                        acc["stage"].add(slug)
            for fld in ("object_asset_keys", "object_asset_paths", "object_roles"):
                p = f"meta/{fld}"
                if p in h:
                    try:
                        v = json.loads(decode(h[p][()]))
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if is_task_obj(kk):
                                    acc[fld].add(kk)
                                    # per-object asset key (only object_asset_keys carries asset codes)
                                    if fld == "object_asset_keys" and is_asset_code(vv):
                                        asset_map[kk] = str(vv)
                    except Exception:
                        acc["err"].add(f"{fld} parse")
    except Exception as e:
        acc["err"].add(f"open: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default=SCENES)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--all", action="store_true", help="Print per-task report even when clean.")
    args = ap.parse_args()

    tasks = sorted(d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d)))
    print(f"dataset tasks: {len(tasks)}")

    problems = 0
    for task in tasks:
        scene = os.path.join(args.scenes, f"{task}.yaml")
        if not os.path.exists(scene):
            print(f"\n[{task}] NO scene YAML at {scene} — skip")
            problems += 1
            continue
        y_objs, y_asset_map, y_stages = task_objs_stages(scene)
        y_obj_set, y_stage_set = set(y_objs), set(y_stages)

        acc = defaultdict(set)
        nfiles = 0
        for root, _, files in os.walk(os.path.join(args.data, task)):
            for fn in files:
                if fn.endswith(".hdf5"):
                    scan_file(os.path.join(root, fn), acc)
                    nfiles += 1

        lines = []
        # object-id drift per location
        locs = ["objects", "resolved", "box3d", "grasp", "object_asset_keys", "object_asset_paths", "object_roles"]
        obj_drift = False
        for loc in locs:
            hdf5_ids = acc.get(loc, set())
            if not hdf5_ids:
                continue
            extra = hdf5_ids - y_obj_set
            missing = y_obj_set - hdf5_ids
            if extra or missing:
                obj_drift = True
                lines.append(f"  [{loc}] YAML={sorted(y_obj_set)} HDF5={sorted(hdf5_ids)}")
                if extra: lines.append(f"      STALE/extra in HDF5 (not in YAML): {sorted(extra)}")
                if missing: lines.append(f"      MISSING in HDF5 (in YAML, not stored): {sorted(missing)}")
        # asset-key value drift — only for obj_ids present in BOTH yaml and hdf5
        hdf5_asset_map = acc.get("asset_map", {})
        if hdf5_asset_map:
            common = y_obj_set & set(hdf5_asset_map)
            for oid in sorted(common):
                ya, ha = y_asset_map.get(oid), hdf5_asset_map.get(oid)
                if ya is not None and ha is not None and ya != ha:
                    obj_drift = True
                    lines.append(f"  [asset_key] {oid}: YAML asset={ya}  HDF5 asset={ha}  (stale asset code)")
        # stage drift
        hdf5_stages = acc.get("stage", set())
        stage_drift = False
        if y_stage_set or hdf5_stages:
            extra_s = hdf5_stages - y_stage_set
            missing_s = y_stage_set - hdf5_stages
            if extra_s or missing_s:
                stage_drift = True
                lines.append(f"  [stage] YAML stage ids={sorted(y_stage_set)} HDF5 stage slugs={sorted(hdf5_stages)}")
                if extra_s: lines.append(f"      STALE stage slugs (not in YAML): {sorted(extra_s)}")
                if missing_s: lines.append(f"      MISSING stage slugs: {sorted(missing_s)}")

        if acc.get("err"):
            lines.append(f"  [errors] {sorted(acc['err'])}")

        if obj_drift or stage_drift or acc.get("err") or args.all:
            print(f"\n[{task}] files={nfiles}")
            if not lines:
                print("  consistent ✓")
            else:
                for ln in lines:
                    print(ln)
        if obj_drift or stage_drift:
            problems += 1

    print(f"\n=== {len(tasks)} tasks scanned; tasks with drift: {problems} ===")


if __name__ == "__main__":
    main()
