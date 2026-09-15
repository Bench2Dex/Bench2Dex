#!/usr/bin/env python3
"""Sync scene background config in HDF5 episodes to match default.yaml.

Reads meta/scene_generalization_sample and meta/scene_generalization_config
from each episode, looks up the correct yaw_deg / scene_offset from the
built-in reference table (derived from configs/collect/default.yaml), and
updates mismatched entries in-place.

Usage:
    python tools/sync_scene_config.py --dataset-dir <path> --dry-run
    python tools/sync_scene_config.py --dataset-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

# ── Reference table from configs/collect/default.yaml ────────────────────
# All 57 FloorPlan entries.  Source of truth for yaw_deg and scene_offset.
REFERENCE_TABLE: Dict[str, Dict[str, Any]] = {
    "1":    {"yaw": 180.0, "offset": [0.0, 0.0, 0.0]},
    "2":    {"yaw": 180.0, "offset": [0.0, 0.0, 0.0]},
    "3":    {"yaw": 180.0, "offset": [0.0, 0.0, -0.22]},
    "4":    {"yaw": 180.0, "offset": [-1.96702, 1.89762, 0.0]},
    "5":    {"yaw": 180.0, "offset": [1.42053, 1.15041, 0.0]},
    "6":    {"yaw": 180.0, "offset": [-0.32854, 1.16282, 0.0]},
    "7":    {"yaw": 180.0, "offset": [0.0, 0.0, 0.0]},
    "8":    {"yaw": 180.0, "offset": [-0.58339, 2.01355, 0.0]},
    "9":    {"yaw": 90.0,  "offset": [0.0, 1.21873, 0.0]},
    "10":   {"yaw": 90.0,  "offset": [-0.73022, 0.88397, 0.0]},
    "11":   {"yaw": 180.0, "offset": [-1.5588, 0.22708, 0.0]},
    "12":   {"yaw": 0.0,   "offset": [0.15, 0.76969, 0.0]},
    "13":   {"yaw": -90.0, "offset": [-3.8804, -2.13417, 0.0]},
    "15":   {"yaw": 0.0,   "offset": [1.39, -1.58, 0.0]},
    "16":   {"yaw": 180.0, "offset": [1.25263, 3.65209, 0.0]},
    "17":   {"yaw": 180.0, "offset": [0.63808, 2.86054, 0.0]},
    "18":   {"yaw": 90.0,  "offset": [2.71921, 0.57477, 0.0]},
    "21":   {"yaw": 0.0,   "offset": [1.1808, 3.07023, 0.0]},
    "24":   {"yaw": 0.0,   "offset": [0.95853, -1.48004, 0.0]},
    "26":   {"yaw": 90.0,  "offset": [2.77407, 2.48283, 0.0]},
    "201":  {"yaw": 0.0,   "offset": [2.16777, -1.20934, 0.0]},
    "202":  {"yaw": 0.0,   "offset": [2.10875, -2.29375, 0.0]},
    "203":  {"yaw": -90.0, "offset": [-2.44349, 0.4154, 0.0]},
    "204":  {"yaw": -90.0, "offset": [-4.22613, -1.90774, 0.0]},
    "205":  {"yaw": 180.0, "offset": [-2.00438, 5.05686, 0.0]},
    "206":  {"yaw": 0.0,   "offset": [0.0, 1.02099, 0.0]},
    "207":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "208":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "209":  {"yaw": -90.0, "offset": [2.47887, -2.52504, 0.0]},
    "210":  {"yaw": 180.0, "offset": [-2.39328, 2.39618, 0.0]},
    "211":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "212":  {"yaw": 0.0,   "offset": [-0.97986, -0.25227, 0.0]},
    "213":  {"yaw": 0.0,   "offset": [-11.06503, -0.91725, 0.0]},
    "214":  {"yaw": -90.0, "offset": [-4.29684, -2.4183, 0.0]},
    "215":  {"yaw": 121.0, "offset": [2.35706, 4.6246, 0.0]},
    "216":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "217":  {"yaw": 0.0,   "offset": [2.34252, -2.6582, 0.0]},
    "218":  {"yaw": -60.0, "offset": [-1.35772, -1.88389, 0.0]},
    "220":  {"yaw": 0.0,   "offset": [2.55291, -1.69247, 0.0]},
    "222":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "224":  {"yaw": 90.0,  "offset": [-0.55907, -1.88972, 0.0]},
    "225":  {"yaw": 90.0,  "offset": [2.59771, 2.22918, 0.0]},
    "226":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "227":  {"yaw": 0.0,   "offset": [3.4296, -1.90511, 0.0]},
    "228":  {"yaw": 0.0,   "offset": [3.6471, -2.70407, 0.0]},
    "229":  {"yaw": 0.0,   "offset": [3.68556, -2.52339, 0.0]},
    "230":  {"yaw": 0.0,   "offset": [2.85912, -2.16399, 0.0]},
    "303":  {"yaw": 270.0, "offset": [0.98792, 1.41545, 0.0]},
    "307":  {"yaw": 180.0, "offset": [0.0, 0.77052, 0.0]},
    "309":  {"yaw": 180.0, "offset": [-1.97795, 1.72843, 0.0]},
    "311":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "315":  {"yaw": -90.0, "offset": [1.7814, -0.84301, 0.0]},
    "317":  {"yaw": 180.0, "offset": [0.0, 0.0, 0.0]},
    "318":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "323":  {"yaw": 180.0, "offset": [0.81476, 1.15264, 0.0]},
    "325":  {"yaw": 0.0,   "offset": [0.0, 0.0, 0.0]},
    "326":  {"yaw": 90.0,  "offset": [-1.77262, -0.7326, 0.0]},
}


def _check_match(cur_yaw: float, ref_yaw: float) -> bool:
    """Two yaw values match if they represent the same angle modulo 360."""
    return abs((cur_yaw - ref_yaw + 180.0) % 360.0 - 180.0) < 0.01


def _offset_match(cur: List[float], ref: List[float]) -> bool:
    return all(abs(a - b) < 0.001 for a, b in zip(cur, ref))


def extract_floorplan(uri: str) -> str | None:
    m = re.search(r"FloorPlan(\d+)_physics", uri)
    return m.group(1) if m else None


def load_json_dataset(f: h5py.File, key: str) -> dict | None:
    if key not in f:
        return None
    raw = f[key][()]
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def write_json_dataset(f: h5py.File, key: str, data: dict, str_dtype) -> None:
    if key in f:
        del f[key]
    f.create_dataset(key, data=np.asarray(json.dumps(data, ensure_ascii=False), dtype=str_dtype))


def sync_episode(
    hdf5_path: str,
    ref_table: Dict[str, Dict[str, Any]],
    *,
    dry_run: bool = True,
) -> Tuple[str, List[str]]:
    """Sync one episode file.  Returns (status, change_log)."""
    changes: List[str] = []
    fname = os.path.basename(hdf5_path)

    fp_num: str | None = None
    cur_yaw: float = 0.0
    cur_base_yaw: float = 0.0
    cur_offset: List[float] = [0.0, 0.0, 0.0]

    # ── Read current values ──────────────────────────────────────────
    with h5py.File(hdf5_path, "r") as f:
        sample = load_json_dataset(f, "meta/scene_generalization_sample")
        if sample is None:
            return "skip", ["no scene_generalization_sample"]

        bg = sample.get("appearance", {}).get("background", {})
        uri = bg.get("asset_uri", "") or ""
        if not uri:
            return "skip", ["no asset_uri (clean background)"]

        fp_num = extract_floorplan(uri)
        if fp_num is None:
            return "skip", [f"non-FloorPlan background: {os.path.basename(uri)}"]

        if fp_num not in ref_table:
            return "skip", [f"FloorPlan{fp_num} not in reference table"]

        cur_yaw = float(bg.get("yaw_deg", 0.0))
        cur_base_yaw = float(bg.get("base_yaw_deg", 0.0))
        cur_offset = [float(v) for v in bg.get("scene_offset", [0.0, 0.0, 0.0])]

        ref = ref_table[fp_num]
        ref_yaw = ref["yaw"]
        ref_offset = ref["offset"]

        yaw_ok = _check_match(cur_yaw, ref_yaw)
        base_ok = _check_match(cur_base_yaw, ref_yaw)
        off_ok = _offset_match(cur_offset, ref_offset)

        if yaw_ok and base_ok and off_ok:
            return "ok", []

        # Build change descriptions
        if not yaw_ok:
            changes.append(f"yaw_deg: {cur_yaw} → {ref_yaw}")
        if not base_ok:
            changes.append(f"base_yaw_deg: {cur_base_yaw} → {ref_yaw}")
        if not off_ok:
            changes.append(f"scene_offset: {cur_offset} → {ref_offset}")

        # Read config for later update
        config = load_json_dataset(f, "meta/scene_generalization_config")

    # ── Write updated values ─────────────────────────────────────────
    if dry_run:
        return "dry_run", changes

    with h5py.File(hdf5_path, "a") as f:
        str_dtype = h5py.string_dtype(encoding="utf-8")

        # Update sample
        sample = load_json_dataset(f, "meta/scene_generalization_sample")
        if sample is not None:
            sample["appearance"]["background"]["yaw_deg"] = ref_yaw
            sample["appearance"]["background"]["base_yaw_deg"] = ref_yaw
            sample["appearance"]["background"]["flip_180_applied"] = False
            sample["appearance"]["background"]["scene_offset"] = list(ref_offset)
            write_json_dataset(f, "meta/scene_generalization_sample", sample, str_dtype)

        # Update config (the matching asset entry)
        config = load_json_dataset(f, "meta/scene_generalization_config")
        if config is not None:
            assets = config.get("appearance", {}).get("background", {}).get("assets", [])
            updated = 0
            for asset in assets:
                asset_uri = asset.get("uri", "") or ""
                asset_fp = extract_floorplan(asset_uri)
                if asset_fp == fp_num:
                    cur_asset_yaw = float(asset.get("yaw_deg", 0.0))
                    cur_asset_off = [float(v) for v in asset.get("scene_offset", [0.0, 0.0, 0.0])]
                    if not _check_match(cur_asset_yaw, ref_yaw):
                        asset["yaw_deg"] = ref_yaw
                        updated += 1
                    if not _offset_match(cur_asset_off, ref_offset):
                        asset["scene_offset"] = list(ref_offset)
                        updated += 1
            if updated > 0:
                write_json_dataset(f, "meta/scene_generalization_config", config, str_dtype)

    return "updated", changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync scene config in HDF5 episodes to default.yaml reference.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing episode_*.hdf5 files.")
    parser.add_argument("--dry-run", action="store_true", help="Only report differences, do not modify files.")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: not a directory: {dataset_dir}")
        sys.exit(1)

    hdf5_files = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".hdf5")
    )
    if not hdf5_files:
        print(f"No episode_*.hdf5 files found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} episode(s) in {dataset_dir}")
    if args.dry_run:
        print("[DRY-RUN MODE] No files will be modified.\n")
    else:
        print("[LIVE MODE] Files will be modified in-place.\n")

    stats = {"ok": 0, "updated": 0, "dry_run": 0, "skip": 0}
    fp_hits: Dict[str, int] = {}

    for hdf5_path in hdf5_files:
        status, changes = sync_episode(hdf5_path, REFERENCE_TABLE, dry_run=args.dry_run)

        if status == "ok":
            stats["ok"] += 1
        elif status in ("updated", "dry_run"):
            fname = os.path.basename(hdf5_path)
            # Extract fp for grouping
            with h5py.File(hdf5_path, "r") as f:
                sample = load_json_dataset(f, "meta/scene_generalization_sample")
                uri = (sample or {}).get("appearance", {}).get("background", {}).get("asset_uri", "")
            fp = extract_floorplan(uri) or "?"
            fp_hits[fp] = fp_hits.get(fp, 0) + 1

            if status == "dry_run":
                stats["dry_run"] += 1
                print(f"  [DRY-RUN] {fname} (FloorPlan{fp}):")
            else:
                stats["updated"] += 1
                print(f"  [UPDATE]  {fname} (FloorPlan{fp}):")
            for change in changes:
                print(f"            {change}")
        else:
            stats["skip"] += 1

    print(f"\n{'='*60}")
    print(f"Summary: ok={stats['ok']}  updated={stats['updated']}  "
          f"dry_run={stats['dry_run']}  skip={stats['skip']}")
    if fp_hits:
        print(f"FloorPlans modified: {dict(sorted(fp_hits.items(), key=lambda x: int(x[0])))}")


if __name__ == "__main__":
    main()
