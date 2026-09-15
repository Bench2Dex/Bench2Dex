#!/usr/bin/env python3
"""Verify all episodes' scene config matches the reference table.

Usage:
    python tools/verify_scene_config.py --dataset-dir <path>
"""
from __future__ import annotations

import argparse, json, os, re, sys
import h5py
from sync_scene_config import REFERENCE_TABLE, _check_match, _offset_match, extract_floorplan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    d = os.path.abspath(args.dataset_dir)
    hdf5_files = sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.startswith("episode_") and f.endswith(".hdf5")
    )
    ok = mismatch = skip = 0
    errors = []

    for path in hdf5_files:
        try:
            with h5py.File(path, "r") as f:
                if "meta/scene_generalization_sample" not in f:
                    skip += 1; continue
                gs = json.loads(f["meta/scene_generalization_sample"][()].decode()
                                if isinstance(f["meta/scene_generalization_sample"][()], bytes)
                                else str(f["meta/scene_generalization_sample"][()]))
                bg = gs.get("appearance", {}).get("background", {})
                uri = bg.get("asset_uri", "") or ""
                fp = extract_floorplan(uri)
                if fp is None or fp not in REFERENCE_TABLE:
                    skip += 1; continue

                ref = REFERENCE_TABLE[fp]
                cur_yaw = float(bg.get("yaw_deg", 0))
                cur_base = float(bg.get("base_yaw_deg", 0))
                cur_off = [float(v) for v in bg.get("scene_offset", [0, 0, 0])]

                yaw_ok = _check_match(cur_yaw, ref["yaw"])
                base_ok = _check_match(cur_base, ref["yaw"])
                off_ok = _offset_match(cur_off, ref["offset"])

                if yaw_ok and base_ok and off_ok:
                    ok += 1
                else:
                    mismatch += 1
                    fname = os.path.basename(path)
                    parts = []
                    if not yaw_ok: parts.append(f"yaw={cur_yaw}≠{ref['yaw']}")
                    if not base_ok: parts.append(f"base={cur_base}≠{ref['yaw']}")
                    if not off_ok: parts.append(f"offset={cur_off}≠{ref['offset']}")
                    errors.append(f"  ❌ {fname} (FP{fp}): {', '.join(parts)}")
        except Exception as e:
            mismatch += 1
            errors.append(f"  ❌ {os.path.basename(path)}: ERROR {e}")

    total_fp = ok + mismatch
    for e in errors:
        print(e)
    dp = os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d)
    print(f"\n[{dp}] Total={len(hdf5_files)}  FloorPlan_OK={ok}  Mismatch={mismatch}  Skip={skip}")
    return 0 if mismatch == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
