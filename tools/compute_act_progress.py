#!/usr/bin/env python
"""Compute average latched_stage_completion_rate per task/profile from ACT inference recordings."""

from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage_tracker import StageTracker

ACT_DIR = Path("/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/Bench2Dex/outputs/inference_recordings/act")
SCENES_DIR = Path("/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/Bench2Dex/scenes")
OUTPUT_FILE = Path("/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff/Bench2Dex/outputs/inference_recordings/inference_progress_summary.txt")


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _frame_count(file: h5py.File) -> int:
    meta = file.get("meta")
    if meta is not None and "frame_count" in meta:
        return int(meta["frame_count"][()])
    objects = file.get("objects")
    if objects is not None:
        for grp in objects.values():
            if "pose_world" in grp:
                return int(grp["pose_world"].shape[0])
    return 0


def _load_arrays(file: h5py.File):
    data: dict[str, dict] = {}
    count = _frame_count(file)
    objects = file.get("objects")
    if objects is None:
        return data, count
    for obj_id, grp in objects.items():
        entry: dict = {}
        if "pose_world" in grp:
            entry["pose_world"] = np.asarray(grp["pose_world"][:], dtype=np.float32)
        if "lin_vel_world" in grp:
            entry["lin_vel_world"] = np.asarray(grp["lin_vel_world"][:], dtype=np.float32)
        if "ang_vel_world" in grp:
            entry["ang_vel_world"] = np.asarray(grp["ang_vel_world"][:], dtype=np.float32)
        if "qpos" in grp:
            entry["qpos"] = np.asarray(grp["qpos"][:], dtype=np.float32)
        if "qvel" in grp:
            entry["qvel"] = np.asarray(grp["qvel"][:], dtype=np.float32)
        if "joint_names" in grp:
            entry["joint_names"] = [_decode(n) for n in grp["joint_names"][:]]
        data[obj_id] = entry
    return data, count


def _frame_states(data: dict[str, dict], frame_index: int):
    states: dict[str, dict] = {}
    for obj_id, entry in data.items():
        state: dict = {}
        pose = entry.get("pose_world")
        if pose is not None:
            if not np.all(np.isfinite(pose[frame_index])):
                continue
            state["pose_world"] = pose[frame_index]
        if "lin_vel_world" in entry:
            state["lin_vel_world"] = entry["lin_vel_world"][frame_index]
        if "ang_vel_world" in entry:
            state["ang_vel_world"] = entry["ang_vel_world"][frame_index]
        if "qpos" in entry:
            state["qpos"] = entry["qpos"][frame_index]
        if "qvel" in entry:
            state["qvel"] = entry["qvel"][frame_index]
        if "joint_names" in entry:
            state["joint_names"] = entry["joint_names"]
        states[obj_id] = state
    return states


def recompute_episode(hdf_path: Path, stages: list[dict], dt: float):
    with h5py.File(hdf_path, "r") as file:
        data, count = _load_arrays(file)
        if count == 0 or not data:
            return None
        tracker = StageTracker(stages, dt=dt)
        for i in range(count):
            tracker.update(_frame_states(data, i), sim_step=i)
        return tracker._result()


def load_stages(task_id: str) -> list[dict] | None:
    candidates = sorted(glob.glob(str(SCENES_DIR / f"{task_id}_*.yaml")))
    if not candidates:
        num_part = "".join(c for c in task_id if c.isdigit())
        if num_part:
            candidates = sorted(glob.glob(str(SCENES_DIR / f"{num_part}_*.yaml")))
    if not candidates:
        return None
    scene = yaml.safe_load(open(candidates[0])) or {}
    stages = (scene.get("metrics") or {}).get("stages") or scene.get("stages") or []
    return stages if stages else None


def main():
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    skip_no_stages = []
    skip_no_episodes = []

    task_dirs = sorted(d for d in ACT_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not task_dirs:
        task_dirs = sorted(d for d in ACT_DIR.iterdir() if d.is_dir() and any(c.isdigit() for c in d.name))

    for task_dir in task_dirs:
        task_id = task_dir.name
        stages = load_stages(task_id)
        if stages is None:
            skip_no_stages.append(task_id)
            continue

        profile_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        for profile_dir in profile_dirs:
            profile = profile_dir.name
            hdf_files = []
            for subdir in ("success", "failure"):
                sub_path = profile_dir / subdir
                if sub_path.is_dir():
                    hdf_files.extend(sorted(sub_path.glob("*.hdf5")))
            if not hdf_files:
                skip_no_episodes.append((task_id, profile))
                continue

            rates = []
            for hdf in hdf_files:
                try:
                    result = recompute_episode(hdf, stages, dt=1 / 60)
                    if result is not None:
                        rates.append(result.latched_stage_completion_rate)
                except Exception as e:
                    print(f"  [warn] {hdf}: {e}", file=sys.stderr)

            if rates:
                results[task_id][profile] = rates

    lines = []
    for task_id in sorted(results.keys(), key=lambda x: int("".join(c for c in x if c.isdigit()) or "999")):
        task_profiles = results[task_id]
        lines.append(f"task {task_id}")
        for profile in sorted(task_profiles.keys()):
            rates = task_profiles[profile]
            avg = float(np.mean(rates))
            lines.append(f"  {profile}  {avg:.4f}")
        lines.append("=" * 40)

    output = "\n".join(lines) + "\n"
    print(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)
    print(f"Written to {OUTPUT_FILE}", file=sys.stderr)

    if skip_no_stages:
        print(f"\nSkipped (no scene stages): {skip_no_stages}", file=sys.stderr)
    if skip_no_episodes:
        print(f"Skipped (no episode hdf5): {skip_no_episodes}", file=sys.stderr)


if __name__ == "__main__":
    main()
