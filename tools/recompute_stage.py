#!/usr/bin/env python
"""Recompute stage progress (latched_stage_completion_rate) from recorded episodes.

Replays a task's StageTracker over each recorded episode hdf5
(``objects/<id>/{pose_world, lin_vel_world, ang_vel_world, qpos, qvel,
joint_names}``) frame by frame, so stage progress can be recovered even when the
original inference run did not compute stages (or computed them incorrectly).

Usage:
  python tools/recompute_stage.py <eval_dir> [--task-id TASK_ID] [--dt DT]
  python tools/recompute_stage.py <eval_dir> --compare   # also diff vs per_episode.jsonl

Example:
  python tools/recompute_stage.py \
      ../output/end_eval/79/gr00t_all_0908_1856_xarm7_with_ability/none \
      --compare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.stage_tracker import StageTracker  # noqa: E402


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


def _load_arrays(file: h5py.File) -> tuple[dict[str, dict], int]:
    """Read every object's full per-frame arrays once (avoids per-frame h5py I/O)."""
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


def _frame_states(data: dict[str, dict], frame_index: int) -> dict[str, dict]:
    """Slice one frame's states out of the pre-loaded arrays."""
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
        tracker = StageTracker(stages, dt=dt)
        for i in range(count):
            tracker.update(_frame_states(data, i), sim_step=i)
        return tracker._result()


def load_stages(task_id: str) -> list[dict]:
    scene_files = sorted(glob.glob(str(REPO_ROOT / "scenes" / f"{task_id}_*.yaml")))
    if not scene_files:
        raise SystemExit(f"No scene file found for task_id={task_id} under scenes/{task_id}_*.yaml")
    scene = yaml.safe_load(open(scene_files[0])) or {}
    stages = (scene.get("metrics") or {}).get("stages") or scene.get("stages") or []
    if not stages:
        raise SystemExit(f"Task {task_id} has no metrics.stages; nothing to recompute.")
    return stages


def find_episode_hdfs(eval_dir: Path) -> list[Path]:
    hdfs = sorted(glob.glob(str(eval_dir / "episodes" / "**" / "*.hdf5"), recursive=True))
    if not hdfs:
        hdfs = sorted(glob.glob(str(eval_dir / "**" / "*.hdf5"), recursive=True))
    return [Path(p) for p in hdfs]


def infer_task_id(eval_dir: Path) -> str | None:
    parts = eval_dir.resolve().parts
    for i, part in enumerate(parts):
        if part == "end_eval" and i + 1 < len(parts):
            return parts[i + 1]
    for part in reversed(parts):
        if part.isdigit():
            return part
    return None


def _episode_index(path: Path) -> int:
    name = path.stem  # e.g. episode_000003
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute stage progress from recorded episode hdf5 files.")
    parser.add_argument("eval_dir", help="Evaluation directory containing episodes/*.hdf5")
    parser.add_argument("--task-id", help="Task ID (default: inferred from eval_dir path)")
    parser.add_argument("--dt", type=float, default=1 / 60, help="Physics dt for StageTracker (default 1/60)")
    parser.add_argument("--compare", action="store_true", help="Diff against per_episode.jsonl if present")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    task_id = args.task_id or infer_task_id(eval_dir)
    if task_id is None:
        raise SystemExit("Could not infer task_id; pass --task-id explicitly.")

    stages = load_stages(task_id)
    print(f"Task {task_id}: {len(stages)} stages -> {[s.get('id') for s in stages]}")
    print(f"dt={args.dt}, eval_dir={eval_dir}")

    hdfs = find_episode_hdfs(eval_dir)
    if not hdfs:
        raise SystemExit(f"No episode hdf5 files found under {eval_dir}")
    print(f"Found {len(hdfs)} episode hdf5 files\n")

    results: dict[int, tuple[float, dict]] = {}
    for hdf in hdfs:
        ep = _episode_index(hdf)
        result = recompute_episode(hdf, stages, args.dt)
        results[ep] = (result.latched_stage_completion_rate, result.completed)

    # Summary
    dist = Counter(lat for lat, _ in results.values())
    print("=== recomputed latched_stage_completion_rate distribution ===")
    for value in sorted(dist):
        print(f"  {value}: {dist[value]} episodes")
    total = len(results)
    avg = sum(lat * n for lat, n in dist.items()) / total if total else 0.0
    print(f"  total={total}, mean={avg:.4f}")

    # Optional comparison against recorded per_episode.jsonl
    if args.compare:
        per_ep = eval_dir / "per_episode.jsonl"
        if per_ep.exists():
            recorded: dict[int, float] = {}
            with open(per_ep) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "episode_index" in row and "latched_stage_completion_rate" in row:
                        recorded[row["episode_index"]] = row["latched_stage_completion_rate"]
            mismatch = []
            for ep, (lat, _) in results.items():
                if ep in recorded and recorded[ep] != lat:
                    mismatch.append((ep, recorded[ep], lat))
            print(f"\n=== comparison vs {per_ep.name} ===")
            print(f"  recorded episodes: {len(recorded)}, recomputed: {total}, mismatches: {len(mismatch)}")
            for ep, rec, lat in mismatch[:30]:
                print(f"    ep {ep}: recorded={rec} recomputed={lat}")
        else:
            print(f"\n(no {per_ep.name} found; skipped comparison)")


if __name__ == "__main__":
    main()
