#!/usr/bin/env python3
"""Batch re-evaluate all replay-generalization episodes with current success criteria.

Reads every HDF5 episode under each task's replay-generalization directory,
runs the current custom evaluator on the final-frame state, and writes a new
``re_eval_<timestamp>.jsonl`` alongside the original data.

Usage:
    python tools/batch_re_eval.py [--dataset /path/to/dataset]
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# HDF5 state extraction
# ---------------------------------------------------------------------------

def _resolve_eval_frame(hdf5_path: str) -> int:
    """Return the object-state frame index to evaluate.

    Uses ``meta/homing_start_sim_step`` to find the frame just before the
    robot began its homing sequence — this is the human-labelled completion
    point.  Falls back to the last frame if the metadata is missing.
    """
    with h5py.File(hdf5_path, "r") as f:
        homing_step = None
        if "meta/homing_start_sim_step" in f:
            homing_step = int(f["meta/homing_start_sim_step"][()])
        if homing_step is not None and "time/sim_step" in f:
            sim_steps = f["time/sim_step"][:]
            idx = int(np.argmin(np.abs(sim_steps - homing_step)))
            return max(0, min(idx, len(sim_steps) - 1))
        # Fallback: last frame
        if "objects" in f:
            for obj_id in f["objects"]:
                shape = f[f"objects/{obj_id}/pose_world"].shape
                return shape[0] - 1
    return -1


def load_states_from_hdf5(hdf5_path: str) -> Dict[str, Any]:
    """Build states dict from the evaluation frame (homing start).

    Uses **recorded** lin_vel_world / ang_vel_world when available.
    """
    frame_idx = _resolve_eval_frame(hdf5_path)
    states: Dict[str, Any] = {}
    with h5py.File(hdf5_path, "r") as f:
        obj_group = f["objects"]
        for obj_id in obj_group:
            ds_pose = obj_group[obj_id]["pose_world"]
            if ds_pose.ndim != 2 or ds_pose.shape[1] != 7:
                continue
            i = min(frame_idx, ds_pose.shape[0] - 1)

            # Prefer recorded velocity, fall back to pose delta
            lin_vel = np.zeros(3, dtype=np.float64)
            ang_vel = np.zeros(3, dtype=np.float64)
            if f"objects/{obj_id}/lin_vel_world" in f:
                lv_ds = f[f"objects/{obj_id}/lin_vel_world"]
                if lv_ds.ndim == 2 and lv_ds.shape[0] > i:
                    lin_vel = lv_ds[i].astype(np.float64)
            if f"objects/{obj_id}/ang_vel_world" in f:
                av_ds = f[f"objects/{obj_id}/ang_vel_world"]
                if av_ds.ndim == 2 and av_ds.shape[0] > i:
                    ang_vel = av_ds[i].astype(np.float64)

            states[obj_id] = {
                "pose_world": ds_pose[i].astype(np.float64),
                "lin_vel_world": lin_vel,
                "ang_vel_world": ang_vel,
            }
    return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_evaluator(task_name: str) -> tuple[str, Any] | None:
    """Find the custom evaluator module for *task_name*."""
    scene_path = REPO_ROOT / "scenes" / f"{task_name}.yaml"
    if not scene_path.is_file():
        print(f"  [SKIP] Scene YAML not found: {scene_path}")
        return None

    import yaml
    with open(scene_path) as fh:
        scene = yaml.safe_load(fh) or {}

    conditions = scene.get("success_conditions", [])
    for cond in conditions:
        if cond.get("type") == "custom":
            evaluator = cond.get("evaluator", "")
            mod = importlib.import_module(evaluator)
            fn = getattr(mod, "check_success", None)
            if fn is None:
                print(f"  [SKIP] No check_success in {evaluator}")
                return None
            return evaluator, fn

    print(f"  [SKIP] No custom evaluator in scene YAML")
    return None


def evaluate_one_task(
    task_name: str,
    anchor_dir: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Re-evaluate all episodes for one task.  Returns summary dict."""
    resolved = resolve_evaluator(task_name)
    if resolved is None:
        return {"task": task_name, "error": "no_evaluator"}
    evaluator_name, check_success_fn = resolved

    hdf5_files = sorted(
        p for p in Path(anchor_dir).glob("episode_*.hdf5")
        if "_1." not in p.name and "replay" not in p.name.lower()
    )
    if not hdf5_files:
        return {"task": task_name, "error": "no_episodes", "n": 0}

    results: List[Dict[str, Any]] = []
    n_pass = 0

    for ep_path in hdf5_files:
        ep_idx = int(ep_path.stem.replace("episode_", ""))
        try:
            states = load_states_from_hdf5(str(ep_path))
            success = bool(check_success_fn(states, {}, {}))
        except Exception as exc:
            success = False
            results.append({"episode": ep_idx, "success": False, "error": str(exc)})
            continue
        results.append({"episode": ep_idx, "success": success})
        if success:
            n_pass += 1

    # Write output
    ts = datetime.now().strftime("%m%d_%H%M")
    out_path = output_root / f"re_eval_{ts}.jsonl"
    with open(out_path, "w") as f:
        for r in sorted(results, key=lambda x: x["episode"]):
            f.write(json.dumps(r) + "\n")

    summary_path = output_root / "re_eval_summary.json"
    summary: Dict[str, Any] = {
        "task": task_name,
        "evaluator": evaluator_name,
        "total": len(results),
        "pass": n_pass,
        "fail": len(results) - n_pass,
        "rate": n_pass / len(results) if results else 0.0,
    }
    # Merge with existing summary if present
    all_summaries = []
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = []
    all_summaries = [s for s in all_summaries if s.get("task") != task_name]
    all_summaries.append(summary)
    all_summaries.sort(key=lambda s: s.get("task", ""))
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    return summary


def main() -> None:
    parser = ArgumentParser(description="Batch re-evaluate replay-generalization episodes")
    parser.add_argument(
        "--dataset", type=str,
        default="../teleopdata/dataset",
        help="Path to teleopdata dataset root",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Output root directory (default: --dataset/../re_eval_results)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Specific tasks to evaluate (default: all with replay-generalization)",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output_root = Path(args.output) if args.output else dataset.parent / "re_eval_results"
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover tasks
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted(
            d.name for d in dataset.iterdir()
            if d.is_dir() and (d / "replay-generalization").is_dir()
        )

    print(f"Dataset: {dataset}")
    print(f"Output:  {output_root}")
    print(f"Tasks:   {len(task_names)}")
    print(f"{'='*60}")

    all_summaries = []
    for task_name in task_names:
        anchor_dir = dataset / task_name / "replay-generalization"
        task_output = output_root / task_name
        task_output.mkdir(parents=True, exist_ok=True)

        print(f"\n[{task_name}]")
        summary = evaluate_one_task(task_name, str(anchor_dir), task_output)
        all_summaries.append(summary)

        if "error" in summary:
            print(f"  ERROR: {summary['error']}")
        else:
            print(f"  {summary['pass']}/{summary['total']} ({summary['rate']*100:.1f}%)")

    # Write global summary
    global_path = output_root / "re_eval_summary.json"
    with open(global_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{'Task':<45} {'Pass':>5} {'Total':>6} {'Rate':>8}")
    print(f"{'-'*66}")
    total_pass = 0
    total_ep = 0
    for s in all_summaries:
        if "error" in s:
            print(f"{s['task']:<45} {'--':>5} {'--':>6} {'--':>8}  ({s['error']})")
        else:
            print(f"{s['task']:<45} {s['pass']:>5} {s['total']:>6} {s['rate']*100:>7.1f}%")
            total_pass += s["pass"]
            total_ep += s["total"]
    print(f"{'-'*66}")
    print(f"{'OVERALL':<45} {total_pass:>5} {total_ep:>6} {total_pass/total_ep*100:>7.1f}%" if total_ep else "")
    print(f"\nResults: {global_path}")


if __name__ == "__main__":
    main()
