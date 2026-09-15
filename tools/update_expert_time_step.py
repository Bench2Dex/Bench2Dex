#!/usr/bin/env python3
"""Compute mean steps_to_stable_success from replay HDF5 data and write to scene YAML.

Two modes (controlled by ``--mode``):

  ``evaluate`` (default)
      Run the task's check_success() evaluator over saved object trajectories
      to determine success and steps_to_stable_success. **Does not modify HDF5.**
      Requires ``--task`` to specify the evaluator module.

  ``stored``
      Read ``meta/success`` and ``metrics/episode/steps_to_stable_success``
      that were already written into the HDF5 files (by collection or replay).

Usage::

    # Evaluate success on-the-fly (default, recommended)
    python tools/update_expert_time_step.py \\
        --task success.custom.task_51_toilet_lid_cleaner_pour \\
        /path/to/dataset/51_toilet_lid_cleaner_pour/replay-generalization

    # Read previously-stored labels
    python tools/update_expert_time_step.py --mode stored \\
        /path/to/dataset/60_breadbasket_fast_food_loading/replay-generalization

    # Dry-run (print what would change)
    python tools/update_expert_time_step.py --dry-run --task <module> /path/to/...

    # Batch: process ALL scenes
    python tools/update_expert_time_step.py --all --mode evaluate \\
        --dataset-root ../teleopdata/dataset
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _scalar_bool(dataset) -> bool:
    val = dataset[()]
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode("utf-8")
    return bool(val)


def _scalar_int_or_none(dataset) -> Optional[int]:
    val = dataset[()]
    if val is None:
        return None
    if isinstance(val, (bytes, np.bytes_)):
        s = val.decode("utf-8").strip()
        if s in ("null", "None", ""):
            return None
        try:
            return int(s)
        except ValueError:
            return None
    try:
        v = int(val)
        if np.isnan(float(val)):
            return None
        return v
    except (ValueError, TypeError):
        return None


def _load_states_at_frame(h5: h5py.File, obj_ids: list[str], frame_idx: int) -> dict:
    """Build a states dict for one frame from recorded object trajectories."""
    states: dict = {}
    for obj_id in obj_ids:
        group_key = f"objects/{obj_id}"
        pose_key = f"{group_key}/pose_world"
        if pose_key not in h5:
            continue
        state = {"pose_world": h5[pose_key][frame_idx]}
        qpos_key = f"{group_key}/qpos"
        if qpos_key in h5:
            qpos_array = h5[qpos_key][frame_idx]
            jn_key = f"{group_key}/joint_names"
            if jn_key in h5:
                jnames = [n.decode() if isinstance(n, bytes) else str(n) for n in h5[jn_key][:]]
                state["qpos"] = {n: float(qpos_array[i]) for i, n in enumerate(jnames) if i < len(qpos_array)}
                state["joint_names"] = jnames
            else:
                state["qpos"] = {f"joint_{i}": float(v) for i, v in enumerate(qpos_array)}
        states[obj_id] = state
    return states


def _evaluate_success(
    h5: h5py.File,
    check_success_fn,
    *,
    dwell_s: float = 0.5,
    task_params: dict | None = None,
) -> Tuple[bool, Optional[int]]:
    """Run evaluator over saved trajectories. Returns (success, steps_to_stable).

    Does NOT modify the HDF5 file.
    """
    frame_count = int(h5["meta/frame_count"][()])
    obj_ids = list(h5["objects"].keys())
    step_stride = int(h5["meta/step_stride"][()]) if "meta/step_stride" in h5 else 3
    fps = float(h5["meta/effective_fps"][()]) if "meta/effective_fps" in h5 else 20.0
    dwell_frames = max(1, int(round(dwell_s * fps)))

    ctx: dict = {}
    success = False
    stable_success = False
    steps_to_stable = None
    first_success_frame = None
    consecutive = 0

    for t in range(frame_count):
        states = _load_states_at_frame(h5, obj_ids, t)
        frame_ok = check_success_fn(states, ctx, task_params or {})
        if frame_ok:
            if first_success_frame is None:
                first_success_frame = t
            consecutive += 1
            success = True
            if consecutive >= dwell_frames and not stable_success:
                stable_success = True
                steps_to_stable = (first_success_frame + dwell_frames) * step_stride
        else:
            consecutive = 0
            first_success_frame = None

    # Fallback: use homing_start_sim_step (capped to full trajectory length)
    if steps_to_stable is None:
        traj_steps = frame_count * step_stride
        homing = h5.get("meta/homing_start_sim_step")
        if homing is not None:
            steps_to_stable = min(int(homing[()]), traj_steps)
        else:
            steps_to_stable = traj_steps

    return success, steps_to_stable


def compute_stats_stored(
    dataset_dir: str,
) -> Tuple[int, int, float, float, List[int]]:
    """Read pre-stored meta/success and steps_to_stable_success from HDF5."""
    pattern = os.path.join(dataset_dir, "episode_*.hdf5")
    all_files = sorted(glob.glob(pattern))
    main_files = [f for f in all_files if not re.search(r"episode_\d+_\d+\.hdf5$", os.path.basename(f))]
    if not main_files:
        raise SystemExit(f"ERROR: no episode_*.hdf5 files found in {dataset_dir}")

    success_steps: List[int] = []
    total = 0
    for fpath in main_files:
        total += 1
        with h5py.File(fpath, "r") as h5:
            try:
                ok = _scalar_bool(h5["meta"]["success"])
            except (KeyError, TypeError, ValueError):
                continue
            if not ok:
                continue
            try:
                steps = _scalar_int_or_none(h5["metrics"]["episode"]["steps_to_stable_success"])
            except (KeyError, TypeError):
                continue
            if steps is not None and steps > 0:
                success_steps.append(steps)

    n_success = len(success_steps)
    if n_success == 0:
        return total, 0, 0.0, 0.0, []
    arr = np.array(success_steps, dtype=np.float64)
    return total, n_success, float(np.mean(arr)), float(np.median(arr)), success_steps


def compute_stats_evaluate(
    dataset_dir: str,
    task_module: str,
    *,
    task_params: dict | None = None,
) -> Tuple[int, int, float, float, List[int]]:
    """Run the evaluator over saved trajectories (read-only, no HDF5 writes)."""
    mod = importlib.import_module(task_module)
    check_fn = mod.check_success

    pattern = os.path.join(dataset_dir, "episode_*.hdf5")
    all_files = sorted(glob.glob(pattern))
    main_files = [f for f in all_files if not re.search(r"episode_\d+_\d+\.hdf5$", os.path.basename(f))]
    if not main_files:
        raise SystemExit(f"ERROR: no episode_*.hdf5 files found in {dataset_dir}")

    success_steps: List[int] = []
    total = 0
    for fpath in main_files:
        total += 1
        with h5py.File(fpath, "r") as h5:
            success, steps = _evaluate_success(h5, check_fn, task_params=task_params)
            if success and steps is not None and steps > 0:
                success_steps.append(steps)

    n_success = len(success_steps)
    if n_success == 0:
        return total, 0, 0.0, 0.0, []
    arr = np.array(success_steps, dtype=np.float64)
    return total, n_success, float(np.mean(arr)), float(np.median(arr)), success_steps


# ---------------------------------------------------------------------------
# YAML path resolution
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO_ROOT / "scenes"
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _extract_task_name(dataset_dir: str) -> str:
    """Extract task_name from a path like .../dataset/<task_name>/<split>.

    Walks up until we see a ``dataset`` directory, then takes the next segment.
    """
    p = Path(dataset_dir).resolve()
    parts = p.parts
    try:
        idx = parts.index("dataset")
    except ValueError:
        raise SystemExit(
            f"ERROR: cannot locate 'dataset' segment in path: {dataset_dir}\n"
            f"Use --scene to specify the target YAML explicitly."
        )
    if idx + 1 >= len(parts):
        raise SystemExit(
            f"ERROR: path ends at 'dataset' - no task name found: {dataset_dir}"
        )
    return parts[idx + 1]


def resolve_scene_yaml(
    dataset_dir: str,
    scene_override: Optional[str] = None,
) -> Path:
    """Find the scene YAML corresponding to *dataset_dir*.

    Resolution order:
    1. If *scene_override* is given, use it directly.
    2. Exact match: scenes/<task_name>.yaml
    3. Prefix match by numeric prefix (e.g. "60") — requires exactly one hit.
    """
    if scene_override:
        candidate = Path(scene_override)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not candidate.exists():
            raise SystemExit(f"ERROR: --scene target not found: {candidate}")
        return candidate

    task_name = _extract_task_name(dataset_dir)

    # Exact match
    exact = SCENES_DIR / f"{task_name}.yaml"
    if exact.exists():
        return exact

    # Prefix match (numeric)
    m = re.match(r"^(\d+)", task_name)
    if m:
        prefix = m.group(1)
        glob_pattern = f"{prefix}_*.yaml"
        candidates = sorted(SCENES_DIR.glob(glob_pattern))
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise SystemExit(
                f"ERROR: multiple scene YAMLs match prefix '{prefix}':\n"
                + "\n".join(f"  {c.name}" for c in candidates)
                + "\nUse --scene to disambiguate."
            )

    raise SystemExit(
        f"ERROR: no scene YAML found for task '{task_name}'.\n"
        f"  Tried: {exact}\n"
        f"  Use --scene to specify the target YAML explicitly."
    )


# ---------------------------------------------------------------------------
# YAML read / write (line-based to preserve comments & formatting)
# ---------------------------------------------------------------------------

def _read_yaml_lines(path: Path) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def _write_yaml_lines(path: Path, lines: List[str]) -> None:
    with open(path, "w") as f:
        f.writelines(lines)


def update_scene_yaml(
    scene_path: Path,
    expert_time_step: int,
    success_count: int,
    total_count: int,
    split_name: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Write *expert_time_step* into the scene YAML.

    - Replaces an existing ``expert_time_step`` line (idempotent).
    - Removes legacy ``expert_time_s`` line if present.
    - Replaces an existing ``expert_time_source`` line.
    - If neither exists, inserts after the ``task_family`` line.

    Returns True if the file was modified.
    """
    new_step_line = f"  expert_time_step: {expert_time_step}\n"
    new_source_line = '  expert_time_source: "mean steps_to_stable_success over {}/{} successful demos from {}"\n'.format(
        success_count, total_count, split_name
    )

    lines = _read_yaml_lines(scene_path)
    new_lines: List[str] = []
    modified = False
    handled_step = False
    handled_source = False
    insert_after_task_family = -1

    for line in lines:
        stripped = line.lstrip()

        # Skip legacy expert_time_s (replace is handled by expert_time_step below)
        if stripped.startswith("expert_time_s:"):
            modified = True
            continue

        # Replace existing expert_time_step
        if not handled_step and stripped.startswith("expert_time_step:"):
            new_lines.append(new_step_line)
            handled_step = True
            modified = True
            continue

        # Replace existing expert_time_source
        if not handled_source and stripped.startswith("expert_time_source:"):
            new_lines.append(new_source_line)
            handled_source = True
            modified = True
            continue

        if stripped.startswith("task_family:"):
            insert_after_task_family = len(new_lines)

        new_lines.append(line)

    # Insert missing fields after task_family
    if not handled_step or not handled_source:
        if insert_after_task_family < 0:
            for i, line in enumerate(new_lines):
                if line.strip() == "metrics:":
                    insert_after_task_family = i + 1
                    break
        if insert_after_task_family < 0:
            raise SystemExit(f"ERROR: cannot locate insertion point in {scene_path}")

        insert_lines = []
        if not handled_step:
            insert_lines.append(new_step_line)
        if not handled_source:
            insert_lines.append(new_source_line)

        idx = insert_after_task_family + 1
        new_lines[idx:idx] = insert_lines
        modified = True

    if modified and not dry_run:
        _write_yaml_lines(scene_path, new_lines)

    return modified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fill expert_time_step in scene YAML from replay data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "dataset_dir",
        nargs="?",
        help="Path to replay directory (e.g. .../dataset/<task>/replay-generalization)",
    )
    p.add_argument(
        "--mode", choices=("evaluate", "stored"), default="evaluate",
        help="How to determine success: 'evaluate' = run evaluator on trajectories (default, needs --task); "
             "'stored' = read existing meta/success from HDF5",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Python module path for task evaluator (required for --mode evaluate). "
             "e.g. success.custom.task_51_toilet_lid_cleaner_pour",
    )
    p.add_argument(
        "--scene",
        default=None,
        help="Explicit path to the target scene YAML (auto-detected from dataset path by default)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print what would change without writing"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process ALL scenes whose expert_time_step is null / unset",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Root directory under which to search for dataset/<task>/replay-generalization (used with --all)",
    )
    p.add_argument(
        "--split",
        default="replay-generalization",
        help="Dataset split name (default: replay-generalization)",
    )
    return p


def _find_null_scenes() -> List[Path]:
    """Return all scene YAMLs whose metrics section lacks expert_time_step."""
    null_scenes: List[Path] = []
    for yaml_path in sorted(SCENES_DIR.glob("*.yaml")):
        if yaml_path.name.startswith("00_template"):
            continue
        text = yaml_path.read_text()
        has_expert_time_step = "expert_time_step:" in text
        if not has_expert_time_step:
            null_scenes.append(yaml_path)
    return null_scenes


def main() -> None:
    args = _build_parser().parse_args()

    if args.all:
        null_scenes = _find_null_scenes()
        if not null_scenes:
            print("All scene YAMLs already have expert_time_step. Nothing to do.")
            return

        print(f"Found {len(null_scenes)} scenes without expert_time_step:")
        for s in null_scenes:
            print(f"  {s.name}")

        # Try to find datasets for each
        success_count = 0
        for scene_path in null_scenes:
            task_name = scene_path.stem  # e.g. "60_breadbasket_fast_food_loading"

            # Search common dataset roots
            dataset_found = None
            search_roots = []
            if args.dataset_root:
                search_roots.append(args.dataset_root)
            search_roots.extend([
                "../teleopdata_and_ckpt",
                "../teleopdata",
            ])
            for root in search_roots:
                candidate = os.path.join(root, "dataset", task_name, args.split)
                if os.path.isdir(candidate):
                    dataset_found = candidate
                    break

            if dataset_found is None:
                print(f"  SKIP {task_name}: no dataset found in searched roots")
                continue

            total, n_ok, mean_steps, median, _ = compute_stats(dataset_found)
            if n_ok == 0:
                print(f"  SKIP {task_name}: 0/{total} successful episodes")
                continue

            expert_step = int(round(mean_steps))
            print(
                f"  {task_name}: {n_ok}/{total} success, "
                f"mean_steps={mean_steps:.1f} -> expert_time_step={expert_step}"
            )

            modified = update_scene_yaml(
                scene_path, expert_step, n_ok, total, args.split,
                dry_run=args.dry_run,
            )
            if modified:
                success_count += 1

        print(f"\nUpdated {success_count}/{len(null_scenes)} scene YAMLs.")
        return

    # Validate mode-specific requirements
    if args.mode == "evaluate" and not args.task and not args.all:
        raise SystemExit("ERROR: --task is required for --mode evaluate. "
                         "Use --mode stored to read pre-existing labels instead.")

    # Single-task mode
    if not args.dataset_dir:
        raise SystemExit("ERROR: either provide dataset_dir or use --all")

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(f"ERROR: dataset_dir not found: {args.dataset_dir}")

    split_name = os.path.basename(args.dataset_dir)
    scene_path = resolve_scene_yaml(args.dataset_dir, args.scene)

    if args.mode == "evaluate":
        print(f"Mode: evaluate (running {args.task} on trajectories)")
        # Auto-read params from scene YAML's success_conditions
        task_params = None
        try:
            import yaml as _yaml
            with open(scene_path) as _f:
                _d = _yaml.safe_load(_f)
            _sc = _d.get("success_conditions", [])
            if _sc and isinstance(_sc[0], dict) and _sc[0].get("params"):
                task_params = dict(_sc[0]["params"])
        except Exception:
            pass
        total, n_ok, mean_steps, median, all_steps = compute_stats_evaluate(
            args.dataset_dir, args.task, task_params=task_params,
        )
    else:
        print("Mode: stored (reading existing meta/success)")
        total, n_ok, mean_steps, median, all_steps = compute_stats_stored(args.dataset_dir)

    print(f"Dataset: {args.dataset_dir}")
    print(f"  Episodes: {total} total")
    print(f"  Successful: {n_ok}/{total}")
    if n_ok > 0:
        print(f"  Mean steps_to_stable_success:  {mean_steps:.1f}")
        print(f"  Median steps_to_stable_success: {median:.1f}")
        print(f"  Min/Max: {min(all_steps)} / {max(all_steps)}")
        expert_step = int(round(mean_steps))
        print(f"\n  -> expert_time_step = {expert_step}")
        print(f"  -> expert_time @1/60s  = {expert_step / 60.0:.3f}s")
        print(f"  -> 1.5x policy steps    = {int(expert_step * 1.5 / 3)}")
    else:
        print("  WARNING: no successful episodes found!")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would update: {scene_path}")
        return

    modified = update_scene_yaml(
        scene_path, expert_step, n_ok, total, split_name,
        dry_run=False,
    )

    if modified:
        print(f"\nUpdated: {scene_path}")
    else:
        print(f"\nNo changes needed: {scene_path}")


if __name__ == "__main__":
    main()
