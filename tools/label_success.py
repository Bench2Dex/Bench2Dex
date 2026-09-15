"""Offline success labeler.

Reads HDF5 episodes that were collected without a running success evaluator
(meta/success_available=False) and retroactively computes success labels by
replaying the saved object-pose trajectories through the task's check_success()
function. Results are written back in-place to each HDF5 file.

Usage
-----
conda activate manus
python tools/label_success.py \
    --hdf5_dir outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading/replay \
    --task success.custom.task_06_fruit_bowl_loading

Optional flags
--------------
--force         Re-label even if meta/success_available is already True
--dry_run       Print results without modifying HDF5 files
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import h5py
import numpy as np

# Make sure the dex2bench root is on sys.path so success.* imports work
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_states_at_frame(f: h5py.File, obj_ids: list[str], frame_idx: int) -> dict:
    """Build a states dict for one frame from recorded object trajectories."""
    states: dict = {}
    for obj_id in obj_ids:
        group_key = f"objects/{obj_id}"
        pose_key = f"{group_key}/pose_world"
        if pose_key not in f:
            continue
        state = {"pose_world": f[pose_key][frame_idx]}
        for vel_key in ("lin_vel_world", "ang_vel_world"):
            full_key = f"{group_key}/{vel_key}"
            if full_key in f:
                state[vel_key] = f[full_key][frame_idx]
        # Load articulation joint state (needed for joint_state condition checks)
        qpos_key = f"{group_key}/qpos"
        if qpos_key in f:
            qpos_array = f[qpos_key][frame_idx]
            joint_names_key = f"{group_key}/joint_names"
            if joint_names_key in f:
                joint_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f[joint_names_key][:]]
                state["qpos"] = {name: float(qpos_array[i]) for i, name in enumerate(joint_names) if i < len(qpos_array)}
                state["joint_names"] = joint_names
            else:
                state["qpos"] = {f"joint_{i}": float(v) for i, v in enumerate(qpos_array)}
        states[obj_id] = state
    return states


def label_episode(
    hdf5_path: str,
    check_success_fn,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Return True if success was found (stable, after dwell) in this episode."""
    with h5py.File(hdf5_path, "r" if dry_run else "a") as f:
        # Skip if already labeled (unless --force)
        success_available = bool(f["meta/success_available"][()]) if "meta/success_available" in f else False
        if success_available and not force:
            return bool(f["meta/success"][()])

        frame_count = int(f["meta/frame_count"][()])
        obj_ids = list(f["objects"].keys())
        step_stride = int(f["meta/step_stride"][()]) if "meta/step_stride" in f else 3
        fps = float(f["meta/effective_fps"][()]) if "meta/effective_fps" in f else 20.0

        # Dwell: success must hold for N consecutive frames
        dwell_s = 0.5
        dwell_frames = max(1, int(round(dwell_s * fps)))

        ctx: dict = {}
        success = False
        stable_success = False
        steps_to_stable = None
        first_success_frame = None
        consecutive = 0

        for t in range(frame_count):
            states = _load_states_at_frame(f, obj_ids, t)
            frame_ok = check_success_fn(states, ctx, {})
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

        if not dry_run:
            str_dtype = h5py.string_dtype(encoding="utf-8")
            for key, value, dtype in [
                ("success", success, np.bool_),
                ("success_available", True, np.bool_),
            ]:
                full_key = f"meta/{key}"
                if full_key in f:
                    f[full_key][()] = np.asarray(value, dtype=dtype)
                else:
                    f["meta"].create_dataset(key, data=np.asarray(value, dtype=dtype))

            # Write stable_success and steps_to_stable_success to metrics/episode
            metrics_ep = f.require_group("metrics").require_group("episode")
            str_dtype = h5py.string_dtype(encoding="utf-8")
            for key, value, is_bool in [
                ("stable_success", stable_success, True),
            ]:
                if key in metrics_ep:
                    del metrics_ep[key]
                metrics_ep.create_dataset(key, data=np.asarray(value, dtype=np.bool_))

            # steps_to_stable_success: physics steps when stable success first achieved
            sts_key = "steps_to_stable_success"
            if sts_key in metrics_ep:
                del metrics_ep[sts_key]
            if steps_to_stable is not None:
                metrics_ep.create_dataset(sts_key, data=np.asarray(int(steps_to_stable), dtype=np.int64))
            else:
                metrics_ep.create_dataset(sts_key, data=np.asarray("null", dtype=str_dtype))

            # first_success_step: policy frame index of first success
            fss_key = "first_success_step"
            if fss_key in metrics_ep:
                del metrics_ep[fss_key]
            if first_success_frame is not None:
                metrics_ep.create_dataset(fss_key, data=np.asarray(int(first_success_frame), dtype=np.int64))
            else:
                metrics_ep.create_dataset(fss_key, data=np.asarray("null", dtype=str_dtype))

        return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline success labeler for dex2bench HDF5 episodes")
    parser.add_argument("--hdf5_dir", required=True, help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--task", required=True,
                        help="Python module path for the task evaluator, e.g. "
                             "success.custom.task_06_fruit_bowl_loading")
    parser.add_argument("--force", action="store_true",
                        help="Re-label even if success_available is already True")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print results without modifying HDF5 files")
    args = parser.parse_args()

    # Import the task module and get check_success
    mod = importlib.import_module(args.task)
    check_success_fn = mod.check_success

    files = sorted(
        os.path.join(args.hdf5_dir, f)
        for f in os.listdir(args.hdf5_dir)
        if f.endswith(".hdf5")
    )
    if not files:
        print(f"[label_success] No HDF5 files found in {args.hdf5_dir}")
        return

    total = len(files)
    n_success = 0
    for i, path in enumerate(files):
        result = label_episode(path, check_success_fn, force=args.force, dry_run=args.dry_run)
        n_success += int(result)
        tag = "✓" if result else "✗"
        print(f"[{i+1:3d}/{total}] {tag}  {os.path.basename(path)}")

    mode = "[dry-run] " if args.dry_run else ""
    print(f"\n{mode}Results: {n_success}/{total} episodes labeled as success "
          f"({100*n_success/total:.1f}%)")


if __name__ == "__main__":
    main()
