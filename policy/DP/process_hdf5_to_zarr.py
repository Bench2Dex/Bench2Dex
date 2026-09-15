#!/usr/bin/env python3
"""Convert dex2scene HDF5 replay episodes into a preprocessed Zarr store.

One-time preprocessing — decodes all JPEG camera images, applies active-DOF
selection, and writes a single Zarr directory that the Dex2SceneZarrDataset
reads with zero decode overhead at training time.

Usage:
    python process_hdf5_to_zarr.py \\
        --dataset_dir /path/to/replay-generalization \\
        --output /path/to/output.zarr \\
        [--max_episodes 10] [--image_height 480] [--image_width 640]

The output Zarr has:
    data/
      state          float32  (total_steps, active_dof)
      action         float32  (total_steps, active_dof)
      right_cam      uint8    (total_steps, 3, H, W)
      left_cam       uint8    (total_steps, 3, H, W)
      stereo_left_cam  uint8  (total_steps, 3, H, W)
      stereo_right_cam uint8  (total_steps, 3, H, W)
    meta/
      episode_ends   int64    (n_episodes,)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# resolve package root so we can import from diffusion_policy and dex2bench
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_DEX2BENCH_ROOT = _SCRIPT_DIR.parent.parent
if str(_DEX2BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEX2BENCH_ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.dex2scene_hdf5_dataset import (
    DEFAULT_CAMERA_OBS_MAP,
    _decode_rgb,
    _resize_rgb,
    _fill_nan_action_rows,
)
from robots.active_dof_utils import (
    get_active_dof_info_for_hdf5,
    select_active,
)


def _find_homing_cutoff(
    hdf5_path: str,
    ep_len: int,
    *,
    truncate: bool = True,
    min_frames: int = 1,
) -> Tuple[Optional[int], str]:
    """Determine the frame index at which to truncate an episode.

    Reads ``meta/homing_start_sim_step`` and ``time/sim_step`` from the
    HDF5 file, following the same pattern as ACT's
    ``_resolve_effective_length`` (``policy/ACT/utils.py``).  The
    ``homing_start_sim_step`` is the sim-step at which the operator
    triggered "go home" — a reliable human label that the task is complete
    and subsequent frames are non-task noise.

    Args:
        hdf5_path: Path to the episode HDF5 file.
        ep_len: Total number of observation frames (length of ``time/sim_step``).
        truncate: If False, immediately returns ``(None, info)`` — no
            truncation is applied.
        min_frames: Minimum number of frames to keep after truncation.

    Returns:
        ``(cutoff, info)`` where:
          * ``cutoff`` — int frame count for the truncated episode (same
            semantics as ``ep_len``, i.e. ``[:cutoff]``).  The homing frame
            itself is **excluded**.  ``None`` if no truncation should be
            applied.
          * ``info`` — str describing what happened (for log output).
    """
    if not truncate:
        return None, "truncation disabled"

    with h5py.File(hdf5_path, "r") as f:
        # -- homing_start_sim_step ---------------------------------------------
        homing_start = f.get("meta/homing_start_sim_step")
        if homing_start is None:
            return None, "homing_start_sim_step missing — keeping full episode"

        homing_val = int(homing_start[()])
        if homing_val < 0:
            return None, f"homing_start_sim_step={homing_val} (no homing) — keeping full episode"

        # -- map sim step → frame index ---------------------------------------
        time_sim_step = f["time/sim_step"][:]  # (ep_len,) int64
        # searchsorted returns first i where time_sim_step[i] >= homing_val
        frame_idx = int(np.searchsorted(time_sim_step, homing_val))

        # Following ACT: require 0 < idx < total  (exclude the homing frame)
        if not (min_frames <= frame_idx < ep_len):
            return None, (
                f"homing at frame {frame_idx} out of valid range "
                f"[{min_frames}, {ep_len}) — keeping full episode"
            )

        cutoff = frame_idx  # exclusive: frames [:frame_idx] kept
        cutoff = max(cutoff, min_frames)
        cutoff = min(cutoff, ep_len)

        frames_removed = ep_len - cutoff
        info = (
            f"homing at frame {frame_idx} (sim_step={homing_val}), "
            f"cutoff={cutoff}, removed={frames_removed}"
        )
        return cutoff, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HDF5 → Zarr conversion for dex2scene DP")
    p.add_argument("--dataset_dir", required=True, help="Directory with episode_*.hdf5 files")
    p.add_argument("--output", required=True, help="Output .zarr path")
    p.add_argument("--image_height", type=int, default=480)
    p.add_argument("--image_width", type=int, default=640)
    p.add_argument("--max_episodes", type=int, default=None, help="Cap number of episodes (for testing)")
    p.add_argument("--camera_obs_map", type=str, default=None,
                   help="JSON string of {hdf5_cam_id: obs_key}, e.g. '{\"cam_wrist_right\":\"right_cam\"}'")
    p.add_argument("--no_active_dof", action="store_true", help="Disable active-DOF selection")
    p.add_argument("--truncate_at_homing", action="store_true", default=False,
                   help="Truncate each episode at the homing-start frame (exclusive)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        sys.exit(f"dataset_dir does not exist: {dataset_dir}")

    hdf5_paths = sorted(dataset_dir.glob("episode_*.hdf5"))
    if not hdf5_paths:
        sys.exit(f"No episode_*.hdf5 files found in {dataset_dir}")

    if args.max_episodes is not None and args.max_episodes > 0:
        hdf5_paths = hdf5_paths[: int(args.max_episodes)]

    use_active_dof = not args.no_active_dof
    image_shape = (3, int(args.image_height), int(args.image_width))

    # Resolve camera mapping
    import json
    camera_obs_map = DEFAULT_CAMERA_OBS_MAP
    if args.camera_obs_map is not None:
        camera_obs_map = json.loads(args.camera_obs_map)

    # Detect active-DOF info from first episode
    adi = None
    if use_active_dof:
        adi = get_active_dof_info_for_hdf5(str(hdf5_paths[0]))
        if adi is not None:
            print(f"[auto] robot_key={adi.robot_key}  active_dof={adi.active_dof}")
        else:
            print("[auto] could not detect active DOF — storing full dof")

    # Discover available cameras from first episode
    with h5py.File(str(hdf5_paths[0]), "r") as f:
        available_cams = set(f["cameras"].keys())
    cam_ids = [cid for cid in camera_obs_map if cid in available_cams]
    obs_keys = [camera_obs_map[cid] for cid in cam_ids]
    missing = set(camera_obs_map) - available_cams
    if missing:
        print(f"[warn] configured cameras missing in data: {missing}")
    print(f"cameras: {cam_ids} → {obs_keys}")
    print(f"image_shape: {image_shape}")
    print(f"episodes: {len(hdf5_paths)}")

    # ------------------------------------------------------------------
    # Process episodes directly into a disk-backed Zarr to avoid OOM.
    # 100 episodes × ~800 frames × 4 cameras × 480×640×3 ≈ 280 GB
    # uncompressed — far too large for in-memory numpy accumulation.
    # Zarr writes each episode to disk incrementally.
    # ------------------------------------------------------------------
    import zarr
    output_path = Path(args.output)
    if output_path.exists():
        import shutil
        shutil.rmtree(str(output_path))
    store = zarr.DirectoryStore(str(output_path))
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    total_steps = 0

    for ep_idx, path in enumerate(hdf5_paths):
        with h5py.File(str(path), "r") as f:
            ep_len = int(f["robot/qpos"].shape[0])

            # -- Determine truncation cutoff ----------------------------------
            cutoff, trunc_info = _find_homing_cutoff(
                str(path), ep_len,
                truncate=args.truncate_at_homing,
            )

            read_len = cutoff if cutoff is not None else ep_len

            qpos = f["robot/qpos"][:read_len].astype(np.float32)
            action = f["action/commanded"][:read_len].astype(np.float32)
            action = _fill_nan_action_rows(action, qpos)

            if adi is not None:
                qpos = select_active(qpos, adi)
                action = select_active(action, adi)

            # Decode camera frames  (T, 3, H, W) uint8
            cam_frames: dict[str, np.ndarray] = {}
            for cid, okey in zip(cam_ids, obs_keys):
                rgb_ds = f[f"cameras/{cid}/rgb"]
                frames = []
                for t in range(read_len):
                    img = _decode_rgb(rgb_ds[t])
                    img = _resize_rgb(img, height=image_shape[1], width=image_shape[2])
                    # HWC uint8 → CHW uint8
                    frames.append(np.moveaxis(img.astype(np.uint8), -1, 0))
                cam_frames[okey] = np.stack(frames, axis=0)  # (T, 3, H, W)

        # Build episode data dict  (all values (T, ...))
        episode_data: dict[str, np.ndarray] = {
            "state": qpos,
            "action": action,
        }
        episode_data.update(cam_frames)

        rb.add_episode(episode_data, compressors="disk")
        total_steps += read_len

        print(f"  episode {ep_idx + 1}/{len(hdf5_paths)}  "
              f"({read_len} steps)  total_steps={total_steps}  [{trunc_info}]")

    print(f"\nDone.  Zarr saved to {output_path}  "
          f"({total_steps} total steps, {rb.n_episodes} episodes)")


if __name__ == "__main__":
    main()
