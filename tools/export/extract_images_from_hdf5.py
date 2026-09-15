#!/usr/bin/env python3
"""Extract RGB and Depth images from replay HDF5 to PNG files.

For tactile TacMap streams, use tools/vis/visualize_tacmap_hdf5.py.

Usage:
    # Basic usage with defaults:
    python tools/extract_images_from_hdf5.py \
        --hdf5 /path/to/episode_replay.hdf5

    # Custom output directory and frame count:
    python tools/extract_images_from_hdf5.py \
        --hdf5 /path/to/episode_replay.hdf5 \
        --output-dir /path/to/output_images \
        --num-frames 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


def decode_hdf5_string(value: Any) -> str:
    """Decode HDF5 string to Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_hdf5_string(value.item())
    return str(value)


def select_frames(frame_count: int, num_frames: int) -> list[int]:
    """Uniformly sample frame indices."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    num = min(max(int(num_frames), 1), int(frame_count))
    frames = np.linspace(0, frame_count - 1, num=num, dtype=np.int64)
    return sorted({int(frame) for frame in frames})


def extract_images(hdf5_path: Path, output_dir: Path, num_frames: int = 8) -> dict[str, Any]:
    """Extract RGB and Depth images from replay HDF5 to PNG files.

    Args:
        hdf5_path: Path to replay HDF5 file containing /cameras group
        output_dir: Directory to save extracted images
        num_frames: Number of frames to uniformly sample from each camera

    Returns:
        Summary dictionary with extraction metadata
    """
    hdf5_path = hdf5_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "hdf5": str(hdf5_path),
        "output_dir": str(output_dir),
        "num_frames_requested": num_frames,
        "cameras": {},
    }

    with h5py.File(hdf5_path, "r") as f:
        if "cameras" not in f:
            raise ValueError("No /cameras group in HDF5. Is this a replay file?")

        # Read metadata if available
        meta = f.get("meta")
        if meta is not None:
            if "fps" in meta:
                summary["fps"] = int(meta["fps"][()])
            if "frame_count" in meta:
                summary["total_frames"] = int(meta["frame_count"][()])

        cameras_group = f["cameras"]
        camera_ids = sorted(cameras_group.keys())

        print(f"[extract] Found cameras: {camera_ids}")

        for cam_id in camera_ids:
            cam_group = cameras_group[cam_id]
            cam_out_dir = output_dir / cam_id
            cam_out_dir.mkdir(exist_ok=True)

            cam_info: dict[str, Any] = {"frames": []}

            # Extract RGB
            if "rgb" in cam_group:
                rgb_dataset = cam_group["rgb"]
                rgb_data = np.asarray(rgb_dataset[:])  # (T, H, W, 3)
                T, H, W, C = rgb_data.shape

                selected = select_frames(T, num_frames)

                for idx in selected:
                    rgb_frame = rgb_data[idx]
                    # Ensure uint8 format
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)

                    img_path = cam_out_dir / f"frame_{idx:06d}_rgb.png"
                    Image.fromarray(rgb_frame).save(img_path)
                    cam_info["frames"].append({
                        "frame_idx": int(idx),
                        "rgb_path": str(img_path.relative_to(output_dir)),
                    })

                cam_info["rgb_shape"] = [H, W, C]
                cam_info["rgb_frames_saved"] = len(selected)
                print(f"[{cam_id}] RGB: {len(selected)} frames saved, shape={H}x{W}x{C}")

            # Extract Depth (visualization + raw)
            if "depth" in cam_group:
                depth_dataset = cam_group["depth"]
                depth_data = np.asarray(depth_dataset[:])  # (T, H, W)
                T, H, W = depth_data.shape

                selected = select_frames(T, num_frames)

                depth_mins = []
                depth_maxs = []

                for idx in selected:
                    depth_frame = depth_data[idx]

                    # Compute min/max for this frame (excluding NaN/Inf)
                    finite_depth = depth_frame[np.isfinite(depth_frame)]
                    if len(finite_depth) > 0:
                        d_min, d_max = float(np.min(finite_depth)), float(np.max(finite_depth))
                    else:
                        d_min, d_max = 0.0, 1.0
                    depth_mins.append(d_min)
                    depth_maxs.append(d_max)

                    # 1. Visualization version (grayscale, normalized per-frame)
                    if d_max > d_min:
                        depth_norm = ((depth_frame - d_min) / (d_max - d_min) * 255)
                    else:
                        depth_norm = np.zeros_like(depth_frame)
                    depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)

                    vis_path = cam_out_dir / f"frame_{idx:06d}_depth_vis.png"
                    Image.fromarray(depth_norm).save(vis_path)

                    # 2. Raw version (16-bit PNG, preserves depth in meters with 0.1mm precision)
                    # Clamp to max 6.55m to fit in uint16 with 0.1mm scaling
                    depth_mm = (np.clip(depth_frame, 0, 6.5535) * 10000).astype(np.uint16)
                    raw_path = cam_out_dir / f"frame_{idx:06d}_depth_raw.png"
                    Image.fromarray(depth_mm).save(raw_path)

                cam_info["depth_shape"] = [H, W]
                cam_info["depth_frames_saved"] = len(selected)
                cam_info["depth_range_m"] = {
                    "per_frame_min": depth_mins,
                    "per_frame_max": depth_maxs,
                    "global_min": float(min(depth_mins)),
                    "global_max": float(max(depth_maxs)),
                }
                print(f"[{cam_id}] Depth: {len(selected)} frames saved, shape={H}x{W}")

            # Extract camera parameters if available
            if "intrinsic" in cam_group:
                intrinsic = np.asarray(cam_group["intrinsic"][:])
                cam_info["intrinsic"] = intrinsic.tolist()

            if "extrinsic_world_from_cam" in cam_group:
                extrinsic = np.asarray(cam_group["extrinsic_world_from_cam"][:])
                # Save first frame's extrinsic as reference
                if extrinsic.ndim == 3:  # (T, 4, 4)
                    cam_info["extrinsic_world_from_cam_first_frame"] = extrinsic[0].tolist()
                else:  # (4, 4) constant
                    cam_info["extrinsic_world_from_cam"] = extrinsic.tolist()

            summary["cameras"][cam_id] = cam_info

    # Write summary JSON
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[extract] Summary written: {summary_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract RGB and Depth images from replay HDF5 to PNG files."
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        required=True,
        help="Path to replay HDF5 file containing /cameras group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./extracted_images"),
        help="Directory to save extracted images. Default: ./extracted_images",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to uniformly sample from each camera. Default: 8",
    )
    args = parser.parse_args()

    try:
        extract_images(args.hdf5, args.output_dir, args.num_frames)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
