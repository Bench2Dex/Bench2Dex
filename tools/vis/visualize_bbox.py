#!/usr/bin/env python3
"""Visualize box3d and box2d labels overlaid on camera RGB images.

Reads 3D bounding boxes from HDF5 episode files, projects 3D box wireframes
and draws 2D bounding boxes onto each camera view, and saves multi-view tiled
PNG images.  No Isaac Sim dependency -- runs purely offline.

Usage:
    python tools/labels/visualize_bbox.py <episode.hdf5> --frame 0
    python tools/labels/visualize_bbox.py <episode.hdf5> --frame-range 0:5
    python tools/labels/visualize_bbox.py <episode.hdf5> --frame-range all --output-dir out/
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.camera_geometry import (
    CameraFrame,
    project_world_points_to_image_dispatch,
    quat_xyzw_to_rot,
)
from tools.labels._label_common import (
    find_hdf5_files,
    read_camera_frames,
)

# 12 edges of a 3D box (indices into 8 corners)
_BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]

# Color palette for objects (tab10)
_COLORS = plt.cm.tab10.colors


def _compute_box_corners(center: np.ndarray, size: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Compute 8 corners of a 3D box in world frame.

    Args:
        center: [3] center position.
        size: [3] full size (length, width, height).
        quat_xyzw: [4] quaternion (x, y, z, w).

    Returns:
        [8, 3] corner positions.
    """
    half = size / 2.0
    # 8 corners in local frame
    signs = np.array([
        [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
        [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
    ], dtype=np.float32)
    corners_local = signs * half
    R = quat_xyzw_to_rot(quat_xyzw)
    corners_world = (corners_local @ R.T) + center
    return corners_world.astype(np.float32)


def _project_edges(corners_world: np.ndarray, cam_frame: CameraFrame):
    """Project 3D box edges to 2D image coordinates.

    For fisheye cameras, each edge is subdivided into small segments so that
    the projected line follows the lens distortion curve instead of being a
    straight line between the two corner projections.

    Returns list of np.ndarray (N,2) polylines for visible edges.
    """
    H, W = cam_frame.image_shape
    is_fisheye = getattr(cam_frame, "camera_model", "pinhole") == "fisheye"
    n_subdiv = 20 if is_fisheye else 1  # subdivisions per edge

    polylines: list[np.ndarray] = []
    for i, j in _BOX_EDGES:
        # Interpolate 3D points along the edge
        t = np.linspace(0.0, 1.0, n_subdiv + 1, dtype=np.float32)
        pts_3d = corners_world[i][None, :] * (1 - t[:, None]) + corners_world[j][None, :] * t[:, None]

        uv, valid = project_world_points_to_image_dispatch(pts_3d, cam_frame)

        # Apply fisheye valid mask
        fmask = getattr(cam_frame, "fisheye_valid_mask", None)
        if fmask is not None:
            u_int = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W - 1)
            v_int = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H - 1)
            valid = valid & fmask[v_int, u_int]

        # Bounds check
        in_bounds = valid & (uv[:, 0] >= -W * 0.5) & (uv[:, 0] < W * 1.5) & \
                    (uv[:, 1] >= -H * 0.5) & (uv[:, 1] < H * 1.5)

        if not np.any(in_bounds):
            continue

        # Collect contiguous valid segments
        uv_valid = uv.copy()
        uv_valid[~in_bounds] = np.nan
        polylines.append(uv_valid)

    return polylines


def render_bbox_frame(
    episode_path: str,
    frame_idx: int,
    output_path: str,
    *,
    exclude_cameras: set[str] | None = None,
) -> None:
    """Render box3d wireframes and box2d rectangles on multi-view tiled image."""
    with h5py.File(episode_path, "r") as f:
        cam_frames = read_camera_frames(f, frame_idx, need_images=True)
        if exclude_cameras:
            cam_frames = {k: v for k, v in cam_frames.items() if k not in exclude_cameras}

        # Read box3d labels
        box3d_grp = f.get("labels/box3d")
        box3d_data: dict[str, dict] = {}
        if box3d_grp:
            for oid in box3d_grp.keys():
                og = box3d_grp[oid]
                c = np.asarray(og["center_world"][frame_idx], dtype=np.float32)
                s = np.asarray(og["size_lwh"][frame_idx], dtype=np.float32)
                q = np.asarray(og["quat_world"][frame_idx], dtype=np.float32)
                if np.all(np.isfinite(c)) and np.all(np.isfinite(s)) and np.all(s > 0):
                    box3d_data[oid] = {"center": c, "size": s, "quat": q}

        # Read box2d labels
        box2d_grp = f.get("labels/box2d")
        box2d_data: dict[str, dict[str, dict]] = {}
        if box2d_grp:
            for cam_id in box2d_grp.keys():
                box2d_data[cam_id] = {}
                for oid in box2d_grp[cam_id].keys():
                    og = box2d_grp[cam_id][oid]
                    xyxy = np.asarray(og["xyxy"][frame_idx], dtype=np.int32)
                    vis = bool(og["visible"][frame_idx])
                    if vis and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                        box2d_data[cam_id][oid] = {"xyxy": xyxy}

    if not cam_frames:
        print(f"  Frame {frame_idx}: no cameras, skipping.")
        return

    # Assign colors to objects
    obj_ids = sorted(box3d_data.keys())
    obj_colors = {oid: _COLORS[i % len(_COLORS)] for i, oid in enumerate(obj_ids)}

    n_cams = len(cam_frames)
    n_cols = min(n_cams, 3)
    n_rows = math.ceil(n_cams / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    cam_ids = sorted(cam_frames.keys())
    for idx, cam_id in enumerate(cam_ids):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        cf = cam_frames[cam_id]
        H, W = cf.image_shape

        # Background RGB
        if cf.rgb is not None:
            ax.imshow(cf.rgb)
        else:
            ax.imshow(np.full((H, W, 3), 180, dtype=np.uint8))

        # Draw box3d wireframes (projected)
        for oid, bd in box3d_data.items():
            corners = _compute_box_corners(bd["center"], bd["size"], bd["quat"])
            polylines = _project_edges(corners, cf)
            color = obj_colors[oid]
            for poly in polylines:
                ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.0, alpha=0.8)

        # Draw box2d rectangles
        cam_box2d = box2d_data.get(cam_id, {})
        for oid, bd in cam_box2d.items():
            color = obj_colors.get(oid, (1, 1, 1))
            x1, y1, x2, y2 = bd["xyxy"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, oid.replace("obj_", ""), fontsize=6,
                    color=color, backgroundcolor=(0, 0, 0, 0.5),
                    verticalalignment="bottom")

        ax.set_title(f"{cam_id} ({cf.camera_model})", fontsize=9)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    # Hide unused
    for j in range(n_cams, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)

    n_obj = len(box3d_data)
    fig.suptitle(f"Frame {frame_idx}  |  {n_obj} objects  |  box3d wireframe + box2d dashed", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_frame_spec(spec: str, total_frames: int) -> list[int]:
    if spec == "all":
        return list(range(total_frames))
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_frames
        return list(range(start, min(end, total_frames)))
    return [int(spec)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize box3d/box2d labels on camera images.")
    parser.add_argument("episode", help="Path to episode HDF5 file or directory")
    parser.add_argument("--frame", default=None, help="Frame index (default: 0)")
    parser.add_argument("--frame-range", default=None, help='Frame range, e.g. "0:5" or "all"')
    parser.add_argument("--output-dir", default=None, help="Output directory for PNGs")
    parser.add_argument(
        "--exclude-cameras", type=str, default="",
        help="Comma-separated camera IDs to skip (default: none)",
    )
    args = parser.parse_args()

    exclude_cameras = set(s.strip() for s in args.exclude_cameras.split(",") if s.strip()) if args.exclude_cameras else set()

    episodes = find_hdf5_files(args.episode)
    if not episodes:
        print(f"Error: no HDF5 files found at {args.episode}")
        sys.exit(1)

    for ep_path in episodes:
        print(f"\n=== {ep_path} ===")
        with h5py.File(ep_path, "r") as f:
            total_frames = int(f["frame_valid"].shape[0]) if "frame_valid" in f else 0

        if args.frame_range is not None:
            frames = parse_frame_spec(args.frame_range, total_frames)
        elif args.frame is not None:
            frames = parse_frame_spec(args.frame, total_frames)
        else:
            frames = [0]

        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = str(Path(ep_path).parent / "vis_bbox")
        os.makedirs(out_dir, exist_ok=True)

        for fi in frames:
            if fi < 0 or fi >= total_frames:
                print(f"  Frame {fi}: out of range, skipping.")
                continue
            out_path = os.path.join(out_dir, f"frame_{fi:04d}.png")
            print(f"  Rendering frame {fi}/{total_frames - 1} -> {out_path}")
            render_bbox_frame(ep_path, fi, out_path, exclude_cameras=exclude_cameras)

        print(f"  Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
