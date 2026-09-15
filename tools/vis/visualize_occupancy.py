#!/usr/bin/env python3
"""Visualize occupancy labels overlaid on camera RGB images.

Reads occupancy voxel grids from HDF5 episode files (inline or sidecar),
projects OCCUPIED voxels onto each camera view, and saves images (one PNG per
camera view by default, or a single tiled image with ``--tiled``).  No Isaac
Sim dependency -- runs purely offline.

Usage:
    # One PNG per camera view (default):
    python tools/vis/visualize_occupancy.py <episode.hdf5> --frame 0

    # Include wrist cameras (pass empty --exclude-cameras):
    python tools/vis/visualize_occupancy.py <episode.hdf5> --frame 0 --exclude-cameras ""

    # Tiled grid of all views in one image:
    python tools/vis/visualize_occupancy.py <episode.hdf5> --frame 0 --tiled

    # Batch:
    python tools/vis/visualize_occupancy.py <episode.hdf5> --frame-range 0:10
    python tools/vis/visualize_occupancy.py <episode.hdf5> --frame-range all
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
    read_camera_frames,
    read_local_bboxes,
    read_object_states,
    find_hdf5_files,
)

# Occupancy state constants (matching collector/occupancy.py)
_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2


# ---------------------------------------------------------------------------
# Occupancy data reading
# ---------------------------------------------------------------------------

def _find_occupancy_group(episode_path: str, label_key: str = "occupancy_tsdf") -> tuple[str, str] | None:
    """Locate occupancy data in the episode or its sidecar.

    Returns (hdf5_path, group_path) or None if not found.
    """
    # Check inline first
    with h5py.File(episode_path, "r") as f:
        if "labels" in f and label_key in f["labels"]:
            return episode_path, f"labels/{label_key}"

    # Check sidecar
    source = Path(episode_path)
    sidecar = source.parent / "derived" / f"{source.stem}_derived.hdf5"
    if sidecar.exists():
        with h5py.File(str(sidecar), "r") as f:
            if "labels" in f and label_key in f["labels"]:
                return str(sidecar), f"labels/{label_key}"

    return None


def read_occupancy(episode_path: str, label_key: str = "occupancy_tsdf") -> dict:
    """Read occupancy grid metadata and state array.

    Returns dict with keys: state, bounds, voxel_size, grid_shape, source_path.
    """
    loc = _find_occupancy_group(episode_path, label_key=label_key)
    if loc is None:
        raise FileNotFoundError(
            f"No '{label_key}' labels found in {episode_path} or its sidecar."
        )
    hdf5_path, grp_path = loc

    with h5py.File(hdf5_path, "r") as f:
        grp = f[grp_path]
        state = np.asarray(grp["state"][:], dtype=np.uint8)
        bounds = np.asarray(grp["bounds"][()], dtype=np.float32)
        voxel_size = float(grp["voxel_size"][()])
        grid_shape = np.asarray(grp["grid_shape"][()], dtype=np.int32)
        method = grp["method"][()].decode() if "method" in grp else None
        semantic_id = np.asarray(grp["semantic_id"][:], dtype=np.uint16) if "semantic_id" in grp else None

    return {
        "state": state,
        "bounds": bounds,
        "voxel_size": voxel_size,
        "grid_shape": grid_shape,
        "source_path": hdf5_path,
        "method": method,
        "semantic_id": semantic_id,
    }


def extract_voxel_points(
    state_frame: np.ndarray,
    bounds: np.ndarray,
    voxel_size: float,
    target_state: int = _OCCUPIED,
) -> np.ndarray:
    """Extract world-coordinate centers of voxels matching *target_state*.

    Args:
        state_frame: uint8 array (nx, ny, nz) for a single frame.
        bounds: float32 (2, 3) — [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        voxel_size: Voxel edge length in meters.
        target_state: Which state to extract (default OCCUPIED=2).

    Returns:
        float32 (N, 3) world-coordinate voxel centers.
    """
    ix, iy, iz = np.where(state_frame == target_state)
    if len(ix) == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = bounds[0, 0] + (ix + 0.5) * voxel_size
    y = bounds[0, 1] + (iy + 0.5) * voxel_size
    z = bounds[0, 2] + (iz + 0.5) * voxel_size
    return np.stack([x, y, z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Object AABB filtering
# ---------------------------------------------------------------------------

_CORNER_SIGNS = np.array([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float32)


def _read_table_z(episode_path: str) -> float | None:
    """Try to read table height from episode metadata."""
    try:
        with h5py.File(episode_path, "r") as f:
            meta = f.get("meta")
            if meta is None or "collect_config" not in meta:
                return None
            import json
            raw = meta["collect_config"][()].decode() if isinstance(meta["collect_config"][()], bytes) else str(meta["collect_config"][()])
            cfg = json.loads(raw)
            # Table height is not in collect_config; try scene metadata
            return None
    except Exception:
        return None


def compute_object_world_aabbs(
    episode_path: str,
    frame_idx: int,
    margin: float = 0.0,
    table_z: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute world-frame AABBs for all objects at a given frame.

    Args:
        table_z: If provided, clamp AABB Z_min to ``table_z + voxel_margin``
            to exclude the table surface layer itself (not just below it).

    Returns list of (aabb_min[3], aabb_max[3]) in world coordinates.
    """
    aabbs: list[tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(episode_path, "r") as f:
        local_bboxes = read_local_bboxes(f)
        if not local_bboxes:
            return aabbs
        object_states = read_object_states(f, frame_idx)

    for obj_id, (lo, hi) in local_bboxes.items():
        state = object_states.get(obj_id)
        if state is None:
            continue
        pose = state["pose_world"]
        lo_arr = np.asarray(lo, dtype=np.float32)
        hi_arr = np.asarray(hi, dtype=np.float32)
        size = hi_arr - lo_arr
        center_local = 0.5 * (lo_arr + hi_arr)
        corners_local = _CORNER_SIGNS * (0.5 * size)

        rot = quat_xyzw_to_rot(pose[3:7])
        center_world = pose[:3] + rot @ center_local
        corners_world = (corners_local @ rot.T) + center_world

        aabb_min = corners_world.min(axis=0) - margin
        aabb_max = corners_world.max(axis=0) + margin
        if table_z is not None:
            aabb_min[2] = max(aabb_min[2], table_z)
        aabbs.append((aabb_min.astype(np.float32), aabb_max.astype(np.float32)))

    return aabbs


def filter_points_by_aabbs(
    points: np.ndarray,
    aabbs: list[tuple[np.ndarray, np.ndarray]],
    table_z: float | None = None,
) -> np.ndarray:
    """Keep only points inside any of the given AABBs."""
    if points.shape[0] == 0 or not aabbs:
        return points
    inside_any = np.zeros(points.shape[0], dtype=np.bool_)
    for aabb_min, aabb_max in aabbs:
        inside = np.all((points >= aabb_min) & (points <= aabb_max), axis=1)
        inside_any |= inside
    return points[inside_any]


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_and_filter(
    points_world: np.ndarray,
    cam_frame: CameraFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image and return valid pixel coords + depths.

    Returns:
        uv_valid: float32 (M, 2) — pixel coordinates of valid projections.
        depth_valid: float32 (M,) — distance from camera center.
    """
    if points_world.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    uv, valid = project_world_points_to_image_dispatch(points_world, cam_frame)
    H, W = cam_frame.image_shape

    # Bounds check
    in_bounds = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < W)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H)
    )

    # Fisheye valid-mask check
    if cam_frame.fisheye_valid_mask is not None and np.any(in_bounds):
        u_int = np.clip(uv[in_bounds, 0].astype(np.int32), 0, W - 1)
        v_int = np.clip(uv[in_bounds, 1].astype(np.int32), 0, H - 1)
        mask_ok = cam_frame.fisheye_valid_mask[v_int, u_int]
        idx = np.where(in_bounds)[0]
        in_bounds[idx[~mask_ok]] = False

    uv_valid = uv[in_bounds]
    # Depth = Euclidean distance to camera center
    depth_valid = np.linalg.norm(
        points_world[in_bounds] - cam_frame.cam_pos_w, axis=1
    ).astype(np.float32)

    return uv_valid, depth_valid


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_single_view(
    ax: plt.Axes,
    cam_id: str,
    cam_frame: CameraFrame,
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    voxel_alpha: float,
    point_size: float,
    colormap: str,
    depth_range: tuple[float, float] | None,
) -> None:
    """Draw one camera view with occupancy overlay on *ax*."""
    H, W = cam_frame.image_shape

    # Background: RGB image or grey placeholder
    if cam_frame.rgb is not None:
        ax.imshow(cam_frame.rgb)
    else:
        ax.imshow(np.full((H, W, 3), 180, dtype=np.uint8))

    if uv.shape[0] > 0:
        # Sort back-to-front so closer voxels are drawn last (on top)
        order = np.argsort(-depth)
        uv = uv[order]
        depth = depth[order]
        vmin, vmax = depth_range if depth_range else (depth.min(), depth.max())
        sc = ax.scatter(
            uv[:, 0],
            uv[:, 1],
            c=depth,
            cmap=colormap,
            s=point_size,
            alpha=voxel_alpha,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="depth (m)")

    ax.set_title(f"{cam_id} ({cam_frame.camera_model})", fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def _render_one_view(
    cam_id: str,
    cam_frame: CameraFrame,
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    voxel_alpha: float,
    point_size: float,
    colormap: str,
    depth_range: tuple[float, float] | None,
    free_pts: np.ndarray | None,
    show_free: bool,
    title: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Render a single camera view as a standalone image — no axes, colorbar inside."""
    fig, ax = plt.subplots(figsize=(10, 8))
    render_single_view(
        ax, cam_id, cam_frame, uv, depth,
        voxel_alpha=voxel_alpha,
        point_size=point_size,
        colormap=colormap,
        depth_range=depth_range,
    )

    # Free voxel overlay
    if show_free and free_pts is not None and free_pts.shape[0] > 0:
        uv_free, _ = project_and_filter(free_pts, cam_frame)
        if uv_free.shape[0] > 0:
            ax.scatter(
                uv_free[:, 0], uv_free[:, 1],
                c="cyan", s=point_size * 0.5,
                alpha=voxel_alpha * 0.3, edgecolors="none", rasterized=True,
            )

    ax.set_title("")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_frame(
    episode_path: str,
    frame_idx: int,
    occ_data: dict,
    output_path: str,
    *,
    voxel_alpha: float = 0.6,
    point_size: float = 1.5,
    colormap: str = "jet",
    show_free: bool = False,
    exclude_cameras: set[str] | None = None,
    object_only: bool = False,
    aabb_margin: float = 0.02,
    table_z: float | None = None,
    label_key: str = "occupancy_tsdf",
    split_views: bool = True,
) -> None:
    """Render occupancy for one frame — one PNG per view (default) or tiled."""
    # Extract occupied voxel world positions
    state_frame = occ_data["state"][frame_idx]
    occupied_pts = extract_voxel_points(
        state_frame, occ_data["bounds"], occ_data["voxel_size"], _OCCUPIED
    )

    # Object-only filtering: prefer semantic_id (excludes table=1),
    # fall back to AABB filtering when semantic_id is unavailable.
    if object_only and occupied_pts.shape[0] > 0:
        sem_data = occ_data.get("semantic_id")
        if sem_data is not None:
            sem_frame = sem_data[frame_idx]
            # Keep voxels with semantic_id >= 2 (objects), exclude 0=free, 1=table
            object_mask = sem_frame >= 2
            occupied_pts = extract_voxel_points(
                np.where(object_mask & (state_frame == _OCCUPIED), _OCCUPIED, 0).astype(np.uint8),
                occ_data["bounds"], occ_data["voxel_size"], _OCCUPIED,
            )
        else:
            aabbs = compute_object_world_aabbs(episode_path, frame_idx, margin=aabb_margin, table_z=table_z)
            if aabbs:
                occupied_pts = filter_points_by_aabbs(occupied_pts, aabbs)

    free_pts = None
    if show_free:
        free_pts = extract_voxel_points(
            state_frame, occ_data["bounds"], occ_data["voxel_size"], _FREE
        )

    # Read camera frames from the episode file (not sidecar — cameras are in episode)
    with h5py.File(episode_path, "r") as f:
        cam_frames = read_camera_frames(f, frame_idx, need_images=True)

    # Exclude cameras
    if exclude_cameras:
        cam_frames = {k: v for k, v in cam_frames.items() if k not in exclude_cameras}

    if not cam_frames:
        print(f"  Frame {frame_idx}: no cameras found, skipping.")
        return

    # Compute global depth range across all cameras for consistent coloring
    cam_projections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_depths: list[np.ndarray] = []
    for cam_id, cam_frame in cam_frames.items():
        uv_occ, depth_occ = project_and_filter(occupied_pts, cam_frame)
        cam_projections[cam_id] = (uv_occ, depth_occ)
        if depth_occ.shape[0] > 0:
            all_depths.append(depth_occ)

    depth_range = None
    if all_depths:
        d_all = np.concatenate(all_depths)
        depth_range = (float(d_all.min()), float(d_all.max()))

    cam_ids = sorted(cam_frames.keys())
    n_occ = occupied_pts.shape[0]
    label_tag = occ_data.get("method") or ("GT mesh" if label_key == "occupancy_gt" else "sensor TSDF")

    out_base, out_ext = os.path.splitext(output_path)

    if split_views:
        # One PNG per camera view
        for cam_id in cam_ids:
            uv_occ, depth_occ = cam_projections[cam_id]
            cam_out = f"{out_base}_{cam_id}{out_ext}"
            _render_one_view(
                cam_id, cam_frames[cam_id], uv_occ, depth_occ,
                voxel_alpha=voxel_alpha,
                point_size=point_size,
                colormap=colormap,
                depth_range=depth_range,
                free_pts=free_pts,
                show_free=show_free,
                title="",
                output_path=cam_out,
            )
        print(f"  → {len(cam_ids)} view(s): {out_base}_<cam_id>{out_ext}")
    else:
        # Tiled grid
        n_cams = len(cam_frames)
        n_cols = min(n_cams, 2)
        n_rows = math.ceil(n_cams / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
        )

        for i, cam_id in enumerate(cam_ids):
            r, c = divmod(i, n_cols)
            ax = axes[r][c]
            uv_occ, depth_occ = cam_projections[cam_id]
            render_single_view(
                ax, cam_id, cam_frames[cam_id], uv_occ, depth_occ,
                voxel_alpha=voxel_alpha,
                point_size=point_size,
                colormap=colormap,
                depth_range=depth_range,
            )

            # Free voxel overlay
            if show_free and free_pts is not None and free_pts.shape[0] > 0:
                uv_free, _ = project_and_filter(free_pts, cam_frames[cam_id])
                if uv_free.shape[0] > 0:
                    ax.scatter(
                        uv_free[:, 0], uv_free[:, 1],
                        c="cyan", s=point_size * 0.5,
                        alpha=voxel_alpha * 0.3, edgecolors="none", rasterized=True,
                    )

        for j in range(n_cams, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            f"Frame {frame_idx}  |  {n_occ} occupied voxels  |  "
            f"voxel_size={occ_data['voxel_size']:.3f}m  |  {label_tag}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_frame_spec(spec: str, total_frames: int) -> list[int]:
    """Parse --frame or --frame-range into a list of frame indices."""
    if spec == "all":
        return list(range(total_frames))
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 and parts[1] else total_frames
        return list(range(start, min(end, total_frames)))
    return [int(spec)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize occupancy labels on camera images."
    )
    parser.add_argument("episode", help="Path to episode HDF5 file or directory")
    parser.add_argument("--frame", default=None, help="Frame index (default: 0)")
    parser.add_argument(
        "--frame-range",
        default=None,
        help='Frame range, e.g. "0:10" or "all" (overrides --frame)',
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for PNGs")
    parser.add_argument(
        "--voxel-alpha", type=float, default=0.6, help="Overlay alpha (default: 0.6)"
    )
    parser.add_argument(
        "--point-size", type=float, default=1.5, help="Scatter point size (default: 1.5)"
    )
    parser.add_argument(
        "--colormap", default="jet", help="Matplotlib colormap name (default: jet)"
    )
    parser.add_argument(
        "--show-free",
        action="store_true",
        help="Also show FREE voxels in cyan overlay",
    )
    parser.add_argument(
        "--tiled", action="store_true",
        help="Render all views in a single tiled image (default: one PNG per camera)",
    )
    parser.add_argument(
        "--exclude-cameras", type=str, default="",
        help="Comma-separated camera IDs to skip (default: '' = include all; "
             "use 'cam_wrist_left,cam_wrist_right' to exclude wrist cameras)",
    )
    parser.add_argument(
        "--object-only",
        action="store_true",
        help="Only show occupied voxels inside object bounding boxes (filters out table/floor/environment)",
    )
    parser.add_argument(
        "--aabb-margin", type=float, default=0.02,
        help="Margin around object AABBs in meters (default: 0.02)",
    )
    parser.add_argument(
        "--table-z", type=float, default=0.75,
        help="Table surface height for AABB Z clamping (default: 0.75)",
    )
    parser.add_argument(
        "--label-key", default="occupancy_tsdf",
        choices=["occupancy_tsdf", "occupancy_gt"],
        help="Which occupancy label group to visualize (default: occupancy_tsdf)",
    )
    args = parser.parse_args()

    exclude_cameras = set(s.strip() for s in args.exclude_cameras.split(",") if s.strip()) if args.exclude_cameras else set()

    # Discover episodes
    episodes = find_hdf5_files(args.episode)
    if not episodes:
        print(f"Error: no HDF5 files found at {args.episode}")
        sys.exit(1)

    for ep_path in episodes:
        print(f"\n=== {ep_path} ===")

        # Read occupancy
        try:
            occ_data = read_occupancy(ep_path, label_key=args.label_key)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            continue

        total_frames = occ_data["state"].shape[0]
        print(
            f"  Occupancy: grid_shape={occ_data['grid_shape'].tolist()}, "
            f"voxel_size={occ_data['voxel_size']:.4f}m, "
            f"frames={total_frames}, "
            f"source={occ_data['source_path']}"
        )

        # Determine frames to render
        if args.frame_range is not None:
            frames = parse_frame_spec(args.frame_range, total_frames)
        elif args.frame is not None:
            frames = parse_frame_spec(args.frame, total_frames)
        else:
            frames = [0]

        # Output directory
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = str(Path(ep_path).parent / "vis_occupancy")
        os.makedirs(out_dir, exist_ok=True)

        for fi in frames:
            if fi < 0 or fi >= total_frames:
                print(f"  Frame {fi}: out of range [0, {total_frames}), skipping.")
                continue
            out_path = os.path.join(out_dir, f"frame_{fi:04d}.png")
            print(f"  Rendering frame {fi}/{total_frames - 1} -> {out_dir}/")
            render_frame(
                ep_path,
                fi,
                occ_data,
                out_path,
                voxel_alpha=args.voxel_alpha,
                point_size=args.point_size,
                colormap=args.colormap,
                show_free=args.show_free,
                exclude_cameras=exclude_cameras,
                object_only=args.object_only,
                aabb_margin=args.aabb_margin,
                table_z=args.table_z if args.object_only else None,
                label_key=args.label_key,
                split_views=not args.tiled,
            )

        print(f"  Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
