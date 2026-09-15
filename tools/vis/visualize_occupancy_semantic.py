#!/usr/bin/env python3
"""Visualize occupancy with per-object semantic colouring and table filtering.

Colours match the episode_viewer palette (``hdf5_episode.OBJECT_COLORS``).
The table surface is filtered out by default.

Usage:
    # Frame 0, objects only (no table), one PNG per camera view (default):
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame 0

    # Include wrist cameras (6 views total):
    #   (pass empty --exclude-cameras to keep all cameras)
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame 0 --exclude-cameras ""

    # Tiled grid of all views in one image (old behaviour):
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame 0 --tiled

    # Include the table surface in the visualisation:
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame 0 --show-table

    # Show free-space voxels as well (light blue, low alpha):
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame 0 --show-free

    # Batch: render frames 0–9:
    python tools/vis/visualize_occupancy_semantic.py episode.hdf5 --frame-range 0:10
"""

from __future__ import annotations

import argparse
import json
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
)
from tools.labels._label_common import (
    read_camera_frames,
    find_hdf5_files,
)

# ---------------------------------------------------------------------------
# Occupancy constants (matching collector/occupancy.py)
# ---------------------------------------------------------------------------
_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2

# ---------------------------------------------------------------------------
# Colour palette — matches episode_viewer/hdf5_episode.py OBJECT_COLORS
# ---------------------------------------------------------------------------
_OBJECT_COLORS = [
    (230, 159, 0),     # orange
    (86, 180, 233),    # light blue
    (0, 158, 115),     # teal-green
    (0, 114, 178),     # blue
    (213, 94, 0),      # vermillion
    (204, 121, 167),   # pink
    (120, 120, 120),   # grey
]

_TABLE_COLOUR   = (175, 178, 180)   # medium grey  (semantic_id == 1)
_UNKNOWN_COLOUR = (235, 235, 235)   # light grey   (semantic_id <= 0)


def _semantic_colour(sem_id: int) -> tuple[float, float, float]:
    """RGB colour for a semantic ID, matching episode_viewer conventions.

    * sem_id <= 0 → unknown  (light grey)
    * sem_id == 1 → table    (medium grey)
    * sem_id >= 2 → object   (cycling through OBJECT_COLORS)
    """
    if sem_id <= 0:
        return tuple(c / 255.0 for c in _UNKNOWN_COLOUR)
    if sem_id == 1:
        return tuple(c / 255.0 for c in _TABLE_COLOUR)
    r, g, b = _OBJECT_COLORS[(sem_id - 2) % len(_OBJECT_COLORS)]
    return (r / 255.0, g / 255.0, b / 255.0)


# ---------------------------------------------------------------------------
# Semantic legend
# ---------------------------------------------------------------------------

def _load_legend(hdf5_path: str, label_key: str) -> dict[int, str]:
    """Return {semantic_id: label_name} from the HDF5 group, or empty dict."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f.get(f"labels/{label_key}")
        if grp is None or "semantic_legend" not in grp:
            return {}
        raw = grp["semantic_legend"][()]
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return {int(k): v for k, v in parsed.items()}


# ---------------------------------------------------------------------------
# Occupancy reading
# ---------------------------------------------------------------------------

def _find_occupancy_group(episode_path: str, label_key: str = "occupancy_gt") -> tuple[str, str] | None:
    with h5py.File(episode_path, "r") as f:
        if "labels" in f and label_key in f["labels"]:
            return episode_path, f"labels/{label_key}"
    source = Path(episode_path)
    sidecar = source.parent / "derived" / f"{source.stem}_derived.hdf5"
    if sidecar.exists():
        with h5py.File(str(sidecar), "r") as f:
            if "labels" in f and label_key in f["labels"]:
                return str(sidecar), f"labels/{label_key}"
    return None


def read_occupancy(episode_path: str, label_key: str = "occupancy_gt") -> dict:
    loc = _find_occupancy_group(episode_path, label_key=label_key)
    if loc is None:
        raise FileNotFoundError(f"No '{label_key}' labels found in {episode_path} or its sidecar.")
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


def extract_voxel_points_with_semantics(
    state_frame: np.ndarray,
    semantic_frame: np.ndarray,
    bounds: np.ndarray,
    voxel_size: float,
    target_state: int = _OCCUPIED,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract world-coordinate voxel centres + per-point semantic IDs.

    Returns (points [N,3], sem_ids [N]) — both float32.
    """
    mask = state_frame == target_state
    ix, iy, iz = np.where(mask)
    if len(ix) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    x = bounds[0, 0] + (ix + 0.5) * voxel_size
    y = bounds[0, 1] + (iy + 0.5) * voxel_size
    z = bounds[0, 2] + (iz + 0.5) * voxel_size
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    sems = semantic_frame[ix, iy, iz].astype(np.float32)
    return pts, sems


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_and_filter(
    points_world: np.ndarray,
    cam_frame: CameraFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points → image.  Returns (uv[M,2], depth[M], idx_in[M])."""
    if points_world.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    uv, valid = project_world_points_to_image_dispatch(points_world, cam_frame)
    H, W = cam_frame.image_shape
    in_bounds = (
        valid
        & (uv[:, 0] >= 0) & (uv[:, 0] < W)
        & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    if cam_frame.fisheye_valid_mask is not None and np.any(in_bounds):
        u_int = np.clip(uv[in_bounds, 0].astype(np.int32), 0, W - 1)
        v_int = np.clip(uv[in_bounds, 1].astype(np.int32), 0, H - 1)
        mask_ok = cam_frame.fisheye_valid_mask[v_int, u_int]
        idx = np.where(in_bounds)[0]
        in_bounds[idx[~mask_ok]] = False
    uv_valid = uv[in_bounds]
    depth_valid = np.linalg.norm(
        points_world[in_bounds] - cam_frame.cam_pos_w, axis=1
    ).astype(np.float32)
    idx_in = np.where(in_bounds)[0]
    return uv_valid, depth_valid, idx_in


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_single_view(
    ax: plt.Axes,
    cam_id: str,
    cam_frame: CameraFrame,
    uv: np.ndarray,
    sem_ids: np.ndarray,
    depth: np.ndarray,
    *,
    legend_map: dict[int, str],
    voxel_alpha: float,
    point_size: float,
    show_subtitle: bool = True,
) -> None:
    """Draw one camera view with per-object semantic scatter on *ax*."""
    H, W = cam_frame.image_shape
    if cam_frame.rgb is not None:
        ax.imshow(cam_frame.rgb)
    else:
        ax.imshow(np.full((H, W, 3), 180, dtype=np.uint8))

    if uv.shape[0] > 0:
        # Sort back-to-front by depth so closer voxels paint on top
        order = np.argsort(-depth)
        uv_sorted = uv[order]
        sem_sorted = sem_ids[order]

        for sid in np.unique(sem_sorted).astype(int):
            mask = sem_sorted == sid
            pts_uv = uv_sorted[mask]
            if pts_uv.shape[0] == 0:
                continue
            colour = _semantic_colour(sid)
            label = legend_map.get(sid, f"id_{sid}")
            ax.scatter(
                pts_uv[:, 0], pts_uv[:, 1],
                c=np.tile(colour, (pts_uv.shape[0], 1)),
                s=point_size, alpha=voxel_alpha,
                edgecolors="none", rasterized=True,
                label=label,
            )

    if show_subtitle:
        ax.set_title(f"{cam_id} ({cam_frame.camera_model})", fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    if show_subtitle:
        ax.tick_params(labelsize=6)
    else:
        ax.set_axis_off()


def _render_one_view(
    cam_id: str,
    cam_frame: CameraFrame,
    uv: np.ndarray,
    sem_ids: np.ndarray,
    depth: np.ndarray,
    *,
    colour_map: dict[int, tuple[float, float, float]],
    legend_map: dict[int, str],
    voxel_alpha: float,
    point_size: float,
    free_pts: np.ndarray | None,
    show_free: bool,
    title: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Render a single camera view as a standalone image — no axes, legend inside."""
    H, W = cam_frame.image_shape
    fig, ax = plt.subplots(figsize=(10, 8))
    render_single_view(
        ax, cam_id, cam_frame, uv, sem_ids, depth,
        legend_map=legend_map,
        voxel_alpha=voxel_alpha,
        point_size=point_size,
        show_subtitle=False,
    )

    # Free voxel overlay
    if show_free and free_pts is not None and free_pts.shape[0] > 0:
        uv_free, _, _ = project_and_filter(free_pts, cam_frame)
        if uv_free.shape[0] > 0:
            ax.scatter(
                uv_free[:, 0], uv_free[:, 1],
                c="cyan", s=point_size * 0.5,
                alpha=voxel_alpha * 0.3, edgecolors="none", rasterized=True,
            )

    # Per-view legend — small, inside image, top-right corner
    unique_sems = np.unique(sem_ids).astype(int) if sem_ids.shape[0] > 0 else []
    display_sids = sorted(set(int(s) for s in unique_sems))
    if display_sids:
        handles, labels = [], []
        for sid in display_sids:
            c = colour_map.get(sid, _semantic_colour(sid))
            name = legend_map.get(sid, f"id_{sid}")
            if len(name) > 40:
                name = name[:37] + "..."
            handles.append(plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=c, markersize=6,
            ))
            labels.append(name)
        if handles:
            ax.legend(
                handles, labels,
                loc="upper right",
                fontsize=5,
                title="objects",
                title_fontsize=6,
                framealpha=0.7,
                borderpad=0.3,
                labelspacing=0.3,
                handletextpad=0.5,
            )

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
    show_table: bool = False,
    show_free: bool = False,
    exclude_cameras: set[str] | None = None,
    label_key: str = "occupancy_gt",
    split_views: bool = True,
) -> None:
    """Render semantic-coloured occupancy for one frame.

    When ``split_views=True`` (default), each camera view is saved as a
    separate PNG (``frame_XXXX_semantic_<cam_id>.png``).  Otherwise all
    views are tiled into a single image.
    """

    # ---- extract occupied voxels + semantic IDs ----
    state_frame = occ_data["state"][frame_idx]
    sem_data = occ_data.get("semantic_id")

    if sem_data is None:
        print("  WARNING: no semantic_id found — colouring all occupied voxels the same")
        from visualize_occupancy import extract_voxel_points
        occupied_pts = extract_voxel_points(
            state_frame, occ_data["bounds"], occ_data["voxel_size"], _OCCUPIED
        )
        sem_ids = np.zeros(occupied_pts.shape[0], dtype=np.float32)
    else:
        sem_frame = sem_data[frame_idx]
        occupied_pts, sem_ids = extract_voxel_points_with_semantics(
            state_frame, sem_frame, occ_data["bounds"], occ_data["voxel_size"], _OCCUPIED
        )
        if not show_table:
            keep = sem_ids != 1
            occupied_pts = occupied_pts[keep]
            sem_ids = sem_ids[keep]

    # ---- FREE voxels (optional) ----
    free_pts = None
    if show_free and sem_data is not None:
        free_pts, free_sems = extract_voxel_points_with_semantics(
            state_frame, sem_frame, occ_data["bounds"], occ_data["voxel_size"], _FREE
        )
        if not show_table:
            keep = free_sems != 1
            free_pts = free_pts[keep]

    # ---- legend ----
    legend_map = _load_legend(episode_path, label_key)

    # ---- cameras ----
    with h5py.File(episode_path, "r") as f:
        cam_frames = read_camera_frames(f, frame_idx, need_images=True)
    if exclude_cameras:
        cam_frames = {k: v for k, v in cam_frames.items() if k not in exclude_cameras}
    if not cam_frames:
        print(f"  Frame {frame_idx}: no cameras found, skipping.")
        return

    # ---- colour map ----
    object_sids = sorted(set(int(s) for s in np.unique(sem_ids) if s >= 2))
    colour_map = {sid: _semantic_colour(sid) for sid in object_sids}
    if show_table:
        colour_map[1] = _semantic_colour(1)

    # ---- project per camera ----
    cam_ids = sorted(cam_frames.keys())
    cam_uv_sem_depth: dict[str, tuple] = {}
    all_unique_sems: set[int] = set()

    for cam_id, cf in cam_frames.items():
        uv_occ, depth_occ, idx_in = project_and_filter(occupied_pts, cf)
        sem_occ = sem_ids[idx_in] if idx_in.shape[0] > 0 else np.array([], dtype=np.float32)
        cam_uv_sem_depth[cam_id] = (uv_occ, sem_occ, depth_occ)
        if sem_occ.shape[0] > 0:
            all_unique_sems.update(np.unique(sem_occ).astype(int).tolist())

    n_occ = occupied_pts.shape[0]
    if show_table:
        n_table = int(np.sum(sem_ids == 1)) if sem_ids.shape[0] > 0 else 0
        tag = f"{n_occ} occupied voxels ({n_table} table)"
    else:
        tag = f"{n_occ} occupied voxels (table filtered)"
    label_tag = occ_data.get("method") or ("GT mesh" if label_key == "occupancy_gt" else "sensor TSDF")

    # ---- render ----
    out_base, out_ext = os.path.splitext(output_path)

    if split_views:
        # One PNG per camera view
        for cam_id in cam_ids:
            uv_occ, sem_occ, depth_occ = cam_uv_sem_depth[cam_id]
            cam_out = f"{out_base}_{cam_id}{out_ext}"
            title = (
                f"Frame {frame_idx}  |  {tag}  |  {cam_id}  |  "
                f"voxel_size={occ_data['voxel_size']:.3f}m  |  {label_tag}"
            )
            _render_one_view(
                cam_id, cam_frames[cam_id],
                uv_occ, sem_occ, depth_occ,
                colour_map=colour_map,
                legend_map=legend_map,
                voxel_alpha=voxel_alpha,
                point_size=point_size,
                free_pts=free_pts,
                show_free=show_free,
                title=title,
                output_path=cam_out,
            )
        print(f"  → {len(cam_ids)} view(s): {out_base}_<cam_id>{out_ext}")
    else:
        # Tiled grid (old behaviour)
        n_cams = len(cam_frames)
        n_cols = min(n_cams, 2)
        n_rows = math.ceil(n_cams / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

        for i, cam_id in enumerate(cam_ids):
            r, c = divmod(i, n_cols)
            ax = axes[r][c]
            uv_occ, sem_occ, depth_occ = cam_uv_sem_depth[cam_id]
            render_single_view(
                ax, cam_id, cam_frames[cam_id],
                uv_occ, sem_occ, depth_occ,
                legend_map=legend_map,
                voxel_alpha=voxel_alpha,
                point_size=point_size,
            )
            # Free voxel overlay
            if show_free and free_pts is not None and free_pts.shape[0] > 0:
                uv_free, _, _ = project_and_filter(free_pts, cam_frames[cam_id])
                if uv_free.shape[0] > 0:
                    ax.scatter(
                        uv_free[:, 0], uv_free[:, 1],
                        c="cyan", s=point_size * 0.5,
                        alpha=voxel_alpha * 0.3, edgecolors="none", rasterized=True,
                    )

        for j in range(n_cams, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r][c].set_visible(False)

        # Shared legend
        display_sids = sorted(set(colour_map.keys()) | all_unique_sems)
        display_sids = [s for s in display_sids if s in colour_map]
        if display_sids:
            handles, labels = [], []
            for sid in display_sids:
                c = colour_map[sid]
                name = legend_map.get(sid, f"id_{sid}")
                if len(name) > 60:
                    name = name[:57] + "..."
                handles.append(plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=c, markersize=8,
                ))
                labels.append(name)
            if handles:
                fig.legend(
                    handles, labels,
                    loc="center right",
                    fontsize=7,
                    title="objects",
                    title_fontsize=8,
                    framealpha=0.85,
                    ncol=1,
                )

        fig.suptitle(
            f"Frame {frame_idx}  |  {tag}  |  "
            f"voxel_size={occ_data['voxel_size']:.3f}m  |  {label_tag}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(
        description="Visualize occupancy with per-object semantic colouring "
                    "(palette matches episode_viewer)."
    )
    parser.add_argument("episode", help="Path to episode HDF5 file or directory")
    parser.add_argument("--frame", default=None, help="Frame index (default: 0)")
    parser.add_argument("--frame-range", default=None,
                        help='Frame range, e.g. "0:10" or "all" (overrides --frame)')
    parser.add_argument("--output-dir", default=None, help="Output directory for PNGs")
    parser.add_argument("--voxel-alpha", type=float, default=0.6,
                        help="Overlay alpha (default: 0.6)")
    parser.add_argument("--point-size", type=float, default=1.5,
                        help="Scatter point size (default: 1.5)")
    parser.add_argument("--show-table", action="store_true",
                        help="Include the TABLE surface (excluded by default)")
    parser.add_argument("--show-free", action="store_true",
                        help="Also show FREE voxels in cyan overlay")
    parser.add_argument("--tiled", action="store_true",
                        help="Render all views in a single tiled image (default: one PNG per camera)")
    parser.add_argument("--exclude-cameras", type=str, default="",
                        help="Comma-separated camera IDs to skip (default: '' = include all; "
                             "use 'cam_wrist_left,cam_wrist_right' to exclude wrist cameras)")
    parser.add_argument("--label-key", default="occupancy_gt",
                        choices=["occupancy_tsdf", "occupancy_gt"],
                        help="Which occupancy label group to visualise (default: occupancy_gt)")
    args = parser.parse_args()

    exclude_cameras = (
        set(s.strip() for s in args.exclude_cameras.split(",") if s.strip())
        if args.exclude_cameras else set()
    )

    episodes = find_hdf5_files(args.episode)
    if not episodes:
        print(f"Error: no HDF5 files found at {args.episode}")
        sys.exit(1)

    for ep_path in episodes:
        print(f"\n=== {ep_path} ===")
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
        if occ_data.get("semantic_id") is not None:
            legend = _load_legend(ep_path, args.label_key)
            if legend:
                print(f"  Semantic classes: {json.dumps({k: v for k, v in sorted(legend.items())})}")
            else:
                print(f"  Semantic classes: {np.unique(occ_data['semantic_id'][0]).tolist()} (no legend)")

        if args.frame_range is not None:
            frames = parse_frame_spec(args.frame_range, total_frames)
        elif args.frame is not None:
            frames = parse_frame_spec(args.frame, total_frames)
        else:
            frames = [0]

        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = str(Path(ep_path).parent / "vis_occupancy_semantic")
        os.makedirs(out_dir, exist_ok=True)

        for fi in frames:
            if fi < 0 or fi >= total_frames:
                print(f"  Frame {fi}: out of range [0, {total_frames}), skipping.")
                continue
            out_path = os.path.join(out_dir, f"frame_{fi:04d}_semantic.png")
            print(f"  Rendering frame {fi}/{total_frames - 1} -> {out_dir}/")
            render_frame(
                ep_path, fi, occ_data, out_path,
                voxel_alpha=args.voxel_alpha,
                point_size=args.point_size,
                show_table=args.show_table,
                show_free=args.show_free,
                exclude_cameras=exclude_cameras,
                label_key=args.label_key,
                split_views=not args.tiled,
            )

        print(f"  Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
