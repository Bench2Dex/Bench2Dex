#!/usr/bin/env python3
"""Export per-camera visualizations for a single frame from an HDF5 episode.

Generates 5 image types per camera:
  - rgb       : raw RGB frame
  - depth     : depth with viridis colormap
  - occupancy : occupancy voxel projection overlay
  - box3d     : 3D bounding-box wireframe overlay
  - box2d     : 2D bounding-box overlay

Usage:
    python export_frame_views.py <episode.hdf5> --frame 0 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.camera_geometry import (
    CameraFrame,
    project_world_points_to_image_dispatch,
    quat_xyzw_to_rot,
    world_points_to_camera,
)
from tools.labels._label_common import (
    read_camera_frames,
    read_local_bboxes,
    read_object_states,
)

# ── occupancy constants ────────────────────────────────────────────────────
_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2

# Corner order from _compute_box_corners:
#   0(-1,-1,-1) 1(+1,-1,-1) 2(+1,+1,-1) 3(-1,+1,-1)   <- back face (z=-1)
#   4(-1,-1,+1) 5(+1,-1,+1) 6(+1,+1,+1) 7(-1,+1,+1)   <- front face (z=+1)
# Edges must be the 12 cuboid edges (no face diagonals).
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # back face
    (4, 5), (5, 6), (6, 7), (7, 4),   # front face
    (0, 4), (1, 5), (2, 6), (3, 7),   # connecting edges
]
# Match the episode viewer palette (tools/vis/episode_viewer/hdf5_episode.py) so
# box / occupancy overlays are color-consistent with app.py.
_OBJECT_COLORS = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in [
    (230, 159, 0),   # orange
    (86, 180, 233),  # sky blue
    (0, 158, 115),   # bluish green
    (0, 114, 178),   # blue
    (213, 94, 0),    # vermillion
    (204, 121, 167), # reddish purple
    (120, 120, 120), # gray
]]
_TABLE_GRAY = (175 / 255.0, 178 / 255.0, 180 / 255.0)
_UNKNOWN_GRAY = (235 / 255.0, 235 / 255.0, 235 / 255.0)
MAX_OCCUPANCY_POINTS = 80000


def _occupancy_color(semantic_id: int) -> tuple:
    if semantic_id <= 0:
        return _UNKNOWN_GRAY
    if semantic_id == 1:
        return _TABLE_GRAY
    return _OBJECT_COLORS[(semantic_id - 2) % len(_OBJECT_COLORS)]


# Output resolution is derived from the native image size, preserving the
# aspect ratio (no letterbox / black bars). _MAX_SIDE caps the long edge:
# a 640x480 source -> 2560x1920 (4x). Overridable via --max-side.
_MAX_SIDE = 2560
_FIG_DPI = 150


def _resize_native(rgb: np.ndarray, max_side: int | None = None) -> tuple[np.ndarray, float]:
    """Aspect-preserving upscale of an (h, w, 3) RGB array.

    Returns (resized_rgb, scale) where scale maps source pixel coordinates to
    output coordinates: out_pt = src_pt * scale.
    """
    if max_side is None:
        max_side = _MAX_SIDE
    h, w = rgb.shape[:2]
    scale = max_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8
    )
    return resized, float(scale)


def _new_figure(width: int, height: int) -> tuple[plt.Figure, plt.Axes]:
    """Matplotlib figure that yields an exact width x height PNG at _FIG_DPI."""
    fig, ax = plt.subplots(figsize=(width / _FIG_DPI, height / _FIG_DPI), dpi=_FIG_DPI)
    fig.subplots_adjust(left=0, right=1, top=0.96, bottom=0)
    return fig, ax


# ── helpers ─────────────────────────────────────────────────────────────────

def _decode_rgb_pil(file: h5py.File, cam_id: str, idx: int) -> Image.Image | None:
    ds = file.get(f"cameras/{cam_id}/rgb")
    if not isinstance(ds, h5py.Dataset):
        return None
    value = ds[idx]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
    data = bytes(value) if isinstance(value, (bytes, bytearray)) else bytes(np.asarray(value).tobytes())
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _read_depth(file: h5py.File, cam_id: str, idx: int) -> np.ndarray | None:
    grp = file.get(f"cameras/{cam_id}")
    if grp is None:
        return None
    for key in ("depth", "depth_m"):
        if key in grp:
            d = np.asarray(grp[key][idx], dtype=np.float32)
            d[~np.isfinite(d)] = np.nan
            return d
    return None


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 0.01
    normed = np.clip((depth - lo) / (hi - lo), 0, 1)
    rgba = plt.cm.viridis(normed)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _find_occupancy_data(file: h5py.File) -> dict | None:
    for label_key in ("occupancy_tsdf", "occupancy_gt"):
        grp = file.get(f"labels/{label_key}")
        if grp is None:
            continue
        # Format A: state + bounds + voxel_size (occupancy_gt / occupancy_tsdf label groups).
        #   state may be (nx,ny,nz) for a single grid or (frames,nx,ny,nz) for per-frame grids.
        if "state" in grp:
            state = np.asarray(grp["state"][:])
            bounds = np.asarray(grp["bounds"][()], dtype=np.float32)
            voxel_size = float(grp["voxel_size"][()]) if "voxel_size" in grp else 0.01
            semantic_id = np.asarray(grp["semantic_id"][:]) if "semantic_id" in grp else None
            semantic_legend = None
            if "semantic_legend" in grp:
                raw = grp["semantic_legend"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    semantic_legend = json.loads(raw)
                except Exception:
                    semantic_legend = None
            return {
                "grid": state,
                "origin": bounds[0],
                "voxel_size": voxel_size,
                "method": label_key,
                "per_frame": state.ndim == 4,
                "semantic_id": semantic_id,
                "semantic_legend": semantic_legend,
            }
        # Format B: grid/data + origin + voxel_size
        if "grid" in grp:
            grid = np.asarray(grp["grid"][:])
        elif "data" in grp:
            grid = np.asarray(grp["data"][:])
        else:
            continue
        origin = np.asarray(grp["origin"][:]) if "origin" in grp else np.zeros(3)
        voxel_size = float(grp["voxel_size"][()]) if "voxel_size" in grp else 0.01
        return {"grid": grid, "origin": origin, "voxel_size": voxel_size, "method": label_key}
    # Sidecar
    sidecar = file.attrs.get("sidecar_path", None) or file.attrs.get("label_sidecar", None)
    if sidecar:
        sc_path = sidecar.decode() if isinstance(sidecar, bytes) else str(sidecar)
        if os.path.exists(sc_path):
            try:
                with h5py.File(sc_path, "r") as sf:
                    return _find_occupancy_data(sf)
            except Exception:
                pass
    return None


def _occupied_points(occ_data: dict, frame_idx: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (occupied voxel centers, per-voxel semantic ids).

    semantic ids are None when the occupancy group has no semantic_id channel.
    """
    grid = occ_data["grid"]
    origin = occ_data["origin"]
    vs = occ_data["voxel_size"]
    if grid.ndim == 4:
        grid = grid[frame_idx]
    inds = np.argwhere(grid == _OCCUPIED)
    pts = origin + (inds + 0.5) * vs

    sem = occ_data.get("semantic_id")
    sems = None
    if sem is not None and inds.shape[0] > 0:
        sem_frame = sem[frame_idx] if sem.ndim == 4 else sem
        sems = sem_frame[inds[:, 0], inds[:, 1], inds[:, 2]].astype(np.int32)

    # Cap the number of projected points (matches the episode viewer's
    # MAX_OCCUPANCY_POINTS_PER_VIEW), so heavy scenes stay responsive.
    if pts.shape[0] > MAX_OCCUPANCY_POINTS:
        keep = np.linspace(0, pts.shape[0] - 1, MAX_OCCUPANCY_POINTS, dtype=np.int64)
        pts = pts[keep]
        if sems is not None:
            sems = sems[keep]
    return pts, sems


def _project_occ(pts: np.ndarray, cam_frame: CameraFrame):
    if pts.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,))
    uv, front = project_world_points_to_image_dispatch(pts, cam_frame)
    if not np.any(front):
        return np.empty((0, 2)), np.empty((0,))
    # Camera-frame optical depth for coloring.
    p_cam, _ = world_points_to_camera(pts, cam_frame.extrinsic_world_from_cam)
    depths = p_cam[:, 2]
    H, W = cam_frame.image_shape
    in_bounds = (
        front
        & (uv[:, 0] >= 0) & (uv[:, 0] < W)
        & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    return uv[in_bounds], depths[in_bounds]


def _read_box3d_data(file: h5py.File, frame_idx: int) -> dict:
    # Preferred: ground-truth /labels/box3d group with per-frame center/quat/size.
    box3d_grp = file.get("labels/box3d")
    if box3d_grp is not None:
        result = {}
        for oid in box3d_grp.keys():
            g = box3d_grp[oid]
            if not isinstance(g, h5py.Group):
                continue
            if not (g.get("center_world") and g.get("size_lwh") and g.get("quat_world")):
                continue
            if frame_idx >= g["center_world"].shape[0]:
                continue
            center = np.asarray(g["center_world"][frame_idx], dtype=np.float64)
            size = np.asarray(g["size_lwh"][frame_idx], dtype=np.float64)
            quat = np.asarray(g["quat_world"][frame_idx], dtype=np.float64)
            if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size))):
                continue
            result[oid] = {
                "center": center,
                "size": np.maximum(size, 0.02),  # floor at 2 cm
                "quat": quat,
            }
        if result:
            return result

    # Fall back: object states + local bboxes
    objects = read_object_states(file, frame_idx)
    bboxes = read_local_bboxes(file)  # {obj_id: ((min_x,min_y,min_z), (max_x,max_y,max_z))}
    result = {}
    DEFAULT_SIZE = np.asarray([0.08, 0.08, 0.08], dtype=np.float64)
    for oid, state in objects.items():
        # pose_world = [x, y, z, qx, qy, qz, qw] (xyzw)
        pose = state["pose_world"]
        center = np.asarray(pose[:3], dtype=np.float64)
        quat_xyzw = np.asarray(pose[3:7], dtype=np.float64)
        if bboxes and oid in bboxes:
            lo, hi = bboxes[oid]
            size = np.asarray([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]], dtype=np.float64)
            size = np.maximum(size, 0.02)  # floor at 2 cm
        else:
            size = DEFAULT_SIZE
        result[oid] = {
            "center": center,
            "size": size,
            "quat": quat_xyzw,
        }
    return result


def _read_box2d_data(file: h5py.File, cam_id: str, frame_idx: int) -> dict:
    grp = file.get(f"labels/box2d/{cam_id}")
    if grp is None:
        return {}
    result = {}
    for oid in grp.keys():
        item = grp[oid]
        if isinstance(item, h5py.Group):
            # Group layout: 'visible' (N,) + 'xyxy' (N, 4)
            xyxy = item.get("xyxy")
            if xyxy is None or frame_idx >= xyxy.shape[0]:
                continue
            if "visible" in item and frame_idx < item["visible"].shape[0]:
                vis = np.asarray(item["visible"][frame_idx]).reshape(-1)
                if vis.size and not bool(vis[0]):
                    continue
            row = np.asarray(xyxy[frame_idx], dtype=np.float64)
            if row.size < 4:
                continue
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if x2 > x1 and y2 > y1:
                result[oid] = {"xyxy": (x1, y1, x2, y2)}
        elif isinstance(item, h5py.Dataset):
            # Legacy flat layout: dataset of shape (N, 4)
            if frame_idx < item.shape[0]:
                row = np.asarray(item[frame_idx], dtype=np.float64)
                if row.size >= 4:
                    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    if x2 > x1 and y2 > y1:
                        result[oid] = {"xyxy": (x1, y1, x2, y2)}
    return result


def _compute_box_corners(center, size, quat_xyzw):
    half = size / 2.0
    corners_local = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ]) * half
    R = quat_xyzw_to_rot(np.asarray(quat_xyzw))
    return (R @ corners_local.T).T + center


def _project_edges(corners, cam_frame):
    """Project the cuboid edges, keeping each edge independently.

    Mirrors episode_viewer/hdf5_episode.py: an edge is drawn when both its
    projected endpoints are valid (in front of the camera) and lie within the
    extended frame bounds [-0.5W, 1.5W] x [-0.5H, 1.5H].  This keeps edges that
    cross the image boundary (clipped by the axes) and avoids dropping a whole
    box when only one corner is behind the camera.
    """
    uv, valid = project_world_points_to_image_dispatch(corners, cam_frame)
    W, H = cam_frame.image_shape[1], cam_frame.image_shape[0]
    lo_x, hi_x = -0.5 * W, 1.5 * W
    lo_y, hi_y = -0.5 * H, 1.5 * H
    lines = []
    for i0, i1 in _BOX_EDGES:
        if not (valid[i0] and valid[i1]):
            continue  # a corner behind the camera: skip this edge only
        u0, v0 = uv[i0]
        u1, v1 = uv[i1]
        if (lo_x <= u0 <= hi_x and lo_y <= v0 <= hi_y and
                lo_x <= u1 <= hi_x and lo_y <= v1 <= hi_y):
            lines.append(np.array([[u0, v0], [u1, v1]]))
    return lines


# ── per-camera render ───────────────────────────────────────────────────────

def save_rgb(file, cam_id, idx, out_path):
    img = _decode_rgb_pil(file, cam_id, idx)
    if img is not None:
        canvas, _ = _resize_native(np.asarray(img.convert("RGB"), dtype=np.uint8))
        Image.fromarray(canvas).save(out_path)


def save_depth(file, cam_id, idx, out_path):
    d = _read_depth(file, cam_id, idx)
    if d is not None:
        rgb = _depth_to_rgb(d)
        canvas, _ = _resize_native(rgb)
        Image.fromarray(canvas).save(out_path)


def _canvas_and_transform(file, cam_id, frame_idx, cam_shape=None):
    """Return (canvas RGB, scale) for the aspect-preserving background image."""
    img = _decode_rgb_pil(file, cam_id, frame_idx)
    if img is not None:
        return _resize_native(np.asarray(img.convert("RGB"), dtype=np.uint8))
    h, w = cam_shape if cam_shape else (480, 640)
    rgb = np.full((int(h), int(w), 3), 180, dtype=np.uint8)
    return _resize_native(rgb)


def _src_to_canvas(pts: np.ndarray, scale: float) -> np.ndarray:
    return pts * scale


def save_occupancy(file, cam_id, cam_frame, occ_data, out_path, frame_idx=0):
    pts, sems = _occupied_points(occ_data, frame_idx)
    canvas, scale = _canvas_and_transform(file, cam_id, frame_idx, cam_frame.image_shape)
    Hc, Wc = canvas.shape[:2]
    fig, ax = _new_figure(Wc, Hc)
    ax.imshow(canvas)
    # Same overlay alpha as the episode viewer (170/255).
    _occ_alpha = 170 / 255.0
    if pts.shape[0] > 0:
        uv, depths = _project_occ(pts, cam_frame)
        if uv.shape[0] > 0:
            order = np.argsort(-depths)  # back-to-front: nearest drawn last
            uv_c = _src_to_canvas(uv[order], scale)
            if sems is not None:
                sem_draw = sems[order]
                table_mask = sem_draw == 1
                obj_mask = sem_draw >= 2
            else:
                sem_draw = None
                table_mask = np.zeros(uv_c.shape[0], dtype=bool)
                obj_mask = np.ones(uv_c.shape[0], dtype=bool)

            if np.any(table_mask):
                ax.scatter(uv_c[table_mask, 0], uv_c[table_mask, 1],
                           c=_TABLE_GRAY, s=3, alpha=_occ_alpha,
                           edgecolors="none", rasterized=True)

            if np.any(obj_mask):
                colors = [_occupancy_color(int(s)) for s in sem_draw[obj_mask]]
                ax.scatter(uv_c[obj_mask, 0], uv_c[obj_mask, 1], c=colors,
                           s=4, alpha=_occ_alpha, edgecolors="none", rasterized=True)

                legend = occ_data.get("semantic_legend") or {}
                handles = []
                for s in sorted(np.unique(sem_draw[obj_mask])):
                    label = legend.get(str(int(s))) or f"sem_{int(s)}"
                    handles.append(patches.Patch(color=_occupancy_color(int(s)), label=label))
                if handles:
                    ax.legend(handles=handles, loc="upper left", fontsize=12,
                              framealpha=0.6, facecolor="black", labelcolor="white")

    ax.set_xlim(0, Wc)
    ax.set_ylim(Hc, 0)
    ax.set_title(f"{cam_id} — occupancy", fontsize=18)
    ax.axis("off")
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)


def save_box3d(file, cam_id, cam_frame, box3d_data, obj_colors, out_path, frame_idx=0):
    canvas, scale = _canvas_and_transform(file, cam_id, frame_idx, cam_frame.image_shape)
    Hc, Wc = canvas.shape[:2]
    fig, ax = _new_figure(Wc, Hc)
    ax.imshow(canvas)
    for oid, bd in box3d_data.items():
        corners = _compute_box_corners(bd["center"], bd["size"], bd["quat"])
        lines = _project_edges(corners, cam_frame)
        color = obj_colors.get(oid, (1, 1, 1))
        for poly in lines:
            poly_c = _src_to_canvas(poly, scale)
            ax.plot(poly_c[:, 0], poly_c[:, 1], color=color, linewidth=3, alpha=0.85)
    ax.set_xlim(0, Wc); ax.set_ylim(Hc, 0)
    ax.set_title(f"{cam_id} — 3D bbox", fontsize=18)
    ax.axis("off")
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)


def save_box2d(file, cam_id, cam_frame, box2d_data, obj_colors, out_path, frame_idx=0):
    canvas, scale = _canvas_and_transform(file, cam_id, frame_idx, cam_frame.image_shape)
    Hc, Wc = canvas.shape[:2]
    fig, ax = _new_figure(Wc, Hc)
    ax.imshow(canvas)
    for oid, bd in box2d_data.items():
        color = obj_colors.get(oid, (1, 1, 1))
        x1, y1, x2, y2 = bd["xyxy"]
        p1 = _src_to_canvas(np.array([[x1, y1]], dtype=np.float64), scale)[0]
        p2 = _src_to_canvas(np.array([[x2, y2]], dtype=np.float64), scale)[0]
        rect = patches.Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1],
                                 linewidth=4, edgecolor=color, facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(p1[0], max(0, p1[1] - 12), oid.replace("obj_", ""), fontsize=13,
                color=color, backgroundcolor=(0, 0, 0, 0.5), verticalalignment="bottom")
    ax.set_xlim(0, Wc); ax.set_ylim(Hc, 0)
    ax.set_title(f"{cam_id} — 2D bbox", fontsize=18)
    ax.axis("off")
    fig.savefig(out_path, dpi=_FIG_DPI)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────

def export_all(hdf5_path: str, frame_idx: int, out_root: str) -> None:
    ep_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    ep_dir = os.path.join(out_root, ep_name)

    with h5py.File(hdf5_path, "r") as f:
        cam_frames = read_camera_frames(f, frame_idx)
        if not cam_frames:
            print(f"[SKIP] {ep_name}: no camera frames"); return

        cam_ids = sorted(cam_frames.keys())
        occ_data = _find_occupancy_data(f)
        box3d_data = _read_box3d_data(f, frame_idx)
        # Color map for objects
        obj_ids = sorted(set(list(box3d_data.keys()) + list(
            oid for cam in cam_ids for oid in _read_box2d_data(f, cam, frame_idx).keys()
        )))
        obj_colors = {oid: _OBJECT_COLORS[i % len(_OBJECT_COLORS)] for i, oid in enumerate(obj_ids)}

        # Pre-read box3d + per-camera box2d
        box2d_by_cam = {cam: _read_box2d_data(f, cam, frame_idx) for cam in cam_ids}

        for cam_id in cam_ids:
            cam_dir = os.path.join(ep_dir, cam_id)
            os.makedirs(cam_dir, exist_ok=True)

            cf = cam_frames[cam_id]

            # 1. RGB
            save_rgb(f, cam_id, frame_idx, os.path.join(cam_dir, "rgb.png"))

            # 2. Depth
            save_depth(f, cam_id, frame_idx, os.path.join(cam_dir, "depth.png"))

            # 3. Occupancy
            if occ_data is not None:
                save_occupancy(f, cam_id, cf, occ_data, os.path.join(cam_dir, "occupancy.png"), frame_idx=frame_idx)

            # 4. 3D bbox
            if box3d_data:
                save_box3d(f, cam_id, cf, box3d_data, obj_colors, os.path.join(cam_dir, "box3d.png"), frame_idx=frame_idx)

            # 5. 2D bbox
            b2d = box2d_by_cam.get(cam_id, {})
            if b2d:
                save_box2d(f, cam_id, cf, b2d, obj_colors, os.path.join(cam_dir, "box2d.png"), frame_idx=frame_idx)

        n_cams = len(cam_ids)
        n_types = 5
        print(f"[{ep_name}] {n_cams} cameras × {n_types} types → {ep_dir}")
    print()


def main():
    global _MAX_SIDE
    p = argparse.ArgumentParser(description="Export per-camera frame visualizations")
    p.add_argument("hdf5", nargs="+", help="HDF5 episode files")
    p.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--max-side", type=int, default=_MAX_SIDE,
                   help="Cap the long edge in px, preserving the native aspect ratio "
                        "(no letterbox black bars). Default: 2560")
    args = p.parse_args()
    _MAX_SIDE = args.max_side

    os.makedirs(args.out_dir, exist_ok=True)
    for hp in args.hdf5:
        if not os.path.isfile(hp):
            print(f"[SKIP] not found: {hp}"); continue
        export_all(hp, args.frame, args.out_dir)


if __name__ == "__main__":
    main()
