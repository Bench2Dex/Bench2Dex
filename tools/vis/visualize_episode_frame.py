#!/usr/bin/env python3
"""Visualize a single frame from an HDF5 episode as four 2K images.

Generates:
  1. rgb_frame_NNNN.png       — RGB from all 6 cameras
  2. depth_frame_NNNN.png     — Depth colormap from all 6 cameras (full range)
  3. joint_frame_NNNN.png     — RGB with joint angle HUD overlay
  4. objpose_frame_NNNN.png   — RGB with 3D axes + pose text labels

Usage:
    python tools/visualize_episode_frame.py <episode.hdf5> --frame 4
    python tools/visualize_episode_frame.py <episode.hdf5> --frame 4 --output-dir my_vis
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add project root to path for collector imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collector.camera_geometry import world_points_to_camera, project_camera_points, project_camera_points_fisheye

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 2560, 1440
DPI = 100

CAMERA_ORDER = [
    "cam_chest",
    "cam_overhead",
    "cam_stereo_left",
    "cam_stereo_right",
    "cam_wrist_left",
    "cam_wrist_right",
]

# Colorblind-safe object palette (Okabe-Ito derived, BGR for cv2)
OBJ_PALETTE_BGR = [
    (0, 159, 230),    # orange  #E69F00
    (233, 180, 86),   # sky blue #56B4E9
    (115, 158, 0),    # green   #009E73
    (0, 114, 178),    # blue    #0072B2
    (0, 94, 213),     # vermillion #D55E00
    (167, 121, 204),  # purple  #CC79A7
]
OBJ_PALETTE_RGB = [(b, g, r) for (r, g, b) in OBJ_PALETTE_BGR]

AXIS_COLORS_BGR = [(0, 0, 220), (0, 180, 0), (220, 80, 0)]  # X=red, Y=green, Z=blue
AXIS_LABELS = ["X", "Y", "Z"]

# HUD styling
HUD_BG = (30, 30, 30)
HUD_ALPHA = 0.75
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------
def _project_world(pts_world, world_from_cam, K, cam_model, dist):
    """Project world points -> pixel coords via collector.camera_geometry.

    Handles Isaac body frame (X=fwd, Y=left, Z=up) -> CV optical (X=right, Y=down, Z=fwd).
    """
    pts = np.asarray(pts_world, dtype=np.float32).reshape(-1, 3)
    pts_cam, valid = world_points_to_camera(pts, world_from_cam)
    if not np.all(valid):
        return None, pts_cam
    if cam_model == "fisheye" and dist is not None:
        pts_2d = project_camera_points_fisheye(pts_cam, K, dist)
    else:
        pts_2d = project_camera_points(pts_cam, K)
    return pts_2d, pts_cam


def _read_camera_frame(f: h5py.File, cam_id: str, frame: int):
    g = f[f"cameras/{cam_id}"]
    rgb = g["rgb"][frame]
    depth = g["depth_m"][frame]
    K = g["intrinsic"][:]
    E = g["extrinsic_world_from_cam"][frame]
    raw = g["camera_model"][()]
    cam_model = raw.decode() if isinstance(raw, bytes) else str(raw)
    dist = None
    if cam_model == "fisheye":
        K = g["fisheye_camera_matrix"][:]
        dist = g["distortion_coefficients"][:]
    return rgb, depth, K, E, cam_model, dist


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _rounded_rect(img, pt1, pt2, color, radius=8, alpha=0.75):
    """Draw a filled rounded rectangle with alpha blending."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # Clamp
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    # Draw rounded rect via multiple primitives
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_hud_panel(img, lines, x, y, scale=0.40, line_h=18, pad=8,
                    bg_color=HUD_BG, bg_alpha=HUD_ALPHA):
    """Draw a compact HUD text panel with rounded-rect background."""
    # Measure block size
    max_w = 0
    for text, _ in lines:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
        max_w = max(max_w, tw)

    block_h = len(lines) * line_h + pad * 2
    block_w = max_w + pad * 2

    # Clamp position
    h, w = img.shape[:2]
    x = min(x, w - block_w - 2)
    y = min(y, h - block_h - 2)
    x = max(x, 2)
    y = max(y, 2)

    # Background
    _rounded_rect(img, (x, y), (x + block_w, y + block_h), bg_color, radius=6, alpha=bg_alpha)

    # Text
    cy = y + pad + int(line_h * 0.75)
    for text, color in lines:
        cv2.putText(img, text, (x + pad, cy), FONT, scale, color, 1, cv2.LINE_AA)
        cy += line_h


def _draw_arrow(img, pt1, pt2, color, thickness=2, tip_len=8):
    """Draw a line with an arrowhead."""
    cv2.arrowedLine(img, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA, tipLength=tip_len / max(1, np.linalg.norm(pt2 - pt1)))


def _draw_tag(img, text, pos, color, scale=0.38, pad=3):
    """Draw a small text tag with pill-shaped background."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(pos[0]), int(pos[1])
    h, w = img.shape[:2]
    if x < 0 or x + tw + pad * 2 > w or y - th - pad < 0 or y + pad > h:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def _make_6view_figure(images, cam_ids, title):
    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI, facecolor="white")
    gs = GridSpec(2, 3, figure=fig, wspace=0.02, hspace=0.06, top=0.94, bottom=0.01, left=0.01, right=0.99)
    fig.suptitle(title, fontsize=16, fontweight="bold", fontfamily="sans-serif")
    for idx, (img, cam_id) in enumerate(zip(images, cam_ids)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(cam_id, fontsize=11, pad=3, fontfamily="monospace")
        ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# 1. RGB
# ---------------------------------------------------------------------------
def gen_rgb(f, frame, output_dir):
    imgs, ids = [], []
    for cam in CAMERA_ORDER:
        rgb, *_ = _read_camera_frame(f, cam, frame)
        imgs.append(rgb)
        ids.append(cam)
    fig = _make_6view_figure(imgs, ids, f"RGB  |  frame {frame}")
    path = os.path.join(output_dir, f"rgb_frame_{frame:04d}.png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Depth
# ---------------------------------------------------------------------------
def gen_depth(f, frame, output_dir):
    depths, ids = [], []
    for cam in CAMERA_ORDER:
        _, d, *_ = _read_camera_frame(f, cam, frame)
        depths.append(d)
        ids.append(cam)

    # Per-camera normalization: each camera uses its own valid depth range
    imgs = []
    for d in depths:
        valid = d[np.isfinite(d)]
        if valid.size == 0:
            imgs.append(np.zeros((*d.shape, 3), dtype=np.uint8))
            continue
        vmin_c, vmax_c = float(valid.min()), float(valid.max())
        dc = np.clip(d, vmin_c, vmax_c)
        norm = (dc - vmin_c) / (vmax_c - vmin_c + 1e-8)
        # Grayscale: near=white, far=black
        gray = ((1.0 - norm) * 255).astype(np.uint8)
        c = np.stack([gray, gray, gray], axis=-1)
        c[~np.isfinite(d)] = 0
        imgs.append(c)

    # Global range for title display only
    all_valid = np.concatenate([d[np.isfinite(d)].ravel() for d in depths])
    vmin, vmax = float(all_valid.min()), float(all_valid.max())
    fig = _make_6view_figure(imgs, ids, f"Depth  |  frame {frame}  |  per-camera normalized  |  global [{vmin:.2f}, {vmax:.2f}] m")
    path = os.path.join(output_dir, f"depth_frame_{frame:04d}.png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Joint State — clean HUD overlay
# ---------------------------------------------------------------------------
def gen_joint_state(f, frame, output_dir):
    joint_names = [x.decode() if isinstance(x, bytes) else x for x in f["robot/joint_names"][()]]
    qpos_deg = np.degrees(f["robot/qpos"][frame])

    # Build joint groups: (short_name, value)
    def _short(n):
        return (
            n.replace("L_arm_", "")
            .replace("left_", "L.")
            .replace("right_", "R.")
            .replace("_joint", "")
        )

    left_arm = [(i, _short(n), qpos_deg[i]) for i, n in enumerate(joint_names) if n.startswith("L_arm_")]
    right_arm = [(i, _short(n), qpos_deg[i]) for i, n in enumerate(joint_names)
                 if not n.startswith("L_") and not n.startswith("left_") and not n.startswith("right_") and i < 12]
    left_hand = [(i, _short(n), qpos_deg[i]) for i, n in enumerate(joint_names) if n.startswith("left_")]
    right_hand = [(i, _short(n), qpos_deg[i]) for i, n in enumerate(joint_names) if n.startswith("right_")]

    # Colors
    c_header = (255, 255, 255)
    c_left = (140, 200, 255)   # light blue
    c_right = (255, 180, 100)  # light orange
    c_dim = (170, 170, 170)

    def _build_lines(label, joints, c_val):
        lines = [(f"[ {label} ]", c_header)]
        for _, name, val in joints:
            lines.append((f" {name:<16s}{val:>+7.1f}", c_val))
        return lines

    def _measure_panel_w(lines, scale=0.32):
        return max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in lines) + 20

    imgs, ids = [], []
    for cam in CAMERA_ORDER:
        rgb, *_ = _read_camera_frame(f, cam, frame)
        img = rgb.copy()
        h, w = img.shape[:2]

        if cam == "cam_wrist_left":
            # Fisheye left: only show left arm + left hand
            # Left hand is at bottom-right of image -> put HUD at top-left
            lines = _build_lines("LEFT ARM", left_arm, c_left)
            lines.append(("", c_dim))
            lines += _build_lines("LEFT HAND", left_hand, c_left)
            _draw_hud_panel(img, lines, 6, 6, scale=0.32, line_h=15, pad=6)

        elif cam == "cam_wrist_right":
            # Fisheye right: only show right arm + right hand
            # Right hand is at bottom-left of image -> put HUD at top-right
            lines = _build_lines("RIGHT ARM", right_arm, c_right)
            lines.append(("", c_dim))
            lines += _build_lines("RIGHT HAND", right_hand, c_right)
            rx = w - _measure_panel_w(lines) - 4
            _draw_hud_panel(img, lines, rx, 6, scale=0.32, line_h=15, pad=6)

        else:
            # Non-wrist cameras: show both sides
            left_lines = _build_lines("LEFT ARM", left_arm, c_left)
            left_lines.append(("", c_dim))
            left_lines += _build_lines("LEFT HAND", left_hand, c_left)
            _draw_hud_panel(img, left_lines, 6, 6, scale=0.32, line_h=15, pad=6)

            right_lines = _build_lines("RIGHT ARM", right_arm, c_right)
            right_lines.append(("", c_dim))
            right_lines += _build_lines("RIGHT HAND", right_hand, c_right)
            rx = w - _measure_panel_w(right_lines) - 4
            _draw_hud_panel(img, right_lines, rx, 6, scale=0.32, line_h=15, pad=6)

        imgs.append(img)
        ids.append(cam)

    fig = _make_6view_figure(imgs, ids, f"Joint State  |  frame {frame}")
    path = os.path.join(output_dir, f"joint_frame_{frame:04d}.png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Object Pose — axes with arrows + on-image labels
# ---------------------------------------------------------------------------
def gen_objpose(f, frame, output_dir):
    from scipy.spatial.transform import Rotation

    obj_ids = list(f["objects"].keys())
    obj_poses = {oid: f[f"objects/{oid}/pose_world"][frame] for oid in obj_ids}

    imgs, ids = [], []
    for cam in CAMERA_ORDER:
        rgb, depth, K, E, cam_model, dist = _read_camera_frame(f, cam, frame)
        img = rgb.copy()
        h, w = img.shape[:2]

        for ci, oid in enumerate(obj_ids):
            pose = obj_poses[oid]
            origin = pose[:3]
            quat_xyzw = pose[3:]
            obj_color = OBJ_PALETTE_BGR[ci % len(OBJ_PALETTE_BGR)]
            obj_color_rgb = OBJ_PALETTE_RGB[ci % len(OBJ_PALETTE_RGB)]

            R = Rotation.from_quat(quat_xyzw).as_matrix()

            # Adaptive axis length: scale with distance to camera
            pts_cam_origin, _ = world_points_to_camera(origin.reshape(1, 3).astype(np.float32), E)
            depth_to_cam = float(pts_cam_origin[0, 2])
            if depth_to_cam <= 0:
                continue
            axis_len = np.clip(depth_to_cam * 0.08, 0.04, 0.15)

            endpoints = origin + R @ (np.eye(3) * axis_len)
            pts_world = np.vstack([origin.reshape(1, 3), endpoints])

            pts_2d, pts_cam = _project_world(pts_world, E, K, cam_model, dist)
            if pts_2d is None:
                continue

            o_px = pts_2d[0].astype(int)
            if not (-20 <= o_px[0] < w + 20 and -20 <= o_px[1] < h + 20):
                continue

            # Adaptive thickness based on projected axis pixel length
            avg_px_len = np.mean([np.linalg.norm(pts_2d[i + 1] - pts_2d[0]) for i in range(3)])
            thick = max(2, min(5, int(avg_px_len / 12)))
            tag_scale = max(0.30, min(0.50, avg_px_len / 100))

            # Draw axis arrows with outline for contrast
            for i in range(3):
                e_px = pts_2d[i + 1].astype(int)
                # Dark outline
                _draw_arrow(img, o_px, e_px, (0, 0, 0), thickness=thick + 2, tip_len=12)
                # Colored arrow
                _draw_arrow(img, o_px, e_px, AXIS_COLORS_BGR[i], thickness=thick, tip_len=12)
                # Axis label at tip
                label_pos = e_px + np.array([6, -4])
                _draw_tag(img, AXIS_LABELS[i], label_pos, AXIS_COLORS_BGR[i], scale=tag_scale)

            # Origin dot with object color ring
            cv2.circle(img, tuple(o_px), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(o_px), 5, obj_color, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(o_px), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Info label: name + pos + rpy
            short_name = oid.replace("obj_", "").replace("_", " ")
            euler = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
            label_lines = [
                (short_name, obj_color_rgb),
                (f"pos ({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f})", (230, 230, 230)),
                (f"rpy ({euler[0]:+.0f},{euler[1]:+.0f},{euler[2]:+.0f})", (190, 190, 190)),
            ]

            # Position label: connect with a thin leader line
            lx = int(o_px[0]) + 18
            ly = int(o_px[1]) + 12 + ci * 55  # stagger vertically per object
            lx = np.clip(lx, 4, w - 250)
            ly = np.clip(ly, 4, h - 60)
            # Leader line from origin to label
            cv2.line(img, tuple(o_px), (lx, ly + 8), obj_color, 1, cv2.LINE_AA)
            _draw_hud_panel(img, label_lines, lx, ly, scale=0.35, line_h=16, pad=5, bg_alpha=0.72)

        imgs.append(img)
        ids.append(cam)

    fig = _make_6view_figure(imgs, ids, f"Object Poses  |  frame {frame}  |  X(red)  Y(green)  Z(blue)")
    path = os.path.join(output_dir, f"objpose_frame_{frame:04d}.png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", help="HDF5 episode file")
    parser.add_argument("--frame", type=int, default=4, help="Frame index (default: 4)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <episode_dir>/vis_frame)")
    args = parser.parse_args()

    if not os.path.isfile(args.episode):
        print(f"Error: {args.episode} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.episode), "vis_frame")
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.episode, "r") as f:
        n_frames = f["time/sim_step"].shape[0]
        if args.frame >= n_frames:
            print(f"Error: frame {args.frame} out of range (0..{n_frames-1})", file=sys.stderr)
            sys.exit(1)

        print(f"Episode: {args.episode}  |  frame: {args.frame}/{n_frames-1}")
        for gen_fn in (gen_rgb, gen_depth, gen_joint_state, gen_objpose):
            p = gen_fn(f, args.frame, out_dir)
            print(f"  -> {p}")

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
