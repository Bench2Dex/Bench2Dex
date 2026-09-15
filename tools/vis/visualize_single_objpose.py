#!/usr/bin/env python3
"""Visualize a single object's pose on a single camera view.

Usage:
    python tools/visualize_single_objpose.py <episode.hdf5> \\
        --camera cam_stereo_right --object obj_104_microwave_2 --frame 4

    python tools/visualize_single_objpose.py <episode.hdf5> \\
        --camera cam_stereo_right --object obj_104_microwave_2 \\
        --axis-len 0.2 --thickness 3
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collector.camera_geometry import (
    world_points_to_camera,
    project_camera_points,
    project_camera_points_fisheye,
)

# Okabe-Ito accent color (orange) for object highlight
ACCENT_BGR = (0, 159, 230)
ACCENT_RGB = (230, 159, 0)

AXIS_BGR = [(30, 30, 240), (30, 200, 30), (240, 140, 30)]  # X=red Y=green Z=blue
AXIS_LABEL = ["X", "Y", "Z"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _rounded_rect(img, x1, y1, x2, y2, color, radius=6, alpha=0.78):
    """Semi-transparent rounded rectangle."""
    ov = img.copy()
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(ov, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(ov, (cx, cy), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _draw_axis_tag(img, text, pos, color, scale=0.55):
    """Axis letter: white outline + colored fill, with circular background."""
    (tw, th), bl = cv2.getTextSize(text, FONT_BOLD, scale, 2)
    x, y = int(pos[0]), int(pos[1])
    cx, cy = x + tw // 2, y - th // 2
    radius = max(tw, th) // 2 + 6
    # Circle background
    _rounded_rect(img, cx - radius, cy - radius, cx + radius, cy + radius, (15, 15, 15), radius=radius, alpha=0.70)
    # White outline text
    cv2.putText(img, text, (x, y), FONT_BOLD, scale, (255, 255, 255), 3, cv2.LINE_AA)
    # Colored fill text
    cv2.putText(img, text, (x, y), FONT_BOLD, scale, color, 2, cv2.LINE_AA)


def _find_label_pos(o_px, w, h, box_w, box_h):
    """Find best label position: try right-above first, then other positions."""
    candidates = [
        (o_px[0] + 25, o_px[1] - box_h - 15),  # right-above (preferred)
        (o_px[0] + 25, o_px[1] + 20),           # right-below
        (o_px[0] - box_w - 25, o_px[1] - box_h - 15),  # left-above
        (o_px[0] - box_w - 25, o_px[1] + 20),   # left-below
    ]
    for lx, ly in candidates:
        if 5 <= lx <= w - box_w - 5 and 5 <= ly <= h - box_h - 5:
            return int(lx), int(ly)
    # Fallback: clamp first candidate
    lx, ly = candidates[0]
    lx = max(5, min(lx, w - box_w - 5))
    ly = max(5, min(ly, h - box_h - 5))
    return int(lx), int(ly)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", help="HDF5 episode file")
    parser.add_argument("--camera", default="cam_stereo_right")
    parser.add_argument("--object", default="obj_104_microwave_2")
    parser.add_argument("--frame", type=int, default=4)
    parser.add_argument("--axis-len", type=float, default=0.12, help="Axis arrow length in meters")
    parser.add_argument("--thickness", type=int, default=3, help="Axis line thickness")
    parser.add_argument("--output", default=None, help="Output PNG path")
    args = parser.parse_args()

    with h5py.File(args.episode, "r") as f:
        g = f[f"cameras/{args.camera}"]
        rgb = g["rgb"][args.frame].copy()
        K = g["intrinsic"][:]
        E = g["extrinsic_world_from_cam"][args.frame]
        raw = g["camera_model"][()]
        cam_model = raw.decode() if isinstance(raw, bytes) else str(raw)
        dist = None
        if cam_model == "fisheye":
            K = g["fisheye_camera_matrix"][:]
            dist = g["distortion_coefficients"][:]
        pose = f[f"objects/{args.object}/pose_world"][args.frame]

    origin = pose[:3]
    quat_xyzw = pose[3:]
    R_obj = Rotation.from_quat(quat_xyzw).as_matrix()
    endpoints = origin + R_obj @ (np.eye(3) * args.axis_len)
    pts_world = np.vstack([origin.reshape(1, 3), endpoints]).astype(np.float32)

    pts_cam, valid = world_points_to_camera(pts_world, E)
    if not np.all(valid):
        print("Warning: some points behind camera")
    if cam_model == "fisheye" and dist is not None:
        pts_2d = project_camera_points_fisheye(pts_cam, K, dist)
    else:
        pts_2d = project_camera_points(pts_cam, K)

    o_px = pts_2d[0].astype(int)
    h, w = rgb.shape[:2]
    tk = args.thickness

    # --- Draw axes ---
    for i in range(3):
        e_px = pts_2d[i + 1].astype(int)
        direction = e_px - o_px
        norm = np.linalg.norm(direction)
        if norm < 2:
            continue

        # Shaft (line without arrowhead)
        shaft_end = o_px + (direction * 0.78).astype(int)

        # Layer 1: glow (wide, semi-transparent)
        glow_color = tuple(min(255, c + 60) for c in AXIS_BGR[i])
        ov = rgb.copy()
        cv2.line(ov, tuple(o_px), tuple(shaft_end), glow_color, tk + 5, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.25, rgb, 0.75, 0, rgb)

        # Layer 2: dark outline
        cv2.arrowedLine(rgb, tuple(o_px), tuple(e_px), (0, 0, 0), tk + 3, cv2.LINE_AA, tipLength=0.18)

        # Layer 3: colored arrow
        cv2.arrowedLine(rgb, tuple(o_px), tuple(e_px), AXIS_BGR[i], tk, cv2.LINE_AA, tipLength=0.18)

        # Layer 4: bright highlight along shaft center
        highlight = tuple(min(255, c + 100) for c in AXIS_BGR[i])
        cv2.line(rgb, tuple(o_px), tuple(shaft_end), highlight, max(1, tk - 2), cv2.LINE_AA)

    # Axis labels at tips
    for i in range(3):
        e_px = pts_2d[i + 1].astype(int)
        direction = (e_px - o_px).astype(float)
        norm = np.linalg.norm(direction)
        if norm < 2:
            continue
        offset = (direction / norm * 16).astype(int)
        _draw_axis_tag(rgb, AXIS_LABEL[i], e_px + offset, AXIS_BGR[i], scale=0.60)

    # --- Origin marker ---
    cv2.circle(rgb, tuple(o_px), 9, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(rgb, tuple(o_px), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(rgb, tuple(o_px), 5, ACCENT_BGR, -1, cv2.LINE_AA)

    # --- Info card ---
    short_name = args.object.replace("obj_", "").replace("_", " ")
    euler = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)

    title = short_name.upper()
    row_pos = f"pos  {pose[0]:+.3f}  {pose[1]:+.3f}  {pose[2]:+.3f}"
    row_rpy = f"rpy  {euler[0]:+.1f}   {euler[1]:+.1f}   {euler[2]:+.1f}"

    sc_title = 0.45
    sc_data = 0.40
    lh_title = 22
    lh_data = 18
    pad = 8

    # Measure card size
    tw_title = cv2.getTextSize(title, FONT, sc_title, 1)[0][0]
    tw_pos = cv2.getTextSize(row_pos, FONT, sc_data, 1)[0][0]
    tw_rpy = cv2.getTextSize(row_rpy, FONT, sc_data, 1)[0][0]
    card_w = max(tw_title, tw_pos, tw_rpy) + pad * 2 + 4
    card_h = lh_title + lh_data * 2 + pad * 2 + 6  # title + 2 rows + padding + accent bar

    # Find best position
    lx, ly = _find_label_pos(o_px, w, h, card_w, card_h)

    # Leader line: curved elbow
    mid_x = (o_px[0] + lx) // 2
    cv2.line(rgb, tuple(o_px), (mid_x, o_px[1]), ACCENT_BGR, 1, cv2.LINE_AA)
    cv2.line(rgb, (mid_x, o_px[1]), (lx, ly + card_h // 2), ACCENT_BGR, 1, cv2.LINE_AA)
    cv2.circle(rgb, (mid_x, o_px[1]), 2, ACCENT_BGR, -1, cv2.LINE_AA)  # elbow dot

    # Card background
    _rounded_rect(rgb, lx, ly, lx + card_w, ly + card_h, (25, 25, 25), radius=6, alpha=0.80)

    # Accent bar on left edge
    cv2.rectangle(rgb, (lx + 2, ly + 4), (lx + 5, ly + card_h - 4), ACCENT_BGR, -1)

    # Title
    tx = lx + pad + 6
    ty = ly + pad + 14
    cv2.putText(rgb, title, (tx, ty), FONT, sc_title, ACCENT_RGB, 1, cv2.LINE_AA)

    # Separator line
    sep_y = ty + 6
    cv2.line(rgb, (lx + pad, sep_y), (lx + card_w - pad, sep_y), (60, 60, 60), 1, cv2.LINE_AA)

    # Data rows
    dy = sep_y + lh_data
    # pos row with dim labels
    cv2.putText(rgb, "pos", (tx, dy), FONT, sc_data, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(rgb, f" {pose[0]:+.3f}  {pose[1]:+.3f}  {pose[2]:+.3f}", (tx + 28, dy), FONT, sc_data, (230, 230, 230), 1, cv2.LINE_AA)
    dy += lh_data
    cv2.putText(rgb, "rpy", (tx, dy), FONT, sc_data, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(rgb, f" {euler[0]:+.1f}  {euler[1]:+.1f}  {euler[2]:+.1f}", (tx + 28, dy), FONT, sc_data, (200, 200, 200), 1, cv2.LINE_AA)

    # Save
    out = args.output or os.path.join(
        os.path.dirname(args.episode),
        "vis_frame",
        f"objpose_{args.camera}_{args.object}_frame_{args.frame:04d}.png",
    )
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"-> {out}  ({w}x{h})")


if __name__ == "__main__":
    main()
