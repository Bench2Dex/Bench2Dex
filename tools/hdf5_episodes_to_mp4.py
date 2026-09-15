#!/usr/bin/env python3
"""Convert eval episode HDF5 recordings into a single tiled MP4 video.

All successful episodes are played first (sorted by episode number), then all
failed episodes.  Each episode is rendered as a 3×2 grid of the five cameras
with the episode number and success/failure status overlaid in the
bottom-right corner.

Usage:
    python tools/hdf5_episodes_to_mp4.py \\
        /path/to/output/eval/44/dp_all_0708_0803_schunk_hand/none/episodes \\
        --out /path/to/output.mp4 \\
        --fps 20
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

CAMERA_ORDER = [
    "cam_overhead",
    "cam_stereo_left",
    "cam_stereo_right",
    "cam_wrist_left",
    "cam_wrist_right",
]

# Grid layout: 3 cols × 2 rows
#   [ overhead,  stereo_left,  stereo_right ]
#   [ wrist_left, wrist_right, (empty)      ]
GRID_COLS = 3
GRID_ROWS = 2

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_SCALE = 0.7
OVERLAY_THICKNESS = 2
OVERLAY_COLOR_SUCCESS = (0, 200, 0)   # green BGR
OVERLAY_COLOR_FAILURE = (0, 0, 200)   # red BGR
OVERLAY_BG = (0, 0, 0, 160)  # semi-transparent black background


# ── helpers ──────────────────────────────────────────────────────────────────

def _episode_number(filename: str) -> int:
    m = re.search(r"episode_(\d+)", filename)
    return int(m.group(1)) if m else 0


def _decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode a single JPEG frame to a BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG frame")
    return img


def _draw_overlay(
    frame: np.ndarray, text: str, color: tuple[int, int, int]
) -> None:
    """Draw *text* with a semi-transparent background in the bottom-right corner."""
    h, w = frame.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, OVERLAY_FONT, OVERLAY_SCALE, OVERLAY_THICKNESS)

    # position: bottom-right with margin
    margin = 16
    x1 = w - tw - margin * 2
    y1 = h - th - margin * 2
    x2 = w - margin
    y2 = h - margin

    # semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_BG[:3], -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # text
    cv2.putText(
        frame, text,
        (x1 + margin // 2, y2 - margin // 2),
        OVERLAY_FONT, OVERLAY_SCALE, color, OVERLAY_THICKNESS, cv2.LINE_AA,
    )


def _read_camera_data(hf: h5py.File) -> dict[str, np.ndarray]:
    """Read all camera JPEG byte arrays from the HDF5 file."""
    jpeg_data: dict[str, np.ndarray] = {}
    for cam in CAMERA_ORDER:
        if cam not in hf["cameras"]:
            continue
        ds = hf["cameras"][cam]["rgb"]
        jpeg_data[cam] = ds[:]
    return jpeg_data


def _encode_episode(
    writer: cv2.VideoWriter,
    jpeg_data: dict[str, np.ndarray],
    cell_h: int,
    cell_w: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Write all frames of one episode to *writer* with tiled layout and overlay."""
    n_frames = min(len(d[1]) for d in jpeg_data.items())
    canvas = np.zeros((cell_h * GRID_ROWS, cell_w * GRID_COLS, 3), dtype=np.uint8)

    for i in range(n_frames):
        canvas.fill(0)

        # Decode and place each camera
        for ci, cam in enumerate(CAMERA_ORDER):
            if cam not in jpeg_data:
                continue
            row = ci // GRID_COLS
            col = ci % GRID_COLS
            img = _decode_frame(jpeg_data[cam][i])
            resized = cv2.resize(img, (cell_w, cell_h))
            canvas[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized

        _draw_overlay(canvas, label, color)
        writer.write(canvas)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Convert eval HDF5 episodes to a tiled MP4")
    p.add_argument("episodes_dir", type=Path,
                   help="Path to the 'episodes/' directory containing success/ and failure/ subdirs")
    p.add_argument("--out", type=Path, required=True,
                   help="Output MP4 file path")
    p.add_argument("--fps", type=int, default=20,
                   help="Output video FPS (default: 20)")
    p.add_argument("--cell-w", type=int, default=360,
                   help="Width of each camera cell in the grid (default: 360)")
    p.add_argument("--cell-h", type=int, default=240,
                   help="Height of each camera cell in the grid (default: 240)")
    args = p.parse_args()

    episodes_dir: Path = args.episodes_dir
    if not episodes_dir.is_dir():
        sys.exit(f"Not a directory: {episodes_dir}")

    success_dir = episodes_dir / "success"
    failure_dir = episodes_dir / "failure"

    # Collect episodes
    episodes: list[tuple[int, bool, Path]] = []  # (ep_num, is_success, path)

    for is_success, subdir in [(True, success_dir), (False, failure_dir)]:
        if not subdir.is_dir():
            continue
        for p in sorted(subdir.glob("episode_*.hdf5")):
            ep_num = _episode_number(p.name)
            episodes.append((ep_num, is_success, p))

    # Sort: successes first (by ep num), then failures (by ep num)
    episodes.sort(key=lambda x: (not x[1], x[0]))

    if not episodes:
        sys.exit(f"No episode_*.hdf5 files found under {episodes_dir}")

    n_success = sum(1 for _, s, _ in episodes if s)
    n_failure = sum(1 for _, s, _ in episodes if not s)

    print(f"Found {len(episodes)} episodes ({n_success} success, {n_failure} failure)")
    print(f"Order: all successes first, then all failures")

    # Determine cell dimensions from first episode's first camera frame
    first_ep = episodes[0][2]
    with h5py.File(first_ep, "r") as hf:
        first_cam = next(c for c in CAMERA_ORDER if c in hf["cameras"])
        first_frame = _decode_frame(hf["cameras"][first_cam]["rgb"][0])
    native_h, native_w = first_frame.shape[:2]
    if args.cell_w == 360 and args.cell_h == 240:
        # Auto-scale: maintain aspect ratio with cell_w=360
        scale = 360.0 / native_w
        cell_w, cell_h = 360, int(native_h * scale)
    else:
        cell_w, cell_h = args.cell_w, args.cell_h

    grid_w = cell_w * GRID_COLS
    grid_h = cell_h * GRID_ROWS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (grid_w, grid_h))

    total_frames = 0
    for idx, (ep_num, is_success, ep_path) in enumerate(episodes):
        status_text = "SUCCESS" if is_success else "FAILURE"
        color = OVERLAY_COLOR_SUCCESS if is_success else OVERLAY_COLOR_FAILURE
        label = f"Ep {ep_num}  [{status_text}]"

        print(f"[{idx + 1}/{len(episodes)}] Episode {ep_num} ({status_text}) ...", end=" ", flush=True)

        with h5py.File(ep_path, "r") as hf:
            jpeg_data = _read_camera_data(hf)
            n_frames = min(len(v) for v in jpeg_data.values())
            _encode_episode(writer, jpeg_data, cell_h, cell_w, label, color)

        total_frames += n_frames
        print(f"{n_frames} frames")

    writer.release()

    size_mb = args.out.stat().st_size / 1024 / 1024
    duration_s = total_frames / args.fps
    print(f"\nDone: {args.out}")
    print(f"  Episodes: {len(episodes)}  |  Frames: {total_frames}  |  "
          f"Duration: {int(duration_s // 60)}m {int(duration_s % 60)}s  |  "
          f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
