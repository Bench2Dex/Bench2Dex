#!/usr/bin/env python3
"""Export each camera in an HDF5 episode to a separate 2K MP4 video.

Usage:
    python export_cameras_2k.py <episode.hdf5> --out-dir <dir>
Output:
    <out-dir>/<episode_name>/
        cam_overhead.mp4
        cam_stereo_left.mp4
        ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

TARGET_W = 2560
TARGET_H = 1440
JPEG_QUALITY = 95  # for intermediate temp frames


def _decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG frame")
    return img  # BGR


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[..., ::-1].copy()


def _resize_to_2k(frame_bgr: np.ndarray) -> np.ndarray:
    """Resize frame to 2560x1440 with letterbox/pillarbox, preserving aspect ratio.
    Returns RGB uint8 array."""
    h, w = frame_bgr.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    y0 = (TARGET_H - new_h) // 2
    x0 = (TARGET_W - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return _to_rgb(canvas)


def export_cameras(hdf5_path: str, out_dir: str, fps: int | None = None) -> None:
    ep_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    ep_out = os.path.join(out_dir, ep_name)
    os.makedirs(ep_out, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        if "cameras" not in f:
            print(f"[SKIP] {hdf5_path}: no cameras group")
            return

        if fps is None:
            try:
                fps = int(f["meta/fps"][()])
            except Exception:
                fps = 20

        cam_ids = sorted(f["cameras"].keys())
        rgb_cams = [c for c in cam_ids if "rgb" in f["cameras"][c]]

        if not rgb_cams:
            print(f"[SKIP] {hdf5_path}: no RGB data in any camera")
            return

        frame_count = min(
            int(f["cameras"][c]["rgb"].shape[0]) for c in rgb_cams
        )

        print(f"[{ep_name}] {len(rgb_cams)} cameras, {frame_count} frames, {fps} fps")

        # Pre-load dataset handles
        ds_handles = {cam_id: f["cameras"][cam_id]["rgb"] for cam_id in rgb_cams}

        # Open imageio writers (ffmpeg H.264 — balanced for quality + smooth playback)
        writers = {}
        cam_paths = {}
        for cam_id in rgb_cams:
            mp4_path = os.path.join(ep_out, f"{cam_id}.mp4")
            writers[cam_id] = imageio.get_writer(
                mp4_path, fps=fps, codec="libx264",
                pixelformat="yuv420p",
                output_params=[
                    "-crf", "20",
                    "-preset", "medium",
                    "-tune", "film",
                    "-movflags", "+faststart",
                ],
            )
            cam_paths[cam_id] = mp4_path

        # Stream frames: decode → resize → write, one frame at a time
        for i in range(frame_count):
            for cam_id in rgb_cams:
                frame_bgr = _decode_frame(ds_handles[cam_id][i])
                frame_rgb = _resize_to_2k(frame_bgr)
                writers[cam_id].append_data(frame_rgb)

            if (i + 1) % 100 == 0 or i == frame_count - 1:
                print(f"  {i + 1}/{frame_count} frames", flush=True)

        for w in writers.values():
            w.close()

        # Print file sizes
        total_mb = 0
        for cam_id in rgb_cams:
            size_mb = os.path.getsize(cam_paths[cam_id]) / 1024 / 1024
            total_mb += size_mb
            print(f"  {cam_id}.mp4  {size_mb:.1f} MB")
        print(f"  total: {total_mb:.1f} MB")
        print(f"  → {ep_out}")
        print()


def main():
    p = argparse.ArgumentParser(description="Export each camera to a separate 2K MP4")
    p.add_argument("hdf5", nargs="+", help="One or more HDF5 episode files")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--fps", type=int, default=None, help="Override FPS")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for hdf5_path in args.hdf5:
        if not os.path.isfile(hdf5_path):
            print(f"[SKIP] not found: {hdf5_path}")
            continue
        export_cameras(hdf5_path, args.out_dir, args.fps)


if __name__ == "__main__":
    main()
