"""Export a multi-camera HDF5 episode to a tiled MP4 preview.

用法:
    python ./tools/export/export_video.py ./outputs/dex2scene_dataset/scenes/06_fruit_bowl_loading/episode_000025.hdf5 --output video.mp4 --fps 20

输出: 2×3 网格视频, 6 个相机视角时间对齐。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable

import h5py
import numpy as np


PREFERRED_ORDER = [
    "cam_chest",
    "cam_overhead",
    "cam_wrist_right",
    "cam_wrist_left",
    "cam_stereo_left",
    "cam_stereo_right",
]


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_fps(meta_group, cli_fps: int | None) -> int:
    if cli_fps is not None and cli_fps > 0:
        return int(cli_fps)
    if meta_group is None or "fps" not in meta_group:
        return 30
    try:
        return _safe_int(meta_group["fps"][()], 30)
    except Exception:
        return 30


def _ordered_camera_ids(camera_group: h5py.Group) -> list[str]:
    cam_ids = list(camera_group.keys())
    ordered = [cam_id for cam_id in PREFERRED_ORDER if cam_id in cam_ids]
    ordered.extend(cam_id for cam_id in cam_ids if cam_id not in ordered)
    return ordered


def _grid_shape(camera_count: int) -> tuple[int, int]:
    if camera_count <= 0:
        return 0, 0
    if camera_count <= 3:
        return 1, camera_count
    if camera_count <= 6:
        return 2, 3
    cols = int(math.ceil(math.sqrt(camera_count)))
    rows = int(math.ceil(camera_count / cols))
    return rows, cols


def _fit_rgb_to_cell(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    import cv2

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if src_h <= 0 or src_w <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    scale = min(cell_w / src_w, cell_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _read_rgb_frame(rgb_dataset: h5py.Dataset, frame_idx: int) -> np.ndarray:
    import cv2

    raw = rgb_dataset[frame_idx]
    rgb = np.asarray(raw)
    if rgb.ndim == 1 and rgb.dtype == np.uint8:
        if rgb.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        decoded_bgr = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 2:
        return np.repeat(rgb[..., None], 3, axis=2)
    if rgb.ndim == 3:
        if rgb.shape[2] >= 3:
            return rgb[..., :3]
        return np.repeat(rgb[..., :1], 3, axis=2)
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _iter_export_indices(file: h5py.File, frame_count: int, skip_frames: int, valid_only: bool) -> list[int]:
    start = min(max(0, int(skip_frames)), frame_count)
    indices = list(range(start, frame_count))
    if not valid_only or "frame_valid" not in file:
        return indices
    frame_valid = np.asarray(file["frame_valid"][:], dtype=np.bool_)
    return [idx for idx in indices if idx < len(frame_valid) and bool(frame_valid[idx])]


def _iter_grid_frames(
    file: h5py.File,
    ordered_cam_ids: list[str],
    frame_indices: list[int],
    cell_w: int,
    cell_h: int,
) -> Iterable[np.ndarray]:
    import cv2

    rows, cols = _grid_shape(len(ordered_cam_ids))
    canvas_h = rows * cell_h
    canvas_w = cols * cell_w
    camera_groups = {cam_id: file[f"cameras/{cam_id}"] for cam_id in ordered_cam_ids}

    for frame_idx in frame_indices:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for cam_pos, cam_id in enumerate(ordered_cam_ids):
            cam_grp = camera_groups[cam_id]
            if "rgb" not in cam_grp:
                continue

            rgb = _read_rgb_frame(cam_grp["rgb"], frame_idx)
            tile = _fit_rgb_to_cell(rgb, cell_w, cell_h)
            tile_bgr = tile[..., ::-1].copy()

            label = cam_id.replace("cam_", "").upper()
            cv2.putText(
                tile_bgr,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            row = cam_pos // cols
            col = cam_pos % cols
            y0 = row * cell_h
            x0 = col * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = tile_bgr
        yield canvas


def _write_with_imageio(out_path: str, frames: Iterable[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame[..., ::-1])


def _write_with_cv2(out_path: str, frames: Iterable[np.ndarray], fps: int, size: tuple[int, int]) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def export_video(
    hdf5_path: str,
    output_path: str,
    fps: int | None = None,
    skip_frames: int = 0,
    valid_only: bool = False,
    cell_width: int = 640,
    cell_height: int = 480,
) -> None:
    with h5py.File(hdf5_path, "r") as f:
        cameras = f.get("cameras")
        if cameras is None:
            print(f"[ERROR] {hdf5_path}: no /cameras group")
            return

        ordered = _ordered_camera_ids(cameras)
        ordered = [cam_id for cam_id in ordered if "rgb" in cameras[cam_id]]
        if not ordered:
            print(f"[ERROR] {hdf5_path}: no camera RGB data found")
            return

        frame_count = min(int(cameras[cam_id]["rgb"].shape[0]) for cam_id in ordered)
        export_indices = _iter_export_indices(f, frame_count, skip_frames=skip_frames, valid_only=valid_only)
        if not export_indices:
            print(f"[ERROR] {hdf5_path}: no frames selected for export")
            return

        resolved_fps = _get_fps(f.get("meta"), fps)
        rows, cols = _grid_shape(len(ordered))
        canvas_size = (cols * cell_width, rows * cell_height)

        print(f"Found cameras: {ordered}")
        print(f"Selected frames: {len(export_indices)}/{frame_count}, fps={resolved_fps}, grid={rows}x{cols}, canvas={canvas_size[0]}x{canvas_size[1]}")

        frames = _iter_grid_frames(f, ordered, export_indices, cell_width, cell_height)
        try:
            _write_with_cv2(output_path, frames, resolved_fps, canvas_size)
        except Exception as exc:
            print(f"[WARN] OpenCV export failed, falling back to imageio: {exc}")
            frames = _iter_grid_frames(f, ordered, export_indices, cell_width, cell_height)
            _write_with_imageio(output_path, frames, resolved_fps)

    print(f"[INFO] Video saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export multi-camera HDF5 episode to an MP4 grid video.")
    parser.add_argument("hdf5", help="Path to episode HDF5 file.")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 path. Default: <episode>_grid.mp4")
    parser.add_argument("--fps", type=int, default=None, help="Override output FPS. Default: use meta/fps from HDF5.")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip the first N frames before writing the video.")
    parser.add_argument("--valid-only", action="store_true", help="Export only frames where /frame_valid is True.")
    parser.add_argument("--cell-width", type=int, default=640, help="Grid cell width in pixels.")
    parser.add_argument("--cell-height", type=int, default=480, help="Grid cell height in pixels.")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        print(f"File not found: {args.hdf5}")
        return 1

    output_path = args.output or (os.path.splitext(args.hdf5)[0] + "_grid.mp4")
    export_video(
        args.hdf5,
        output_path,
        fps=args.fps,
        skip_frames=args.skip_frames,
        valid_only=args.valid_only,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
