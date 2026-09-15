"""Export RGB videos from dex2scene HDF5 episodes."""
# python tools\\export_videos.py <hdf5文件路径或包含hdf5的目录>

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import h5py
import numpy as np


def _find_hdf5_files(path: str) -> list[str]:
    if os.path.isdir(path):
        out: list[str] = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.endswith(".hdf5"):
                    out.append(os.path.join(root, name))
        return sorted(out)
    if os.path.isfile(path):
        return [path]
    return []


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _get_fps(meta_group) -> int:
    if meta_group is None:
        return 30
    if "fps" not in meta_group:
        return 30
    try:
        return _safe_int(meta_group["fps"][()], 30)
    except Exception:
        return 30


def _iter_frames(ds) -> Iterable[np.ndarray]:
    for i in range(ds.shape[0]):
        frame = ds[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        yield frame


def _write_with_imageio(out_path: str, frames: Iterable[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_with_cv2(out_path: str, frames: Iterable[np.ndarray], fps: int, size: tuple[int, int]) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    for frame in frames:
        bgr = frame[..., ::-1]
        writer.write(bgr)
    writer.release()


def export_file(hdf5_path: str, out_dir: str | None = None) -> list[str]:
    outputs: list[str] = []
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(hdf5_path), "videos")
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        cameras = f.get("cameras")
        if cameras is None:
            print(f"[WARN] {hdf5_path}: no /cameras group")
            return outputs
        fps = _get_fps(f.get("meta"))

        for cam_id in cameras.keys():
            cam_grp = cameras[cam_id]
            if "rgb" not in cam_grp:
                print(f"[WARN] {hdf5_path}: camera '{cam_id}' has no rgb dataset")
                continue
            rgb = cam_grp["rgb"]
            if rgb.ndim != 4 or rgb.shape[-1] < 3:
                print(f"[WARN] {hdf5_path}: camera '{cam_id}' rgb shape={rgb.shape} not supported")
                continue

            h, w = int(rgb.shape[1]), int(rgb.shape[2])
            out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(hdf5_path))[0]}_{cam_id}.mp4")

            frames = _iter_frames(rgb)
            try:
                _write_with_imageio(out_path, frames, fps)
            except Exception:
                try:
                    frames = _iter_frames(rgb)
                    _write_with_cv2(out_path, frames, fps, (w, h))
                except Exception as exc:
                    print(f"[ERROR] {hdf5_path}: failed to write '{cam_id}' mp4: {exc}")
                    continue

            outputs.append(out_path)
            print(f"[INFO] Wrote {out_path}")

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RGB mp4 videos from dex2scene HDF5 episodes.")
    parser.add_argument(
        "input",
        help="HDF5 file or directory to scan for *.hdf5",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for mp4 files (default: <episode_dir>/videos)",
    )
    args = parser.parse_args()

    inputs = _find_hdf5_files(args.input)
    if not inputs:
        print(f"[ERROR] No HDF5 files found at: {args.input}")
        return 2

    wrote_any = False
    for path in inputs:
        outputs = export_file(path, out_dir=args.out_dir)
        if outputs:
            wrote_any = True

    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
