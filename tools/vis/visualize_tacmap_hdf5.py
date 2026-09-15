#!/usr/bin/env python3
r"""Visualize TacMap tactile streams stored in a dex2bench HDF5 episode.

Reads /robot/tactile/tacmap data from a replay HDF5, uniformly samples frames,
and writes one overview PNG per frame plus a summary.json with statistics.

Each overview PNG contains one TacMap heatmap per tactile site.

Usage:
    python tools/vis/visualize_tacmap_hdf5.py \
        --hdf5 /path/to/episode_replay.hdf5

    python tools/vis/visualize_tacmap_hdf5.py \
        --hdf5 /path/to/episode_replay.hdf5 \
        --output-dir /path/to/output_vis \
        --num-frames 16 \
        --dpi 160
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


DEFAULT_NUM_FRAMES = 8
DEFAULT_DPI = 120
DEFAULT_CMAP = "inferno"
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize /robot/tactile/tacmap data from a dex2bench HDF5 episode.")
    parser.add_argument("--hdf5", type=Path, required=True, help="Input replay HDF5 path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of uniformly sampled frames to visualize.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output PNG DPI.")
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP, help="Matplotlib colormap name.")
    return parser.parse_args()


def decode_hdf5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_hdf5_string(value.item())
    return str(value)


def read_string_list(dataset: h5py.Dataset) -> list[str]:
    return [decode_hdf5_string(value) for value in dataset[()]]


def resolve_output_dir(hdf5_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return hdf5_path.with_name(f"{hdf5_path.stem}_tacmap_vis")


def require_dataset(file: h5py.File, path: str) -> h5py.Dataset:
    dataset = file.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Missing dataset: {path}")
    return dataset


def read_sensor_type(file: h5py.File) -> str:
    dataset = require_dataset(file, "robot/tactile/meta/sensor_type")
    return decode_hdf5_string(dataset[()]).strip().lower()


def read_frame_count(file: h5py.File) -> int:
    if "meta/frame_count" in file:
        return int(file["meta/frame_count"][()])
    tacmap_group = file.get("robot/tactile/tacmap")
    if isinstance(tacmap_group, h5py.Group):
        for dataset in tacmap_group.values():
            if isinstance(dataset, h5py.Dataset):
                return int(dataset.shape[0])
    raise ValueError("Unable to infer frame count from HDF5.")


def read_site_names(file: h5py.File) -> list[str]:
    meta_sites = file.get("robot/tactile/meta/site_names")
    if isinstance(meta_sites, h5py.Dataset):
        return read_string_list(meta_sites)
    tacmap_group = file.get("robot/tactile/tacmap")
    if isinstance(tacmap_group, h5py.Group):
        return sorted(str(key) for key in tacmap_group.keys())
    raise ValueError("Missing /robot/tactile/tacmap group.")


def read_meta_value(file: h5py.File, path: str, default: Any = None) -> Any:
    dataset = file.get(path)
    if not isinstance(dataset, h5py.Dataset):
        return default
    value = dataset[()]
    if isinstance(value, (bytes, np.bytes_)):
        return decode_hdf5_string(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def select_frames(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    num = min(max(int(num_frames), 1), int(frame_count))
    frames = np.linspace(0, frame_count - 1, num=num, dtype=np.int64)
    return sorted({int(frame) for frame in frames})


def read_tacmap_frame(file: h5py.File, site_name: str, frame_idx: int) -> np.ndarray:
    dataset = require_dataset(file, f"robot/tactile/tacmap/{site_name}")
    frame = np.asarray(dataset[frame_idx], dtype=np.uint8)
    if frame.ndim != 2:
        raise ValueError(f"TacMap site '{site_name}' frame must be 2-D, got shape {frame.shape}.")
    return frame


def robust_limits(arrays: list[np.ndarray], *, lower_percentile: float = 1.0, upper_percentile: float = 99.5) -> tuple[float, float]:
    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        return 0.0, 1.0
    merged = np.concatenate(finite_values)
    lo = float(np.percentile(merged, lower_percentile))
    hi = float(np.percentile(merged, upper_percentile))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def compute_grid(n_items: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(max(1, n_items))))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def sample_arrays(file: h5py.File, site_names: list[str], frame_indices: list[int]) -> list[np.ndarray]:
    arrays = []
    for frame_idx in frame_indices:
        for site_name in site_names:
            arrays.append(read_tacmap_frame(file, site_name, frame_idx))
    return arrays


def build_frame_payload(file: h5py.File, site_names: list[str], frame_idx: int) -> dict[str, np.ndarray]:
    return {site_name: read_tacmap_frame(file, site_name, frame_idx) for site_name in site_names}


def plot_frame_overview(
    frame_payload: dict[str, np.ndarray],
    frame_idx: int,
    output_path: Path,
    *,
    dpi: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    site_names = list(frame_payload.keys())
    n_sites = len(site_names)
    n_rows, n_cols = compute_grid(n_sites)
    fig_width = max(10.0, 3.6 * n_cols)
    fig_height = max(6.0, 3.2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle(f"TacMap overview - frame {frame_idx}", fontsize=16)

    frame_stats = {
        "frame": int(frame_idx),
        "max_value": 0,
        "mean_value": 0.0,
        "sites": {},
    }

    site_means = []
    for site_idx, site_name in enumerate(site_names):
        row_idx = site_idx // n_cols
        col_idx = site_idx % n_cols
        ax = axes[row_idx, col_idx]
        image = np.asarray(frame_payload[site_name], dtype=np.uint8)
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(site_name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        site_max = int(np.max(image))
        site_mean = float(np.mean(image))
        site_means.append(site_mean)
        frame_stats["sites"][site_name] = {
            "max_value": site_max,
            "mean_value": site_mean,
        }
        frame_stats["max_value"] = max(frame_stats["max_value"], site_max)

    if site_means:
        frame_stats["mean_value"] = float(np.mean(site_means))

    for site_idx in range(n_sites, n_rows * n_cols):
        row_idx = site_idx // n_cols
        col_idx = site_idx % n_cols
        axes[row_idx, col_idx].axis("off")

    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.96))
    cax = fig.add_axes([0.18, 0.02, 0.64, 0.02])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, orientation="horizontal", label="TacMap uint8 value")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return frame_stats


def visualize(hdf5_path: Path, output_dir: Path, num_frames: int, dpi: int, cmap: str) -> dict[str, Any]:
    hdf5_path = hdf5_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as file:
        if "robot/tactile" not in file:
            raise ValueError("Missing /robot/tactile group in HDF5.")

        sensor_type = read_sensor_type(file)
        if sensor_type != "tacmap":
            raise ValueError(f"Expected sensor_type=tacmap, got {sensor_type!r}.")

        site_names = read_site_names(file)
        frame_count = read_frame_count(file)
        image_size = int(read_meta_value(file, "robot/tactile/meta/image_size", 0) or 0)
        resolution_step = int(read_meta_value(file, "robot/tactile/meta/resolution_step", 0) or 0)
        max_distance_m = float(read_meta_value(file, "robot/tactile/meta/max_distance_m", 0.0) or 0.0)
        robot_key = str(read_meta_value(file, "robot/tactile/meta/robot_key", "") or "")

        selected_frames = select_frames(frame_count, num_frames)
        vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX

        summary = {
            "hdf5": hdf5_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "sensor_type": sensor_type,
            "robot_key": robot_key,
            "frame_count": int(frame_count),
            "sampled_frames": selected_frames,
            "site_count": len(site_names),
            "site_names": site_names,
            "image_size": image_size,
            "resolution_step": resolution_step,
            "max_distance_m": max_distance_m,
            "colormap": cmap,
            "visualization_range": {"vmin": vmin, "vmax": vmax},
            "frames": [],
        }

        for frame_idx in selected_frames:
            frame_payload = build_frame_payload(file, site_names, frame_idx)
            image_path = output_dir / f"frame_{frame_idx:06d}.png"
            frame_stats = plot_frame_overview(
                frame_payload,
                frame_idx,
                image_path,
                dpi=dpi,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            frame_stats["image_path"] = image_path.as_posix()
            summary["frames"].append(frame_stats)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.hdf5, args.output_dir)
    summary = visualize(args.hdf5, output_dir, args.num_frames, args.dpi, args.cmap)
    print(f"[tacmap_vis] HDF5: {summary['hdf5']}")
    print(f"[tacmap_vis] Frame count: {summary['frame_count']}")
    print(f"[tacmap_vis] Output dir: {summary['output_dir']}")
    print(f"[tacmap_vis] Sampled frames: {summary['sampled_frames']}")
    print(f"[tacmap_vis] Sites: {summary['site_count']}")
    print(f"[tacmap_vis] Summary: {Path(summary['output_dir']) / 'summary.json'}")


if __name__ == "__main__":
    main()
