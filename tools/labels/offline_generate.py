#!/usr/bin/env python3
"""Generate all offline labels needed by the episode viewer in one pass.

This is a thin orchestrator around:
  - generate_box_labels.py for /labels/box3d and /labels/box2d
  - generate_occupancy_gt.py for /labels/occupancy_gt, including GPU shell
    voxelization and non-uniform asset scale handling

By default it writes labels in-place into the source HDF5 file.
"""
#  python tools/labels/offline_generate.py /path/to/episode.hdf5

#   如果已有这些 labels，需要覆盖：

#   python tools/labels/offline_generate.py /path/to/episode.hdf5 --overwrite

#   批量目录：

#   python tools/labels/offline_generate.py /path/to/hdf5_dir --workers 4 --overwrite
from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.config import normalize_occupancy_gt_config
from tools.labels._accel import parallel_process
from tools.labels._label_common import find_hdf5_files
from tools.labels.generate_box_labels import generate_box_labels_for_file
from tools.labels.generate_occupancy_gt import generate_occupancy_gt_for_file


def parse_bounds(raw: str) -> list[list[float]]:
    vals = [float(v) for v in raw.split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError(
            "--bounds requires exactly 6 comma-separated values: x0,y0,z0,x1,y1,z1"
        )
    return [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]


def build_occupancy_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "voxel_size": float(args.voxel_size),
        "geometry_source": args.geometry_source,
        "label_version": "occupancy_gt_v1",
    }
    if args.bounds is not None:
        config["bounds"] = args.bounds
    return normalize_occupancy_gt_config(config)


def offline_generate_for_file(
    hdf5_path: str,
    *,
    occupancy_config: dict[str, Any],
    write_mode: str = "inplace",
    overwrite: bool = False,
    scene_override: str | None = None,
    no_table: bool = False,
    no_semantic: bool = False,
    include_box2d: bool = True,
    skip_box: bool = False,
    skip_occupancy: bool = False,
    device: str = "auto",
) -> bool:
    """Generate box and occupancy labels for one HDF5 file.

    Returns True only when every requested label family was written.
    """
    ok = True

    if not skip_box:
        print(f"[INFO] {hdf5_path}: generating box3d{' + box2d' if include_box2d else ''}")
        box_ok = generate_box_labels_for_file(
            hdf5_path,
            box2d=include_box2d,
            write_mode=write_mode,
            overwrite=overwrite,
            device=device,
        )
        ok = ok and bool(box_ok)

    if not skip_occupancy:
        print(f"[INFO] {hdf5_path}: generating occupancy_gt")
        occupancy_ok = generate_occupancy_gt_for_file(
            hdf5_path,
            occupancy_config=occupancy_config,
            write_mode=write_mode,
            overwrite=overwrite,
            scene_override=scene_override,
            no_table=no_table,
            no_semantic=no_semantic,
            device=device,
        )
        ok = ok and bool(occupancy_ok)

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate box2d, box3d, and occupancy_gt labels for HDF5 episodes."
    )
    parser.add_argument("input", help="HDF5 file or directory to scan for *.hdf5")
    parser.add_argument(
        "--write-mode",
        choices=["inplace", "sidecar"],
        default="inplace",
        help="Write into source episode or derived sidecar (default: inplace)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated labels")
    parser.add_argument("--no-box2d", action="store_true", help="Only generate box3d, not box2d")
    parser.add_argument("--skip-box", action="store_true", help="Skip box3d/box2d generation")
    parser.add_argument("--skip-occupancy", action="store_true", help="Skip occupancy_gt generation")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Occupancy voxel size in meters")
    parser.add_argument(
        "--bounds",
        type=parse_bounds,
        default=None,
        help="Occupancy bounds as 'x0,y0,z0,x1,y1,z1'",
    )
    parser.add_argument("--scene", type=str, default=None, help="Override scene YAML path")
    parser.add_argument(
        "--geometry-source",
        choices=["collision", "visual"],
        default="collision",
        help="Mesh source for occupancy_gt voxelization",
    )
    parser.add_argument("--no-table", action="store_true", help="Do not include table in occupancy_gt")
    parser.add_argument("--no-semantic", action="store_true", help="Do not output occupancy semantic_id")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (default: 1; >1 forces CPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for box labels and occupancy_gt",
    )
    args = parser.parse_args()

    if args.skip_box and args.skip_occupancy:
        print("[ERROR] Nothing to do: both --skip-box and --skip-occupancy were set")
        return 2

    paths = find_hdf5_files(args.input)
    if not paths:
        print(f"[ERROR] No HDF5 files found at: {args.input}")
        return 2

    device_pref = args.device
    if args.workers > 1 and device_pref != "cpu":
        print("[INFO] workers > 1: forcing device=cpu (GPU contexts cannot be shared across processes)")
        device_pref = "cpu"

    occupancy_config = build_occupancy_config(args)
    box_label = "box3d" if args.no_box2d else "box3d+box2d"
    requested = []
    if not args.skip_box:
        requested.append(box_label)
    if not args.skip_occupancy:
        requested.append("occupancy_gt")
    print(
        f"[INFO] labels={'+'.join(requested)}, write_mode={args.write_mode}, "
        f"device={device_pref}, workers={args.workers}, "
        f"voxel_size={occupancy_config['voxel_size']}, bounds={occupancy_config['bounds']}"
    )

    process_one = functools.partial(
        offline_generate_for_file,
        occupancy_config=occupancy_config,
        write_mode=args.write_mode,
        overwrite=args.overwrite,
        scene_override=args.scene,
        no_table=args.no_table,
        no_semantic=args.no_semantic,
        include_box2d=not args.no_box2d,
        skip_box=args.skip_box,
        skip_occupancy=args.skip_occupancy,
        device=device_pref,
    )

    n_ok, n_err = parallel_process(process_one, paths, workers=args.workers, label="offline-labels")
    if n_err:
        return 1
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
