"""Generate occupancy labels from stored HDF5 episodes."""

from __future__ import annotations

import argparse
import functools
import os
import sys
import warnings
from copy import deepcopy
from typing import Any

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.labels._label_common import (
    compute_default_occupancy_bounds,
    decode_hdf5_string,
    default_sidecar_path,
    find_hdf5_files,
    prepare_label_slot,
    read_camera_frames,
    read_frame_metadata,
    read_local_bboxes,
    read_object_states,
    resolve_scene_path,
    sync_sidecar,
)
from tools.labels._accel import read_episode_batch, get_device, parallel_process
from collector.config import DEFAULT_OCCUPANCY_CONFIG, normalize_occupancy_config
from collector.episode_buffer import EpisodeBuffer, FrameRecord
from collector.hdf5_writer import HDF5EpisodeWriter
from collector.occupancy import OccupancyLabeler

# Fisheye wrist cameras are excluded by default: their wide-FOV depth maps
# cause systematic occupancy artifacts due to depth-semantics ambiguity at
# extreme angles.  Use --exclude-cameras '' to override.
DEFAULT_EXCLUDE_CAMERAS = {"cam_wrist_left", "cam_wrist_right"}

OFFLINE_OCCUPANCY_DEFAULTS = {
    **deepcopy(DEFAULT_OCCUPANCY_CONFIG),
    "voxel_size": 0.01,
    "max_voxel_count": 0,
    "require_all_cameras": True,
    "fusion_mode": "tsdf",
    "bounds": [[-1.15, -0.6, 0.74], [1.15, 0.6, 1.10]],
}


def _normalize_offline_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(OFFLINE_OCCUPANCY_DEFAULTS)
    if config:
        merged.update(config)
    return normalize_occupancy_config(merged)


def _source_has_depth_maps(file: h5py.File) -> bool:
    cameras = file.get("cameras")
    if cameras is None:
        return False
    return any(
        ("depth_m" in cameras[cam_id] or "depth" in cameras[cam_id])
        for cam_id in cameras.keys()
    )


def _build_label_buffer(
    source_file: h5py.File,
    frame_valid: np.ndarray,
    frame_errors: list[str],
    sim_steps: np.ndarray,
    camera_ids: list[str],
    object_ids: list[str],
    occupancy: OccupancyLabeler,
    exclude_cameras: set[str] | None = None,
) -> EpisodeBuffer:
    buffer = EpisodeBuffer(camera_ids=camera_ids, object_ids=object_ids)
    frame_count = int(frame_valid.shape[0])
    for frame_index in range(frame_count):
        if not bool(frame_valid[frame_index]):
            buffer.add_frame(
                FrameRecord(
                    frame_index=frame_index,
                    timestamp_ns=0,
                    sim_step=int(sim_steps[frame_index]),
                    valid=False,
                    errors=[str(frame_errors[frame_index])],
                    camera={},
                    robot=None,
                    objects={},
                    labels={"occupancy": {}},
                )
            )
            continue

        camera_frames = read_camera_frames(source_file, frame_index)
        if exclude_cameras:
            camera_frames = {k: v for k, v in camera_frames.items() if k not in exclude_cameras}
        object_states = read_object_states(source_file, frame_index)
        occupancy_labels = occupancy.on_step(
            object_states=object_states,
            camera_frames=camera_frames,
            sim_step=frame_index,
        )
        buffer.add_frame(
            FrameRecord(
                frame_index=frame_index,
                timestamp_ns=0,
                sim_step=int(sim_steps[frame_index]),
                valid=True,
                errors=[],
                camera={},
                robot=None,
                objects={},
                labels={"occupancy": occupancy_labels},
            )
        )
    return buffer


def _try_gpu_tsdf(
    f: h5py.File,
    hdf5_path: str,
    frame_valid: np.ndarray,
    frame_errors: list[str],
    sim_steps: np.ndarray,
    camera_ids: list[str],
    object_ids: list[str],
    occupancy_config: dict[str, Any],
    exclude_cameras: set[str] | None,
    device_pref: str,
) -> EpisodeBuffer | None:
    """Attempt GPU TSDF fusion. Returns None if GPU unavailable or fails."""
    try:
        dev = get_device(device_pref)
        if dev.type == "cpu":
            return None  # no GPU available, use CPU path

        from tools.labels._gpu_tsdf import TSDFFuser

        bounds = np.asarray(occupancy_config["bounds"], dtype=np.float32)
        voxel_size = float(occupancy_config["voxel_size"])
        mu = float(occupancy_config.get("truncation_distance", 3.0 * voxel_size))
        threshold = float(occupancy_config.get("surface_threshold", 0.5 * voxel_size))

        batch_data = read_episode_batch(f, need_depth=True, exclude_cameras=exclude_cameras)
        exclude = exclude_cameras or set()
        usable_cams = [c for c in batch_data.camera_static if c not in exclude]

        fuser = TSDFFuser(bounds, voxel_size, dev, mu, threshold)

        buffer = EpisodeBuffer(camera_ids=camera_ids, object_ids=object_ids)
        frame_count = int(frame_valid.shape[0])

        for fi in range(frame_count):
            if not bool(frame_valid[fi]):
                buffer.add_frame(FrameRecord(
                    frame_index=fi, timestamp_ns=0, sim_step=int(sim_steps[fi]),
                    valid=False, errors=[str(frame_errors[fi])],
                    camera={}, robot=None, objects={}, labels={"occupancy": {}},
                ))
                continue

            # Prepare per-frame data for this frame
            cam_depths = {}
            cam_extrs = {}
            for cam_id in usable_cams:
                if batch_data.depth_maps and cam_id in batch_data.depth_maps:
                    depth = batch_data.depth_maps[cam_id][fi]
                    if np.any(np.isfinite(depth)):
                        cam_depths[cam_id] = depth
                        cam_extrs[cam_id] = batch_data.extrinsics[cam_id][fi]

            if not cam_depths:
                buffer.add_frame(FrameRecord(
                    frame_index=fi, timestamp_ns=0, sim_step=int(sim_steps[fi]),
                    valid=True, errors=[], camera={}, robot=None, objects={},
                    labels={"occupancy": {}},
                ))
                continue

            fuser.fuse_frame(cam_depths, cam_extrs, batch_data.camera_static)
            state_grid = fuser.classify()
            fuser.reset()

            occupancy_labels = {
                "grid_shape": np.asarray(fuser.grid_shape, dtype=np.int32),
                "voxel_size": fuser.voxel_size,
                "bounds": fuser.bounds.astype(np.float32),
                "state": state_grid,
            }
            buffer.add_frame(FrameRecord(
                frame_index=fi, timestamp_ns=0, sim_step=int(sim_steps[fi]),
                valid=True, errors=[], camera={}, robot=None, objects={},
                labels={"occupancy": occupancy_labels},
            ))

        return buffer
    except Exception as exc:
        warnings.warn(f"{hdf5_path}: GPU TSDF failed ({exc}), falling back to CPU")
        return None


# ---------------------------------------------------------------------------
# Semantic ID assignment via object AABBs
# ---------------------------------------------------------------------------

_OCCUPIED = 2

_CORNER_SIGNS = np.array([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float32)


def _assign_semantic_ids(
    hdf5_path: str,
    target_path: str,
    label_group: str = "labels/occupancy_tsdf",
) -> None:
    """Assign per-voxel semantic_id to TSDF occupancy using object AABBs.

    Reads object poses and local bounding boxes from the source episode,
    computes world-frame AABBs per frame, and labels each OCCUPIED voxel
    with the semantic_id of the containing object (or 1=table if no object
    matches).

    Writes ``semantic_id``, ``semantic_legend``, and ``frame_valid``
    datasets into the existing occupancy group.
    """
    from collector.camera_geometry import quat_xyzw_to_rot

    with h5py.File(hdf5_path, "r") as src:
        local_bboxes = read_local_bboxes(src)
        object_ids = sorted(local_bboxes.keys())
        frame_valid_src, _, _, _, _ = read_frame_metadata(src)

        # Build semantic legend: 0=free, 1=table, 2+=objects
        obj_to_sid: dict[str, int] = {}
        legend: dict[int, str] = {0: "free", 1: "table"}
        for i, obj_id in enumerate(object_ids):
            sid = i + 2
            obj_to_sid[obj_id] = sid
            legend[sid] = obj_id

        # Read all object poses
        all_poses: dict[str, np.ndarray] = {}
        for obj_id in object_ids:
            grp = src.get(f"objects/{obj_id}")
            if grp is None:
                continue
            for key in ("pose_world", "pose"):
                if key in grp:
                    all_poses[obj_id] = grp[key][()]
                    break

    # Read occupancy state and assign semantics
    with h5py.File(target_path, "r+") as tf:
        occ_grp = tf[label_group]
        state_ds = occ_grp["state"]
        bounds = np.asarray(occ_grp["bounds"][()], dtype=np.float32)
        voxel_size = float(occ_grp["voxel_size"][()])
        grid_shape = tuple(occ_grp["grid_shape"][()].tolist())
        n_frames = state_ds.shape[0]

        # Pre-compute voxel centers once
        nx, ny, nz = grid_shape
        xs = bounds[0, 0] + (np.arange(nx) + 0.5) * voxel_size
        ys = bounds[0, 1] + (np.arange(ny) + 0.5) * voxel_size
        zs = bounds[0, 2] + (np.arange(nz) + 0.5) * voxel_size
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

        # Create semantic_id dataset
        if "semantic_id" in occ_grp:
            del occ_grp["semantic_id"]
        sem_ds = occ_grp.create_dataset(
            "semantic_id",
            shape=(n_frames, *grid_shape),
            dtype=np.uint16,
            chunks=(1, *grid_shape),
            compression="gzip",
            compression_opts=4,
        )

        for fi in range(n_frames):
            if not bool(frame_valid_src[fi]):
                continue

            state_frame = state_ds[fi]
            occ_mask = (state_frame.ravel() == _OCCUPIED)
            if not np.any(occ_mask):
                continue

            sem_flat = np.ones(occ_mask.shape, dtype=np.uint16)  # default: 1=table

            # Compute per-object world AABBs and assign
            for obj_id in object_ids:
                if obj_id not in all_poses:
                    continue
                pose = all_poses[obj_id][fi]
                if not np.all(np.isfinite(pose)):
                    continue

                lo, hi = local_bboxes[obj_id]
                lo_arr = np.asarray(lo, dtype=np.float32)
                hi_arr = np.asarray(hi, dtype=np.float32)
                size = hi_arr - lo_arr
                center_local = 0.5 * (lo_arr + hi_arr)
                corners_local = _CORNER_SIGNS * (0.5 * size)
                rot = quat_xyzw_to_rot(pose[3:7])
                center_world = pose[:3] + rot @ center_local
                corners_world = (corners_local @ rot.T) + center_world
                aabb_min = corners_world.min(axis=0)
                aabb_max = corners_world.max(axis=0)

                in_aabb = occ_mask & np.all(
                    (centers >= aabb_min) & (centers <= aabb_max), axis=1
                )
                sem_flat[in_aabb] = obj_to_sid[obj_id]

            # Voxels that are not occupied get sem=0
            sem_flat[~occ_mask] = 0
            sem_ds[fi] = sem_flat.reshape(grid_shape)

        # Write legend
        str_dtype = h5py.string_dtype(encoding="utf-8")
        if "semantic_legend" in occ_grp:
            del occ_grp["semantic_legend"]
        import json
        occ_grp.create_dataset(
            "semantic_legend",
            data=np.asarray(json.dumps({str(k): v for k, v in legend.items()}), dtype=str_dtype),
        )


def generate_occupancy_for_file(
    hdf5_path: str,
    *,
    occupancy_config: dict[str, Any],
    write_mode: str = "sidecar",
    overwrite: bool = False,
    output_path: str | None = None,
    exclude_cameras: set[str] | None = None,
    device: str = "auto",
    bounds_override: list[list[float]] | None = None,
) -> bool:
    """Generate occupancy labels for a single episode.

    Returns True if labels were written, False if skipped.

    When ``bounds_override`` is None (no explicit ``--bounds``), the occupancy
    bounds are derived per-episode from the actual object-table geometry,
    covering only the object table (excluding the rear robot-support table)
    with the floor pinned to the real tabletop height.
    """
    occupancy_config = deepcopy(occupancy_config)

    with h5py.File(hdf5_path, "r") as f:
        if not _source_has_depth_maps(f):
            raise ValueError(f"{hdf5_path}: occupancy labels require /cameras/*/depth_m in the source episode")

        frame_valid, frame_errors, sim_steps, camera_ids, object_ids = read_frame_metadata(f)

        # Derive bounds from the actual object-table geometry unless explicitly
        # overridden via --bounds. Floor pins to the real tabletop height (with
        # scene-generalization offset); XY covers only the object table.
        if bounds_override is not None:
            occupancy_config["bounds"] = [list(b) for b in bounds_override]
            print(f"[INFO] occupancy bounds: override {occupancy_config['bounds']}")
        else:
            scene_path = resolve_scene_path(hdf5_path, f)
            auto_bounds = compute_default_occupancy_bounds(f, scene_path)
            if auto_bounds is not None:
                occupancy_config["bounds"] = auto_bounds
                print(f"[INFO] occupancy bounds: auto {auto_bounds} (table z={auto_bounds[0][2]:.4f})")
            else:
                print(f"[INFO] occupancy bounds: fallback {occupancy_config.get('bounds')} "
                      f"(scene YAML unavailable for {hdf5_path})")

        # Try GPU TSDF path first
        buffer = _try_gpu_tsdf(
            f, hdf5_path, frame_valid, frame_errors, sim_steps,
            camera_ids, object_ids, occupancy_config, exclude_cameras, device,
        )
        if buffer is None:
            # Fall back to original CPU path
            camera_ids_for_occ = [c for c in camera_ids if c not in (exclude_cameras or set())]
            occupancy = OccupancyLabeler(
                True, {**occupancy_config, "expected_camera_ids": camera_ids_for_occ}
            )
            buffer = _build_label_buffer(
                f, frame_valid, frame_errors, sim_steps, camera_ids, object_ids, occupancy,
                exclude_cameras=exclude_cameras,
            )

    if write_mode == "inplace":
        with h5py.File(hdf5_path, "r+") as target:
            labels_grp = target.require_group("labels")
            prepare_label_slot(labels_grp, "occupancy_tsdf", target_path=hdf5_path, overwrite=overwrite)
            writer = object.__new__(HDF5EpisodeWriter)
            writer._write_occupancy_labels(labels_grp, buffer)
        # Post-process: assign per-voxel semantic_id via object AABBs
        _assign_semantic_ids(hdf5_path, hdf5_path)
    else:
        sidecar = output_path or default_sidecar_path(hdf5_path)
        sync_sidecar(
            sidecar,
            source_path=hdf5_path,
            meta={
                "artifact_type": "derived_labels",
                "occupancy_config": occupancy_config,
                "occupancy_tsdf_semantics": "observed_three_state",
            },
            frame_valid=frame_valid,
            frame_errors=frame_errors,
            sim_steps=sim_steps,
        )
        with h5py.File(sidecar, "r+") as target:
            labels_grp = target.require_group("labels")
            prepare_label_slot(labels_grp, "occupancy_tsdf", target_path=sidecar, overwrite=overwrite)
            writer = object.__new__(HDF5EpisodeWriter)
            writer._write_occupancy_labels(labels_grp, buffer)
        # Post-process: assign per-voxel semantic_id via object AABBs
        _assign_semantic_ids(hdf5_path, sidecar)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate occupancy labels from stored HDF5 episodes."
    )
    parser.add_argument("input", help="HDF5 file or directory to scan for *.hdf5")
    parser.add_argument(
        "--write-mode",
        choices=["inplace", "sidecar"],
        default="sidecar",
        help="Write into source episode or derived sidecar (default: sidecar)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing occupancy labels")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size in meters (default: 0.01)")
    parser.add_argument("--occupied-margin", type=float, default=None, help="Occupied half-width; defaults to voxel-size")
    parser.add_argument("--free-margin", type=float, default=None, help="Free clearance; defaults to occupied-margin")
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        default=True,
        dest="all_cameras",
        help="Require all cameras have depth (default: true)",
    )
    parser.add_argument(
        "--no-all-cameras",
        action="store_false",
        dest="all_cameras",
        help="Allow occupancy when a subset of cameras is missing",
    )
    parser.add_argument(
        "--fusion-mode", choices=["binary", "tsdf"], default=None,
        help="Fusion algorithm (default: tsdf for offline)",
    )
    parser.add_argument(
        "--truncation-distance", type=float, default=None,
        help="TSDF truncation distance mu in meters (default: 3*voxel_size)",
    )
    parser.add_argument(
        "--surface-threshold", type=float, default=None,
        help="TSDF surface detection threshold in meters (default: 0.5*voxel_size)",
    )
    parser.add_argument(
        "--exclude-cameras", type=str, default=None,
        help="Comma-separated camera IDs to exclude (default: cam_wrist_left,cam_wrist_right; use '' for none)",
    )
    parser.add_argument(
        "--bounds", type=str, default=None,
        help="Override bounds as 'x0,y0,z0,x1,y1,z1' (e.g. '-0.5,-0.4,0.70,0.5,0.4,1.05')",
    )
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, 1=sequential debug; >1 forces CPU)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (default: auto)")
    args = parser.parse_args()

    config: dict[str, Any] = {
        "voxel_size": float(args.voxel_size),
        "require_all_cameras": args.all_cameras,
        "label_version": "occupancy_tsdf_v1",
    }
    if args.occupied_margin is not None:
        config["occupied_margin"] = float(args.occupied_margin)
    if args.free_margin is not None:
        config["free_margin"] = float(args.free_margin)
    if args.fusion_mode is not None:
        config["fusion_mode"] = args.fusion_mode
    if args.truncation_distance is not None:
        config["truncation_distance"] = args.truncation_distance
    if args.surface_threshold is not None:
        config["surface_threshold"] = args.surface_threshold
    if args.bounds is not None:
        vals = [float(v) for v in args.bounds.split(",")]
        if len(vals) != 6:
            print("[ERROR] --bounds requires exactly 6 comma-separated values: x0,y0,z0,x1,y1,z1")
            return 2
        config["bounds"] = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]

    # Resolve exclude-cameras: default excludes wrist cameras.
    if args.exclude_cameras is not None:
        exclude_cameras = set(s.strip() for s in args.exclude_cameras.split(",") if s.strip()) if args.exclude_cameras else set()
    else:
        exclude_cameras = set(DEFAULT_EXCLUDE_CAMERAS)

    occupancy_config = _normalize_offline_config(config)

    # GPU and multi-worker are mutually exclusive
    device_pref = args.device
    if args.workers > 1 and device_pref != "cpu":
        print("[INFO] workers > 1: forcing device=cpu (GPU contexts cannot be shared across processes)")
        device_pref = "cpu"

    print(f"[INFO] fusion_mode={occupancy_config['fusion_mode']}, "
          f"voxel_size={occupancy_config['voxel_size']}, "
          f"device={device_pref}, workers={args.workers or 'auto'}, "
          f"exclude_cameras={sorted(exclude_cameras) if exclude_cameras else 'none'}")

    paths = find_hdf5_files(args.input)
    if not paths:
        print(f"[ERROR] No HDF5 files found at: {args.input}")
        return 2

    # Explicit --bounds is forwarded verbatim; otherwise bounds are derived
    # per-episode from the object-table geometry inside generate_occupancy_for_file.
    bounds_override = config.get("bounds")

    process_one = functools.partial(
        generate_occupancy_for_file,
        occupancy_config=occupancy_config,
        write_mode=args.write_mode,
        overwrite=args.overwrite,
        exclude_cameras=exclude_cameras,
        device=device_pref,
        bounds_override=bounds_override,
    )

    n_ok, n_err = parallel_process(process_one, paths, workers=args.workers, label="occupancy-tsdf")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
