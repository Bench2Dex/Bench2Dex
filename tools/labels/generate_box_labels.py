"""Generate box3d and optional box2d labels from stored HDF5 episodes.

Common usage:

    # 1) Generate box3d only into a derived sidecar file (default, safest).
    #    The source episode is not modified.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5

    # 2) Generate box3d + box2d into a derived sidecar file.
    #    box2d is projected from box3d and camera intrinsics/extrinsics.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5 --box2d

    # 3) Generate box3d only and write directly into the source HDF5.
    #    This modifies /path/to/episode.hdf5 in place.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5 \
        --write-mode inplace

    # 4) Generate box3d + box2d and write directly into the source HDF5.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5 \
        --box2d \
        --write-mode inplace

    # 5) Replace existing /labels/box3d and /labels/box2d in the source HDF5.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5 \
        --box2d \
        --write-mode inplace \
        --overwrite

    # 6) Process every *.hdf5 under a directory.
    #    Default output is one sidecar per source episode.
    python tools/labels/generate_box_labels.py /path/to/episode_dir/

    # 7) Directory batch with box2d, CPU workers, and overwrite existing sidecars.
    python tools/labels/generate_box_labels.py /path/to/episode_dir/ \
        --box2d \
        --workers 8 \
        --device cpu \
        --overwrite

    # 8) Sequential debug run on CPU.
    python tools/labels/generate_box_labels.py /path/to/episode.hdf5 \
        --box2d \
        --workers 1 \
        --device cpu

Output modes:

    --write-mode sidecar   Default. Writes labels to a derived sidecar HDF5
                           and leaves the source episode unchanged.
    --write-mode inplace   Writes labels into the source HDF5 under /labels.

Inputs required in the source HDF5:

    box3d requires:
        /meta/local_bboxes
        /objects/<object_id>/pose_world
        /frame_valid

    box2d additionally requires:
        /cameras/<camera_id>/intrinsic
        /cameras/<camera_id>/extrinsic_world_from_cam
        camera image shape from depth/depth_m or rgb datasets

Generated paths:

    /labels/box3d/<object_id>/center_world
    /labels/box3d/<object_id>/size_lwh
    /labels/box3d/<object_id>/quat_world

    /labels/box2d/<camera_id>/<object_id>/xyxy       (only with --box2d)
    /labels/box2d/<camera_id>/<object_id>/visible    (only with --box2d)
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import warnings

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.labels._label_common import (
    default_sidecar_path,
    find_hdf5_files,
    prepare_label_slot,
    read_camera_frames,
    read_frame_metadata,
    read_local_bboxes,
    read_object_states,
    sync_sidecar,
)
from tools.labels._accel import read_episode_batch, get_device, parallel_process
from collector.box_labeler import compute_box2d_labels, compute_box3d_labels


# ---------------------------------------------------------------------------
# Write helpers (match hdf5_writer.py format exactly)
# ---------------------------------------------------------------------------

def _write_box3d(
    labels_grp: h5py.Group,
    object_ids: list[str],
    box3d_all: list[dict],
    frame_count: int,
) -> None:
    box3d_grp = labels_grp.create_group("box3d")
    for obj_id in object_ids:
        obj_grp = box3d_grp.create_group(obj_id)
        center = np.full((frame_count, 3), np.nan, dtype=np.float32)
        size = np.full((frame_count, 3), np.nan, dtype=np.float32)
        quat = np.full((frame_count, 4), np.nan, dtype=np.float32)
        for i, box3d in enumerate(box3d_all):
            b = box3d.get(obj_id)
            if b is not None:
                center[i] = b["center_world"]
                size[i] = b["size_lwh"]
                quat[i] = b["quat_world"]
        obj_grp.create_dataset("center_world", data=center)
        obj_grp.create_dataset("size_lwh", data=size)
        obj_grp.create_dataset("quat_world", data=quat)


def _write_box2d(
    labels_grp: h5py.Group,
    object_ids: list[str],
    box2d_all: list[dict],
    frame_count: int,
) -> None:
    all_cam_ids = sorted({cam_id for box2d in box2d_all for cam_id in box2d.keys()})
    if not all_cam_ids:
        return
    box2d_grp = labels_grp.create_group("box2d")
    for cam_id in all_cam_ids:
        cam_grp = box2d_grp.create_group(cam_id)
        for obj_id in object_ids:
            obj_grp = cam_grp.create_group(obj_id)
            xyxy = np.full((frame_count, 4), -1, dtype=np.int32)
            visible = np.zeros((frame_count,), dtype=np.bool_)
            for i, box2d in enumerate(box2d_all):
                b = box2d.get(cam_id, {}).get(obj_id)
                if b is not None:
                    xyxy[i] = b["xyxy"]
                    visible[i] = bool(b["visible"])
            obj_grp.create_dataset("xyxy", data=xyxy)
            obj_grp.create_dataset("visible", data=visible)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _generate_box_labels_batch(
    f: h5py.File,
    hdf5_path: str,
    frame_valid: np.ndarray,
    frame_count: int,
    object_ids: list[str],
    local_bboxes: dict,
    camera_ids: list[str],
    box2d: bool,
    device_pref: str,
) -> tuple[list[dict], list[dict]]:
    """Try GPU/batch path, fall back to per-frame CPU loop."""
    try:
        dev = get_device(device_pref)
        if dev.type != "cpu":
            return _generate_box_gpu(f, frame_valid, frame_count, object_ids,
                                     local_bboxes, box2d, dev)
    except Exception as exc:
        warnings.warn(f"{hdf5_path}: GPU box labels failed ({exc}), falling back to CPU")

    return _generate_box_cpu(f, frame_valid, frame_count, local_bboxes, box2d)


def _generate_box_gpu(
    f: h5py.File,
    frame_valid: np.ndarray,
    frame_count: int,
    object_ids: list[str],
    local_bboxes: dict,
    box2d: bool,
    device: object,
) -> tuple[list[dict], list[dict]]:
    """GPU/batch path: all frames at once."""
    from tools.labels._gpu_bbox import batch_box3d_all_frames, batch_box2d_all_frames

    batch_data = read_episode_batch(f, need_depth=False)
    box3d_results = batch_box3d_all_frames(batch_data, device)

    # Convert to per-frame dicts
    box3d_all: list[dict] = []
    for fi in range(frame_count):
        if not bool(frame_valid[fi]):
            box3d_all.append({})
            continue
        frame_labels = {}
        for obj_id, data in box3d_results.items():
            if np.any(np.isnan(data["center_world"][fi])):
                continue
            frame_labels[obj_id] = {
                "center_world": data["center_world"][fi],
                "size_lwh": data["size_lwh"],
                "quat_world": data["quat_world"][fi],
                "corners_world": data["corners_world"][fi],
            }
        box3d_all.append(frame_labels)

    box2d_all: list[dict] = []
    if box2d:
        box2d_batch = batch_box2d_all_frames(box3d_results, batch_data, device)
        for fi in range(frame_count):
            if not bool(frame_valid[fi]):
                box2d_all.append({})
                continue
            frame_labels = {}
            for cam_id, cam_data in box2d_batch.items():
                cam_labels = {}
                for obj_id, obj_data in cam_data.items():
                    cam_labels[obj_id] = {
                        "xyxy": obj_data["xyxy"][fi],
                        "visible": bool(obj_data["visible"][fi]),
                    }
                frame_labels[cam_id] = cam_labels
            box2d_all.append(frame_labels)
    else:
        box2d_all = [{} for _ in range(frame_count)]

    return box3d_all, box2d_all


def _generate_box_cpu(
    f: h5py.File,
    frame_valid: np.ndarray,
    frame_count: int,
    local_bboxes: dict,
    box2d: bool,
) -> tuple[list[dict], list[dict]]:
    """Original per-frame CPU path."""
    box3d_all: list[dict] = []
    box2d_all: list[dict] = []

    for i in range(frame_count):
        if not bool(frame_valid[i]):
            box3d_all.append({})
            box2d_all.append({})
            continue

        object_states = read_object_states(f, i)
        box3d_labels = compute_box3d_labels(object_states, local_bboxes)
        box3d_all.append(box3d_labels)

        if box2d:
            camera_frames = read_camera_frames(f, i, need_images=False)
            box2d_all.append(compute_box2d_labels(box3d_labels, camera_frames))
        else:
            box2d_all.append({})

    return box3d_all, box2d_all

def generate_box_labels_for_file(
    hdf5_path: str,
    *,
    box2d: bool = False,
    write_mode: str = "sidecar",
    overwrite: bool = False,
    output_path: str | None = None,
    device: str = "auto",
) -> bool:
    """Generate box3d (and optionally box2d) labels for a single episode.

    Returns True if labels were written, False if skipped.
    """
    with h5py.File(hdf5_path, "r") as f:
        local_bboxes = read_local_bboxes(f)
        if local_bboxes is None:
            print(f"[WARN] {hdf5_path}: /meta/local_bboxes not found, skipping")
            return False

        frame_valid, frame_errors, sim_steps, camera_ids, object_ids = read_frame_metadata(f)
        frame_count = int(frame_valid.shape[0])

        # Try GPU batch path, fall back to per-frame loop on failure
        box3d_all, box2d_all = _generate_box_labels_batch(
            f, hdf5_path, frame_valid, frame_count, object_ids,
            local_bboxes, camera_ids, box2d, device,
        )

    # ---- write results ----
    label_names = ["box3d"] + (["box2d"] if box2d else [])

    if write_mode == "inplace":
        with h5py.File(hdf5_path, "r+") as target:
            labels_grp = target.require_group("labels")
            for name in label_names:
                prepare_label_slot(labels_grp, name, target_path=hdf5_path, overwrite=overwrite)
            _write_box3d(labels_grp, object_ids, box3d_all, frame_count)
            if box2d:
                _write_box2d(labels_grp, object_ids, box2d_all, frame_count)
    else:
        sidecar = output_path or default_sidecar_path(hdf5_path)
        sync_sidecar(
            sidecar,
            source_path=hdf5_path,
            meta={"artifact_type": "derived_labels"},
            frame_valid=frame_valid,
            frame_errors=frame_errors,
            sim_steps=sim_steps,
        )
        with h5py.File(sidecar, "r+") as target:
            labels_grp = target.require_group("labels")
            for name in label_names:
                prepare_label_slot(labels_grp, name, target_path=sidecar, overwrite=overwrite)
            _write_box3d(labels_grp, object_ids, box3d_all, frame_count)
            if box2d:
                _write_box2d(labels_grp, object_ids, box2d_all, frame_count)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate box3d/box2d labels from stored HDF5 episodes."
    )
    parser.add_argument("input", help="HDF5 file or directory to scan for *.hdf5")
    parser.add_argument("--box2d", action="store_true", help="Also generate 2D bounding boxes (projected from 3D)")
    parser.add_argument(
        "--write-mode",
        choices=["inplace", "sidecar"],
        default="sidecar",
        help="Write into source episode or derived sidecar (default: sidecar)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing labels")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, 1=sequential debug; >1 forces CPU)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (default: auto)")
    args = parser.parse_args()

    paths = find_hdf5_files(args.input)
    if not paths:
        print(f"[ERROR] No HDF5 files found at: {args.input}")
        return 2

    # GPU and multi-worker are mutually exclusive
    device_pref = args.device
    if args.workers > 1 and device_pref != "cpu":
        print("[INFO] workers > 1: forcing device=cpu (GPU contexts cannot be shared across processes)")
        device_pref = "cpu"

    mode_label = "box3d+box2d" if args.box2d else "box3d"
    print(f"[INFO] mode={mode_label}, device={device_pref}, workers={args.workers or 'auto'}")

    process_one = functools.partial(
        generate_box_labels_for_file,
        box2d=args.box2d,
        write_mode=args.write_mode,
        overwrite=args.overwrite,
        device=device_pref,
    )

    n_ok, n_err = parallel_process(process_one, paths, workers=args.workers, label=mode_label)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
