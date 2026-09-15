"""Shared infrastructure for offline label generation tools."""

from __future__ import annotations

import json
import io
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.camera_geometry import CameraFrame, rotation_matrix_to_quat_wxyz
from collector.hdf5_writer import HDF5EpisodeWriter


# ---------------------------------------------------------------------------
# HDF5 metadata reading
# ---------------------------------------------------------------------------

def _read_meta_dict(file: h5py.File, key: str) -> dict | None:
    """Read a JSON-encoded metadata field from /meta/."""
    meta = file.get("meta")
    if meta is None or key not in meta:
        return None
    raw = decode_hdf5_string(meta[key][()])
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except (json.JSONDecodeError, TypeError):
        return None


def resolve_scene_path(
    episode_path: str, file: h5py.File, scene_override: str | None = None
) -> str | None:
    """Resolve the scene YAML path from episode metadata.

    Tries an explicit override, then /meta/scene_file (JSON-encoded), then a
    raw /meta/scene_file string, against the episode directory and repo root.
    """
    if scene_override and os.path.isfile(scene_override):
        return scene_override

    scene_file = _read_meta_dict(file, "scene_file")
    if scene_file is None:
        # Try direct (non-JSON) meta field
        meta = file.get("meta")
        if meta is not None and "scene_file" in meta:
            scene_file = decode_hdf5_string(meta["scene_file"][()])

    if scene_file is None:
        return None

    # scene_file is relative to repo root (e.g. "scenes/01_rubiks_cube_flip_and_stack.yaml")
    if os.path.isabs(scene_file) and os.path.isfile(scene_file):
        return scene_file

    # Try relative to episode directory first, then repo root
    ep_dir = os.path.dirname(os.path.abspath(episode_path))
    candidates = [
        os.path.join(ep_dir, scene_file),
        os.path.join(REPO_ROOT, scene_file),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Cross-platform fallback: normalize path separators and extract a
    # recognizable suffix (e.g. "scenes/...") to resolve against repo root.
    normalized = scene_file.replace("\\", "/")
    for marker in ("scenes/",):
        idx = normalized.find(marker)
        if idx >= 0:
            rel_tail = normalized[idx:]
            candidate = os.path.join(REPO_ROOT, rel_tail)
            if os.path.isfile(candidate):
                return candidate
    return None


def _scene_table_height_offset(file: h5py.File) -> float:
    """Actual table-height offset (m) applied by scene generalization.

    Reads /meta/scene_generalization_sample -> spatial.table_height.height_offset_m.
    Returns 0.0 when generalization is off or the field is absent.
    """
    sample = _read_meta_dict(file, "scene_generalization_sample") or {}
    spatial = sample.get("spatial", {}) if isinstance(sample, dict) else {}
    table_h = spatial.get("table_height", {}) if isinstance(spatial, dict) else {}
    try:
        return float(table_h.get("height_offset_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compute_default_occupancy_bounds(
    source_file: h5py.File,
    scene_path: str | None,
    *,
    z_max: float = 1.50,
    xy_margin: float = 0.05,
    y_pos_margin: float = 0.15,
) -> list[list[float]] | None:
    """Compute occupancy bounds from the actual object-table geometry.

    Occupancy should cover only the object table (not the rear robot-support
    table), with the vertical floor pinned just below the tabletop slab so the
    table surface itself is captured as occupied.

    z_min = actual_z - tabletop_thickness (the 4cm slab becomes the bottom
    layers of the grid; objects rest on it above). z_max raised to 1.50 to
    cover objects lifted during the task (observed up to ~1.40m). y_max uses a
    wider positive-Y margin (y_pos_margin) to cover task objects that overhang
    the +Y table edge (e.g. the microwave reaches ~0.69m vs table edge 0.55m).

    Returns None when the scene YAML is unavailable, so callers can fall back
    to a hardcoded default.

    Mirrors ``build/table_geometry.resolve_table_geometry`` for the main-table
    XY extent (including the negative-Y shortening that excludes the robot
    table): x in [-sx/2, sx/2], y in [-sy/2 + shortening, sy/2 + y_pos_margin].
    """
    if scene_path is None or not os.path.isfile(scene_path):
        return None

    from tools.labels._mesh_voxelizer import parse_scene_table_spec
    try:
        from build.table_geometry import NEGATIVE_Y_SHORTENING_M
    except Exception:
        NEGATIVE_Y_SHORTENING_M = 0.25

    try:
        table_size, nominal_height = parse_scene_table_spec(scene_path)
    except Exception:
        return None

    sx, sy = float(table_size[0]), float(table_size[1])
    thickness = float(table_size[2])  # tabletop slab thickness (0.04)
    actual_z = float(nominal_height) + _scene_table_height_offset(source_file)

    x_min = -sx / 2.0 - xy_margin
    x_max = sx / 2.0 + xy_margin
    y_min = -sy / 2.0 + NEGATIVE_Y_SHORTENING_M - xy_margin
    y_max = sy / 2.0 + y_pos_margin
    z_min = actual_z - thickness  # floor below the tabletop slab -> table marked

    return [[x_min, y_min, z_min], [x_max, y_max, float(z_max)]]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_hdf5_files(path: str) -> list[str]:
    """Find episode HDF5 files, excluding derived sidecars."""
    if os.path.isdir(path):
        out: list[str] = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.endswith(".hdf5") and not name.endswith("_derived.hdf5"):
                    out.append(os.path.join(root, name))
        return sorted(out)
    if os.path.isfile(path):
        return [path]
    return []


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def decode_hdf5_string(value) -> str:
    """Decode a scalar string value from HDF5 (bytes, ndarray, or str)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_hdf5_string(value.item())
    return str(value)


def _read_depth_semantics(cam_grp: h5py.Group, frame_index: int) -> str:
    if "depth_semantics" not in cam_grp:
        return "distance_to_image_plane"
    dataset = cam_grp["depth_semantics"]
    value = dataset[()] if dataset.shape == () else dataset[frame_index]
    decoded = decode_hdf5_string(value).strip()
    return decoded or "distance_to_image_plane"


def _shape_pair(value: Any) -> tuple[int, int] | None:
    arr = np.asarray(value).reshape(-1)
    if arr.size < 2:
        return None
    height, width = int(arr[0]), int(arr[1])
    if height <= 0 or width <= 0:
        return None
    return height, width


def _encoded_image_bytes(value: Any) -> bytes | None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.astype(np.uint8, copy=False).tobytes()
    if isinstance(value, np.void):
        return bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    try:
        data = bytes(value)
    except Exception:
        return None
    return data or None


def _encoded_image_shape(dataset: h5py.Dataset) -> tuple[int, int] | None:
    if dataset.shape == ():
        indices: list[int | None] = [None]
    else:
        indices = list(range(min(int(dataset.shape[0]), 16)))

    for idx in indices:
        try:
            value = dataset[()] if idx is None else dataset[idx]
        except Exception:
            continue
        data = _encoded_image_bytes(value)
        if not data:
            continue
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
        except Exception:
            continue
        if height > 0 and width > 0:
            return int(height), int(width)
    return None


def infer_camera_image_shape(cam_grp: h5py.Group) -> tuple[int, int] | None:
    """Infer camera image shape as (height, width) without reading full video data."""
    if "image_shape" in cam_grp.attrs:
        shape = _shape_pair(cam_grp.attrs["image_shape"])
        if shape is not None:
            return shape
    if "image_shape" in cam_grp and isinstance(cam_grp["image_shape"], h5py.Dataset):
        shape = _shape_pair(cam_grp["image_shape"][()])
        if shape is not None:
            return shape
    if "height" in cam_grp.attrs and "width" in cam_grp.attrs:
        shape = _shape_pair([cam_grp.attrs["height"], cam_grp.attrs["width"]])
        if shape is not None:
            return shape

    depth_key = "depth_m" if "depth_m" in cam_grp else ("depth" if "depth" in cam_grp else None)
    if depth_key is not None:
        depth_shape = cam_grp[depth_key].shape
        if len(depth_shape) >= 3:
            return int(depth_shape[1]), int(depth_shape[2])

    if "rgb" not in cam_grp:
        return None

    rgb = cam_grp["rgb"]
    rgb_shape = rgb.shape
    if len(rgb_shape) >= 4:
        return int(rgb_shape[1]), int(rgb_shape[2])
    if len(rgb_shape) == 3:
        if int(rgb_shape[-1]) in (1, 3, 4):
            return int(rgb_shape[0]), int(rgb_shape[1])
        return int(rgb_shape[1]), int(rgb_shape[2])
    return _encoded_image_shape(rgb)


# ---------------------------------------------------------------------------
# Camera frame reading
# ---------------------------------------------------------------------------

def _load_camera_expected_forwards(file: h5py.File) -> dict[str, np.ndarray]:
    """Extract expected forward directions for world-mount cameras from metadata.

    Returns {cam_id: normalized_forward_vector} for cameras with position and target.
    """
    meta = file.get("meta")
    if meta is None or "camera_definitions" not in meta:
        return {}
    try:
        raw = decode_hdf5_string(meta["camera_definitions"][()])
        defs = json.loads(raw)
    except Exception:
        return {}
    out: dict[str, np.ndarray] = {}
    for d in defs:
        if d.get("mount_type") != "world":
            continue
        pos = d.get("position")
        tgt = d.get("target")
        if pos is None or tgt is None:
            continue
        fwd = np.asarray(tgt, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
        norm = np.linalg.norm(fwd)
        if norm < 1e-9:
            continue
        out[str(d["camera_id"])] = (fwd / norm).astype(np.float32)
    return out


def _load_camera_image_shapes(file: h5py.File) -> dict[str, tuple[int, int]]:
    """Extract (height, width) by camera id from metadata when images are encoded."""
    meta = file.get("meta")
    if meta is None or "camera_definitions" not in meta:
        return {}
    try:
        raw = decode_hdf5_string(meta["camera_definitions"][()])
        defs = json.loads(raw)
    except Exception:
        return {}

    out: dict[str, tuple[int, int]] = {}
    for d in defs:
        cam_id = d.get("camera_id")
        height = d.get("height")
        width = d.get("width")
        if cam_id is None or height is None or width is None:
            continue
        try:
            out[str(cam_id)] = (int(height), int(width))
        except (TypeError, ValueError):
            continue
    return out


def _decode_rgb_value(value: Any) -> np.ndarray:
    """Read RGB frames stored either as raw arrays or encoded image bytes."""
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return np.asarray(value, dtype=np.uint8)

    if isinstance(value, np.ndarray):
        data = value.astype(np.uint8, copy=False).tobytes()
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, np.void):
        data = bytes(value)
    else:
        data = bytes(value)

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _fix_extrinsic_flip(
    extrinsic: np.ndarray,
    expected_forward: np.ndarray,
) -> np.ndarray:
    """Correct a 180-degree extrinsic flip caused by Isaac Lab look-at gimbal lock.

    Isaac Lab can produce an inverted rotation for cameras looking exactly along
    a world axis (e.g. straight-down overhead cameras). This manifests as the
    body-frame X axis (forward) pointing opposite to the position→target direction.

    Detection: dot(R[:,0], expected_forward) < -0.5
    Correction: 180° rotation around body Y (left), i.e. negate columns 0 and 2.
    """
    R = extrinsic[:3, :3]
    actual_forward = R[:, 0]
    if np.dot(actual_forward, expected_forward) < -0.5:
        fixed = extrinsic.copy()
        fixed[:3, 0] = -R[:, 0]
        fixed[:3, 2] = -R[:, 2]
        return fixed
    return extrinsic


def read_camera_frames(
    file: h5py.File,
    frame_index: int,
    *,
    need_images: bool = True,
) -> dict[str, CameraFrame]:
    """Read camera frames from an HDF5 episode.

    Args:
        file: Open HDF5 file handle.
        frame_index: Which frame to read.
        need_images: If False, skip reading rgb/depth pixel data (faster for
            geometry-only operations like box2d projection).
    """
    out: dict[str, CameraFrame] = {}
    cameras = file.get("cameras")
    if cameras is None:
        return out
    expected_forwards = _load_camera_expected_forwards(file)
    camera_image_shapes = _load_camera_image_shapes(file)
    for cam_id in cameras.keys():
        cam_grp = cameras[cam_id]
        intrinsic = np.asarray(cam_grp["intrinsic"][()], dtype=np.float32)
        extrinsic = np.asarray(cam_grp["extrinsic_world_from_cam"][frame_index], dtype=np.float32)
        if cam_id in expected_forwards:
            extrinsic = _fix_extrinsic_flip(extrinsic, expected_forwards[cam_id])

        image_shape = infer_camera_image_shape(cam_grp)
        if image_shape is None and cam_id in camera_image_shapes:
            image_shape = camera_image_shapes[cam_id]
        if image_shape is None:
            continue

        # Support both "depth_m" (standard) and "depth" (legacy) field names.
        depth_key = "depth_m" if "depth_m" in cam_grp else ("depth" if "depth" in cam_grp else None)

        rgb, depth = None, None
        if need_images:
            if "rgb" in cam_grp:
                rgb = _decode_rgb_value(cam_grp["rgb"][frame_index])
            if depth_key is not None:
                depth = np.asarray(cam_grp[depth_key][frame_index], dtype=np.float32)

        # ---- camera model & fisheye metadata ----
        camera_model = "pinhole"
        if "camera_model" in cam_grp:
            camera_model = decode_hdf5_string(cam_grp["camera_model"][()])

        distortion_coefficients = None
        fisheye_camera_matrix = None
        fisheye_valid_mask = None
        clipping_range = None

        # Backward compat: accept old model names from previously written HDF5 files
        if camera_model in ("opencv_fisheye", "isaacsim_fisheye", "fisheye"):
            camera_model = "fisheye"
            if "distortion_coefficients" in cam_grp:
                distortion_coefficients = np.asarray(
                    cam_grp["distortion_coefficients"][()], dtype=np.float32
                )
            if "fisheye_camera_matrix" in cam_grp:
                fisheye_camera_matrix = np.asarray(
                    cam_grp["fisheye_camera_matrix"][()], dtype=np.float32
                )
            if "fisheye_valid_mask" in cam_grp:
                fisheye_valid_mask = np.asarray(
                    cam_grp["fisheye_valid_mask"][()], dtype=np.bool_
                )

        # ---- clipping_range (all camera types) ----
        if "clipping_range" in cam_grp:
            clipping_range = tuple(np.asarray(cam_grp["clipping_range"][()], dtype=np.float32).tolist())
        elif "render_clipping_range" in cam_grp:
            clipping_range = tuple(np.asarray(cam_grp["render_clipping_range"][()], dtype=np.float32).tolist())

        out[cam_id] = CameraFrame(
            rgb=rgb,
            depth_m=depth,
            intrinsic=intrinsic,
            extrinsic_world_from_cam=extrinsic,
            cam_pos_w=extrinsic[:3, 3].astype(np.float32),
            cam_quat_wxyz=rotation_matrix_to_quat_wxyz(extrinsic[:3, :3]),
            image_shape=image_shape,
            depth_semantics=_read_depth_semantics(cam_grp, frame_index),
            camera_model=camera_model,
            distortion_coefficients=distortion_coefficients,
            fisheye_camera_matrix=fisheye_camera_matrix,
            fisheye_valid_mask=fisheye_valid_mask,
            clipping_range=clipping_range,
        )
    return out


# ---------------------------------------------------------------------------
# Object state reading
# ---------------------------------------------------------------------------

def read_object_states(file: h5py.File, frame_index: int) -> dict[str, dict]:
    """Read per-object pose_world (and optionally qpos/joint_names) for a single frame."""
    out: dict[str, dict] = {}
    objects = file.get("objects")
    if objects is None:
        return out
    for obj_id in objects.keys():
        obj_grp = objects[obj_id]
        pose = np.asarray(obj_grp["pose_world"][frame_index], dtype=np.float32)
        if not np.all(np.isfinite(pose)):
            continue
        state: dict = {"pose_world": pose}
        if "qpos" in obj_grp:
            state["qpos"] = np.asarray(obj_grp["qpos"][frame_index], dtype=np.float32)
        if "joint_names" in obj_grp:
            state["joint_names"] = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in obj_grp["joint_names"][:]
            ]
        out[obj_id] = state
    return out


# ---------------------------------------------------------------------------
# Local bounding boxes
# ---------------------------------------------------------------------------

def read_local_bboxes(file: h5py.File) -> dict[str, tuple] | None:
    """Read per-object local AABBs from /meta/local_bboxes (JSON-encoded)."""
    meta = file.get("meta")
    if meta is None or "local_bboxes" not in meta:
        return None
    raw = json.loads(decode_hdf5_string(meta["local_bboxes"][()]))
    return {obj_id: (tuple(b[0]), tuple(b[1])) for obj_id, b in raw.items()}


# ---------------------------------------------------------------------------
# Frame-level metadata
# ---------------------------------------------------------------------------

def read_frame_metadata(
    file: h5py.File,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[str]]:
    """Read frame-level metadata arrays.

    Returns:
        (frame_valid, frame_errors, sim_steps, camera_ids, object_ids)
    """
    frame_count = int(file["frame_valid"].shape[0]) if "frame_valid" in file else 0
    camera_ids = list(file["cameras"].keys()) if file.get("cameras") is not None else []
    object_ids = list(file["objects"].keys()) if file.get("objects") is not None else []
    frame_valid = (
        np.asarray(file["frame_valid"][:], dtype=np.bool_)
        if "frame_valid" in file
        else np.ones((frame_count,), dtype=np.bool_)
    )
    frame_errors = [
        str(item)
        for item in (file["frame_errors"][:] if "frame_errors" in file else np.asarray([""] * frame_count))
    ]
    sim_steps = (
        np.asarray(file["time"]["sim_step"][:], dtype=np.int64)
        if "time" in file and "sim_step" in file["time"]
        else np.arange(frame_count, dtype=np.int64)
    )
    return frame_valid, frame_errors, sim_steps, camera_ids, object_ids


# ---------------------------------------------------------------------------
# Sidecar / write helpers
# ---------------------------------------------------------------------------

def default_sidecar_path(hdf5_path: str) -> str:
    """Derive the default sidecar output path for a source episode."""
    source = Path(hdf5_path)
    return str(source.parent / "derived" / f"{source.stem}_derived.hdf5")


def prepare_label_slot(
    labels_grp: h5py.Group,
    label_name: str,
    *,
    target_path: str,
    overwrite: bool,
) -> None:
    """Ensure a label slot is available, raising if it exists and overwrite is False."""
    if label_name not in labels_grp:
        return
    if not overwrite:
        raise ValueError(
            f"{target_path}: {label_name} labels already exist, pass overwrite=True to replace them."
        )
    del labels_grp[label_name]


def sync_sidecar(
    sidecar_path: str,
    *,
    source_path: str,
    meta: dict[str, Any],
    frame_valid: np.ndarray,
    frame_errors: list[str],
    sim_steps: np.ndarray,
) -> None:
    """Create or update a derived-labels sidecar HDF5 file."""
    if not os.path.exists(sidecar_path):
        HDF5EpisodeWriter.init_labels_artifact(
            sidecar_path,
            meta={
                **meta,
                "source_episode_file": os.path.abspath(source_path),
            },
            frame_valid=frame_valid,
            frame_errors=[str(item) for item in frame_errors],
            sim_steps=sim_steps,
        )
        return

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(sidecar_path, "r+") as target:
        if "meta" in target:
            del target["meta"]
        HDF5EpisodeWriter._write_meta_group(
            target,
            {
                **meta,
                "source_episode_file": os.path.abspath(source_path),
            },
        )
        if "frame_valid" in target:
            del target["frame_valid"]
        target.create_dataset("frame_valid", data=np.asarray(frame_valid, dtype=np.bool_))
        if "frame_errors" in target:
            del target["frame_errors"]
        target.create_dataset(
            "frame_errors",
            data=np.asarray([str(item) for item in frame_errors], dtype=str_dtype),
        )
        if "time" in target:
            del target["time"]
        time_grp = target.create_group("time")
        time_grp.create_dataset("sim_step", data=np.asarray(sim_steps, dtype=np.int64))
        target.require_group("labels")
