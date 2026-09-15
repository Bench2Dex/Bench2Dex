#!/usr/bin/env python3
"""Audit wrist-camera table/robot penetration in per-episode HDF5 files.

The check is intentionally geometric:
  - The table is modeled as a world-frame AABB from scene YAML metadata.
  - Each wrist camera optical center comes from extrinsic_world_from_cam[:3, 3].
  - The camera body is conservatively approximated as a sphere around that
    optical center.
  - The robot is modeled from the HDF5 qpos trajectory plus the robot USD joint
    tree.  Each link is approximated by the segment between FK parent/child
    link origins, and the near wrist-camera body/view axis is probed against
    those segments.

With --recam, the script does not judge the saved camera pose directly.
Instead it recovers the wrist parent-link trajectory from the saved camera pose
and saved mount offset, then applies the current collect config mount offset.
This estimates whether edited wrist-camera parameters would collide before a
new replay is rendered.

This flags definite optical-center penetration separately from possible body
overlap, robot proximity/collision, and low-clearance warnings.

Usage:
  python tools/audit_wrist_camera_table_penetration.py <hdf5_dir>
  python tools/audit_wrist_camera_table_penetration.py <hdf5_dir> --recam
"""


from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_TABLE_SIZE = (2.2, 1.1, 0.04)
DEFAULT_TABLE_Z = 0.75
DEFAULT_WRIST_CAMERA_IDS = ("cam_wrist_left", "cam_wrist_right")
ROBOT_BACK_REFERENCE_Y = -0.55

ROBOT_PLACEMENT_DEFAULTS: dict[str, tuple[float, float, float, tuple[float, float, float]]] = {
    # robot_key: (base_x, back_edge_margin_y, base_z_offset, root_rpy_deg)
    "multi_ur5_rh56dfx_with_flange": (0.5, 0.12, 0.0, (0.0, 0.0, 90.0)),
    "multi_ur5_rh5dg2_with_flange": (0.5, 0.12, 0.0, (0.0, 0.0, 90.0)),
    "multi_ur5_schunk_hand_with_flange": (0.5, 0.12, 0.0, (0.0, 0.0, 90.0)),
    "multi_ur5_wuji_with_flange": (0.5, 0.12, 0.0, (0.0, 0.0, 90.0)),
    "multi_ur5_shadow_hand_with_flange": (0.75, 0.12, 0.0, (0.0, 0.0, 90.0)),
}

ROBOT_USD_RELATIVE_CANDIDATES: dict[str, tuple[Path, ...]] = {
    "multi_ur5_rh56dfx_with_flange": (
        Path("dex2bench_dataset/Robots_p/ur5+RH56DFX/usd/Multi_UR5_RH56DFX_with_flange.usd"),
        Path("Bench2Dex/Robots_p/ur5+RH56DFX/usd/Multi_UR5_RH56DFX_with_flange.usd"),
        Path("dex2bench/ur5+RH56DFX/usd/Multi_UR5_RH56DFX_with_flange.usd"),
        Path("ur5+RH56DFX/usd/Multi_UR5_RH56DFX_with_flange.usd"),
    ),
    "multi_ur5_rh5dg2_with_flange": (
        Path("dex2bench_dataset/Robots_p/ur5+RH5DG2/usd/Multi_UR5_RH5DG2_with_flange.usd"),
    ),
    "multi_ur5_schunk_hand_with_flange": (
        Path("dex2bench_dataset/Robots_p/ur5+schunk_hand/usd/Multi_UR5_schunk_hand_with_flange.usd"),
    ),
    "multi_ur5_wuji_with_flange": (
        Path("dex2bench_dataset/Robots_p/ur5+wuji/usd/Multi_UR5_wuji_with_flange.usd"),
    ),
    "multi_ur5_shadow_hand_with_flange": (
        Path("dex2bench_dataset/Robots_p/ur5+shadow_hand/usd/Multi_UR5_shadow_hand_with_flange.usd"),
    ),
}


@dataclass
class CameraAudit:
    file: str
    camera_id: str
    pose_source: str
    parent_link: str
    frame_count: int
    table_z: float
    table_size_x: float
    table_size_y: float
    min_origin_z: float
    min_top_clearance_m: float
    worst_frame: int
    origin_penetration_frames: int
    body_overlap_frames: int
    low_clearance_frames: int
    invalid_transform_frames: int
    table_status: str
    robot_status: str
    robot_min_distance_m: float
    robot_min_clearance_m: float
    robot_worst_frame: int
    robot_penetration_frames: int
    robot_severe_frames: int
    robot_nearest_link: str
    robot_error: str
    status: str
    reasons: str


@dataclass
class FileError:
    file: str
    error: str


@dataclass
class RobotKinematicCache:
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_labels: list[str]
    frame_count: int
    usd_path: str


@dataclass
class RobotProbeResult:
    status: str = "skipped"
    min_distance_m: float = math.inf
    min_clearance_m: float = math.inf
    worst_frame: int = -1
    penetration_frames: int = 0
    severe_frames: int = 0
    nearest_link: str = ""
    error: str = ""


def _decode_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_hdf5_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_meta_scalar(h5: h5py.File, key: str) -> Any | None:
    path = f"meta/{key}"
    if path not in h5:
        return None
    return _decode_hdf5_value(h5[path][()])


def _read_json_meta(h5: h5py.File, key: str) -> Any | None:
    value = _read_meta_scalar(h5, key)
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _parse_float_triplet(value: str, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("expected 'x,y' or 'x,y,z'")
    try:
        floats = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(floats) == 2:
        floats.append(default[2])
    return (floats[0], floats[1], floats[2])


def _scene_candidates_from_hdf5(h5: h5py.File) -> list[Path]:
    candidates: list[Path] = []

    explicit = _read_meta_scalar(h5, "scene_file")
    if isinstance(explicit, str) and explicit:
        path = Path(explicit)
        candidates.append(path)
        parts = path.parts
        if "scenes" in parts:
            idx = parts.index("scenes")
            candidates.append(REPO_ROOT.joinpath(*parts[idx:]))

    manifest = _read_json_meta(h5, "task_manifest")
    if isinstance(manifest, dict):
        scene_file = manifest.get("scene_file")
        if isinstance(scene_file, str) and scene_file:
            path = Path(scene_file)
            candidates.append(path)
            candidates.append(REPO_ROOT / path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _read_scene_table_spec(scene_path: Path) -> tuple[tuple[float, float, float], float]:
    with scene_path.open("r", encoding="utf-8") as f:
        task = yaml.safe_load(f) or {}
    table = task.get("table", {}) or {}
    size_values = table.get("size", DEFAULT_TABLE_SIZE)
    size = tuple(float(v) for v in size_values)
    if len(size) == 2:
        size = (size[0], size[1], DEFAULT_TABLE_SIZE[2])
    height = float(table.get("height", task.get("metrics", {}).get("safety", {}).get("table_z", DEFAULT_TABLE_Z)))
    return (size[0], size[1], size[2]), height


def _table_height_offset(h5: h5py.File) -> float:
    sample = _read_json_meta(h5, "scene_generalization_sample")
    if not isinstance(sample, dict):
        return 0.0
    table_height = sample.get("spatial", {}).get("table_height", {})
    if not isinstance(table_height, dict):
        return 0.0
    for key in ("height_offset_m", "offset_m"):
        value = table_height.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _camera_generalization_sample(h5: h5py.File, camera_id: str) -> dict[str, Any]:
    sample = _read_json_meta(h5, "scene_generalization_sample")
    if not isinstance(sample, dict):
        return {}
    cameras = sample.get("spatial", {}).get("camera", {}).get("cameras", {})
    if not isinstance(cameras, dict):
        return {}
    value = cameras.get(camera_id)
    return value if isinstance(value, dict) else {}


def _rotation_from_rpy_rad(rpy_rad: tuple[float, float, float] | list[float]) -> np.ndarray:
    r, p, y = (float(v) for v in rpy_rad)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _mount_transform(offset_xyz: tuple[float, float, float], offset_rpy: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_rpy_rad(offset_rpy)
    transform[:3, 3] = np.asarray(offset_xyz, dtype=np.float64)
    return transform


def _effective_mount(
    *,
    offset_xyz: tuple[float, float, float],
    offset_rpy: tuple[float, float, float],
    camera_sample: dict[str, Any],
    apply_camera_perturbation: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xyz = np.asarray(offset_xyz, dtype=np.float64)
    rpy = np.asarray(offset_rpy, dtype=np.float64)
    if apply_camera_perturbation:
        xyz = xyz + np.asarray(camera_sample.get("offset_xyz_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        rpy_offset_deg = np.asarray(camera_sample.get("rpy_offset_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
        rpy = rpy + np.radians(rpy_offset_deg)
    return tuple(float(v) for v in xyz), tuple(float(v) for v in rpy)


def _normalize_camera_def(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    camera_id = raw.get("camera_id", raw.get("id"))
    if not camera_id:
        return None
    return {
        "camera_id": str(camera_id),
        "mount_type": str(raw.get("mount_type", "world")).lower(),
        "parent_link": None if raw.get("parent_link") is None else str(raw.get("parent_link")),
        "offset_xyz": tuple(float(v) for v in raw.get("offset_xyz", [0.0, 0.0, 0.0])),
        "offset_rpy": tuple(float(v) for v in raw.get("offset_rpy", [0.0, 0.0, 0.0])),
    }


def _hdf5_camera_definitions(h5: h5py.File) -> dict[str, dict[str, Any]]:
    sources = []
    camera_defs = _read_json_meta(h5, "camera_definitions")
    if isinstance(camera_defs, list):
        sources.extend(camera_defs)

    collect_cfg = _read_json_meta(h5, "collect_config")
    if isinstance(collect_cfg, dict):
        camera_list = collect_cfg.get("camera_list", [])
        if isinstance(camera_list, list):
            sources.extend(camera_list)

    out: dict[str, dict[str, Any]] = {}
    for raw in sources:
        normalized = _normalize_camera_def(raw)
        if normalized is not None:
            out.setdefault(normalized["camera_id"], normalized)
    return out


def _robot_key_from_hdf5(h5: h5py.File) -> str | None:
    value = _read_meta_scalar(h5, "robot_key")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


@contextlib.contextmanager
def _suppress_stderr_fd(enabled: bool):
    if not enabled:
        yield
        return
    old_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)


def _rpy_deg_to_quat_xyzw(rpy_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def _robot_usd_candidates(robot_key: str) -> list[Path]:
    rel_candidates = ROBOT_USD_RELATIVE_CANDIDATES.get(robot_key, ())
    bases = (REPO_ROOT.parent, REPO_ROOT)
    candidates: list[Path] = []
    for rel in rel_candidates:
        if rel.is_absolute():
            candidates.append(rel)
            continue
        for base in bases:
            candidates.append(base / rel)
    return candidates


def _resolve_robot_usd_path(robot_key: str | None, override: str | None) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"--robot-usd does not exist: {path}")
    if not robot_key:
        raise ValueError("missing meta/robot_key; use --robot-usd to enable robot penetration check")
    for candidate in _robot_usd_candidates(robot_key):
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in _robot_usd_candidates(robot_key))
    raise FileNotFoundError(f"no robot USD candidate found for robot_key={robot_key!r}; searched: {searched}")


def _robot_root_pose_xyzw(h5: h5py.File, robot_key: str | None, table_z: float) -> np.ndarray:
    pose = _read_json_meta(h5, "robot_pose")
    if isinstance(pose, dict):
        pos_raw = pose.get("pos", pose.get("position"))
        if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 3:
            pos = tuple(float(v) for v in pos_raw[:3])
        else:
            pos = None

        quat_raw = pose.get("quat_xyzw")
        if isinstance(quat_raw, (list, tuple)) and len(quat_raw) == 4:
            quat_xyzw = tuple(float(v) for v in quat_raw)
        else:
            quat_wxyz_raw = pose.get("quat_wxyz", pose.get("rot"))
            if isinstance(quat_wxyz_raw, (list, tuple)) and len(quat_wxyz_raw) == 4:
                qw, qx, qy, qz = (float(v) for v in quat_wxyz_raw)
                quat_xyzw = (qx, qy, qz, qw)
            else:
                rpy_raw = pose.get("rpy_deg")
                if isinstance(rpy_raw, (list, tuple)) and len(rpy_raw) == 3:
                    quat_xyzw = _rpy_deg_to_quat_xyzw(tuple(float(v) for v in rpy_raw))
                else:
                    quat_xyzw = None

        if pos is not None and quat_xyzw is not None:
            return np.asarray([*pos, *quat_xyzw], dtype=np.float64)

    placement = ROBOT_PLACEMENT_DEFAULTS.get(str(robot_key))
    if placement is None:
        placement = (0.5, 0.12, 0.0, (0.0, 0.0, 90.0))
    base_x, back_margin_y, base_z_offset, rpy_deg = placement
    pos = (base_x, ROBOT_BACK_REFERENCE_Y + back_margin_y, table_z + base_z_offset)
    return np.asarray([*pos, *_rpy_deg_to_quat_xyzw(rpy_deg)], dtype=np.float64)


def _read_robot_joint_names(h5: h5py.File) -> list[str]:
    if "robot/joint_names" not in h5:
        raise KeyError("missing robot/joint_names")
    raw = h5["robot/joint_names"][()]
    if isinstance(raw, np.ndarray):
        values = raw.reshape(-1).tolist()
    else:
        values = list(raw)
    return [str(_decode_hdf5_value(value)) for value in values]


_FK_CACHE: dict[str, Any] = {}


def _load_articulation_fk(usd_path: Path, *, quiet_usd_warnings: bool) -> Any:
    cache_key = str(usd_path)
    if cache_key not in _FK_CACHE:
        from tools.labels._mesh_voxelizer import ArticulationFK

        with _suppress_stderr_fd(quiet_usd_warnings):
            fk = ArticulationFK(cache_key)
        if not getattr(fk, "joints", None):
            raise ValueError(f"robot USD has no parsed joints: {usd_path}")
        _FK_CACHE[cache_key] = fk
    return _FK_CACHE[cache_key]


def _build_robot_kinematic_cache(
    h5: h5py.File,
    *,
    table_z: float,
    robot_usd_override: str | None,
    quiet_usd_warnings: bool,
) -> RobotKinematicCache:
    robot_key = _robot_key_from_hdf5(h5)
    usd_path = _resolve_robot_usd_path(robot_key, robot_usd_override)
    fk = _load_articulation_fk(usd_path, quiet_usd_warnings=quiet_usd_warnings)

    if "robot/qpos" not in h5:
        raise KeyError("missing robot/qpos")
    qpos = np.asarray(h5["robot/qpos"], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"robot/qpos has shape {qpos.shape}, expected T x J")
    joint_names = _read_robot_joint_names(h5)
    root_pose = _robot_root_pose_xyzw(h5, robot_key, table_z)

    joints = list(fk.joints)
    frame_count = int(qpos.shape[0])
    segment_starts = np.full((frame_count, len(joints), 3), np.nan, dtype=np.float32)
    segment_ends = np.full((frame_count, len(joints), 3), np.nan, dtype=np.float32)
    segment_labels = [f"{joint.parent_link}->{joint.child_link}" for joint in joints]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for frame_idx in range(frame_count):
            transforms = fk.compute_link_transforms(root_pose, qpos[frame_idx], joint_names)
            for joint_idx, joint in enumerate(joints):
                parent_tf = transforms.get(joint.parent_link)
                child_tf = transforms.get(joint.child_link)
                if parent_tf is None or child_tf is None:
                    continue
                segment_starts[frame_idx, joint_idx] = np.asarray(parent_tf[:3, 3], dtype=np.float32)
                segment_ends[frame_idx, joint_idx] = np.asarray(child_tf[:3, 3], dtype=np.float32)

    return RobotKinematicCache(
        segment_starts=segment_starts,
        segment_ends=segment_ends,
        segment_labels=segment_labels,
        frame_count=frame_count,
        usd_path=str(usd_path),
    )


def _load_recam_camera_configs(config_path: Path, robot_key: str | None) -> dict[str, Any]:
    from collector.config import load_collect_config

    cfg = load_collect_config(str(config_path), robot_key=robot_key)
    return {cam.camera_id: cam for cam in cfg.cameras}


def _recam_extrinsic_from_config(
    h5: h5py.File,
    camera_id: str,
    *,
    new_cam_cfg: Any,
    apply_new_camera_perturbation: bool,
) -> tuple[np.ndarray, str]:
    old_defs = _hdf5_camera_definitions(h5)
    old_def = old_defs.get(camera_id)
    if old_def is None:
        raise KeyError(f"missing saved camera definition for {camera_id}")
    if old_def.get("mount_type") != "robot_link":
        raise ValueError(f"{camera_id} saved mount_type is {old_def.get('mount_type')!r}, expected robot_link")
    if str(getattr(new_cam_cfg, "mount_type", "")).lower() != "robot_link":
        raise ValueError(f"{camera_id} new config mount_type is {getattr(new_cam_cfg, 'mount_type', None)!r}, expected robot_link")
    old_parent = old_def.get("parent_link")
    new_parent = getattr(new_cam_cfg, "parent_link", None)
    if old_parent and new_parent and str(old_parent) != str(new_parent):
        raise ValueError(
            f"{camera_id} parent_link changed from {old_parent!r} to {new_parent!r}; "
            "cannot recover the new parent trajectory from saved camera extrinsics"
        )

    extrinsic_path = f"cameras/{camera_id}/extrinsic_world_from_cam"
    if extrinsic_path not in h5:
        raise KeyError(f"missing {extrinsic_path}; --recam needs saved old camera extrinsics")
    old_camera_world = np.asarray(h5[extrinsic_path], dtype=np.float64)
    if old_camera_world.ndim != 3 or old_camera_world.shape[1:] != (4, 4):
        raise ValueError(f"{extrinsic_path} has shape {old_camera_world.shape}, expected T x 4 x 4")

    camera_sample = _camera_generalization_sample(h5, camera_id)
    old_xyz, old_rpy = _effective_mount(
        offset_xyz=old_def["offset_xyz"],
        offset_rpy=old_def["offset_rpy"],
        camera_sample=camera_sample,
        apply_camera_perturbation=True,
    )
    new_xyz, new_rpy = _effective_mount(
        offset_xyz=tuple(float(v) for v in getattr(new_cam_cfg, "offset_xyz")),
        offset_rpy=tuple(float(v) for v in getattr(new_cam_cfg, "offset_rpy")),
        camera_sample=camera_sample,
        apply_camera_perturbation=apply_new_camera_perturbation,
    )

    old_parent_from_camera = np.linalg.inv(_mount_transform(old_xyz, old_rpy))
    new_parent_to_camera = _mount_transform(new_xyz, new_rpy)
    parent_world = old_camera_world @ old_parent_from_camera[None, :, :]
    new_camera_world = parent_world @ new_parent_to_camera[None, :, :]
    return new_camera_world.astype(np.float64), str(new_parent or old_parent or "")


def _resolve_table_spec(
    h5: h5py.File,
    *,
    scene_path: Path | None,
    table_z: float | None,
    base_table_z: float | None,
    table_size: tuple[float, float, float] | None,
    apply_height_offset: bool,
) -> tuple[tuple[float, float, float], float]:
    resolved_size = table_size
    resolved_base_z = base_table_z

    if scene_path is not None:
        scene_size, scene_height = _read_scene_table_spec(scene_path)
        resolved_size = resolved_size or scene_size
        resolved_base_z = scene_height if resolved_base_z is None else resolved_base_z

    if resolved_size is None or resolved_base_z is None:
        for candidate in _scene_candidates_from_hdf5(h5):
            if candidate.exists():
                scene_size, scene_height = _read_scene_table_spec(candidate)
                resolved_size = resolved_size or scene_size
                resolved_base_z = scene_height if resolved_base_z is None else resolved_base_z
                break

    if resolved_size is None:
        resolved_size = DEFAULT_TABLE_SIZE
    if resolved_base_z is None:
        resolved_base_z = DEFAULT_TABLE_Z

    if table_z is not None:
        final_table_z = table_z
    else:
        final_table_z = resolved_base_z
        if apply_height_offset:
            final_table_z += _table_height_offset(h5)
    return resolved_size, final_table_z


def _find_hdf5_files(root: Path, *, recursive: bool) -> list[Path]:
    patterns = ("*.hdf5", "*.h5")
    files: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(files))


def _discover_wrist_cameras(
    h5: h5py.File,
    requested: Iterable[str] | None,
    *,
    config_camera_ids: set[str] | None = None,
) -> list[str]:
    cameras = h5.get("cameras")
    if not isinstance(cameras, h5py.Group):
        return []
    def available(camera_id: str) -> bool:
        return camera_id in cameras and (config_camera_ids is None or camera_id in config_camera_ids)

    if requested:
        return [camera_id for camera_id in requested if available(camera_id)]
    discovered = [camera_id for camera_id in DEFAULT_WRIST_CAMERA_IDS if available(camera_id)]
    if discovered:
        return discovered
    return sorted(camera_id for camera_id in cameras.keys() if "wrist" in camera_id.lower() and available(camera_id))


def _sphere_aabb_overlap(centers: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return np.zeros((centers.shape[0],), dtype=np.bool_)
    closest = np.minimum(np.maximum(centers, aabb_min[None, :]), aabb_max[None, :])
    dist2 = np.sum((centers - closest) ** 2, axis=1)
    return dist2 <= (radius * radius)


def _camera_probe_points(extrinsic_frame: np.ndarray, *, probe_depth: float, camera_radius: float) -> np.ndarray:
    origin = np.asarray(extrinsic_frame[:3, 3], dtype=np.float64)
    rot = np.asarray(extrinsic_frame[:3, :3], dtype=np.float64)
    # Isaac raw camera convention in this codebase: +X forward, +Y left, +Z up.
    forward = rot[:, 0]
    left = rot[:, 1]
    up = rot[:, 2]

    points = [origin]
    for frac in (0.25, 0.5, 0.75, 1.0):
        points.append(origin + forward * (probe_depth * frac))

    lateral = min(max(camera_radius, 0.0), max(probe_depth, 1.0e-6) * 0.5)
    if lateral > 0.0:
        center = origin + forward * min(probe_depth * 0.5, probe_depth)
        points.extend((center + left * lateral, center - left * lateral, center + up * lateral, center - up * lateral))
    return np.asarray(points, dtype=np.float64)


def _min_distance_points_to_segments(
    points: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
) -> tuple[float, int]:
    valid = np.all(np.isfinite(segment_starts), axis=1) & np.all(np.isfinite(segment_ends), axis=1)
    if not np.any(valid) or points.size == 0:
        return math.inf, -1

    starts = segment_starts[valid].astype(np.float64)
    ends = segment_ends[valid].astype(np.float64)
    valid_indices = np.flatnonzero(valid)
    seg_vec = ends - starts
    denom = np.sum(seg_vec * seg_vec, axis=1)
    denom_safe = np.where(denom > 1.0e-12, denom, 1.0)

    rel = points[:, None, :] - starts[None, :, :]
    t = np.sum(rel * seg_vec[None, :, :], axis=2) / denom_safe[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    flat_idx = int(np.argmin(dist))
    point_idx, segment_idx = np.unravel_index(flat_idx, dist.shape)
    del point_idx
    return float(dist.reshape(-1)[flat_idx]), int(valid_indices[segment_idx])


def _probe_robot_penetration(
    extrinsic: np.ndarray,
    robot_cache: RobotKinematicCache | None,
    *,
    robot_error: str,
    collision_radius: float,
    severe_radius: float,
    probe_depth: float,
    camera_radius: float,
) -> RobotProbeResult:
    if robot_error:
        return RobotProbeResult(status="warning", error=robot_error)
    if robot_cache is None:
        return RobotProbeResult()

    frame_count = min(int(extrinsic.shape[0]), robot_cache.frame_count)
    if frame_count <= 0:
        return RobotProbeResult(status="warning", error="no overlapping camera/robot frames")

    min_distance = math.inf
    worst_frame = -1
    nearest_segment_idx = -1
    penetration_frames = 0
    severe_frames = 0

    for frame_idx in range(frame_count):
        if not np.all(np.isfinite(extrinsic[frame_idx])):
            continue
        points = _camera_probe_points(
            extrinsic[frame_idx],
            probe_depth=probe_depth,
            camera_radius=camera_radius,
        )
        distance, segment_idx = _min_distance_points_to_segments(
            points,
            robot_cache.segment_starts[frame_idx],
            robot_cache.segment_ends[frame_idx],
        )
        if distance <= collision_radius:
            penetration_frames += 1
        if distance <= severe_radius:
            severe_frames += 1
        if distance < min_distance:
            min_distance = distance
            worst_frame = frame_idx
            nearest_segment_idx = segment_idx

    if math.isinf(min_distance):
        return RobotProbeResult(status="warning", error="no finite robot segment distance")

    nearest = ""
    if 0 <= nearest_segment_idx < len(robot_cache.segment_labels):
        nearest = robot_cache.segment_labels[nearest_segment_idx]
    clearance = min_distance - collision_radius
    if severe_frames > 0:
        status = "penetration"
    elif penetration_frames > 0:
        status = "possible_body_overlap"
    else:
        status = "ok"
    return RobotProbeResult(
        status=status,
        min_distance_m=float(min_distance),
        min_clearance_m=float(clearance),
        worst_frame=int(worst_frame),
        penetration_frames=int(penetration_frames),
        severe_frames=int(severe_frames),
        nearest_link=nearest,
    )


def _audit_camera(
    file_path: Path,
    camera_id: str,
    *,
    extrinsic: np.ndarray,
    pose_source: str,
    parent_link: str,
    table_size: tuple[float, float, float],
    table_z: float,
    camera_radius: float,
    min_clearance: float,
    xy_margin: float,
    z_epsilon: float,
    robot_cache: RobotKinematicCache | None,
    robot_error: str,
    robot_collision_radius: float,
    robot_severe_radius: float,
    robot_probe_depth: float,
) -> CameraAudit:
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (4, 4):
        raise ValueError(f"{camera_id} extrinsic has shape {extrinsic.shape}, expected T x 4 x 4")

    positions = extrinsic[:, :3, 3]
    finite_positions = np.all(np.isfinite(positions), axis=1)
    rotations = extrinsic[:, :3, :3]
    finite_rotations = np.all(np.isfinite(rotations), axis=(1, 2))
    invalid = ~(finite_positions & finite_rotations)

    sx, sy = float(table_size[0]), float(table_size[1])
    half_x = sx * 0.5
    half_y = sy * 0.5
    xy_inside = (
        (positions[:, 0] >= -half_x - xy_margin)
        & (positions[:, 0] <= half_x + xy_margin)
        & (positions[:, 1] >= -half_y - xy_margin)
        & (positions[:, 1] <= half_y + xy_margin)
        & finite_positions
    )
    top_clearance = positions[:, 2] - table_z
    origin_penetration = xy_inside & (top_clearance <= z_epsilon)

    aabb_min = np.asarray([-half_x, -half_y, 0.0], dtype=np.float64)
    aabb_max = np.asarray([half_x, half_y, table_z], dtype=np.float64)
    body_overlap = _sphere_aabb_overlap(positions, aabb_min, aabb_max, camera_radius) & finite_positions
    low_clearance = xy_inside & (top_clearance < min_clearance)

    valid_clearance = np.where(finite_positions, top_clearance, np.inf)
    worst_frame = int(np.argmin(valid_clearance)) if valid_clearance.size else -1
    min_origin_z = float(np.nanmin(np.where(finite_positions, positions[:, 2], np.nan)))
    min_top_clearance = float(np.nanmin(np.where(finite_positions, top_clearance, np.nan)))

    reasons: list[str] = []
    if np.any(origin_penetration):
        reasons.append("origin_below_or_at_table_surface")
    if np.any(body_overlap):
        reasons.append("camera_body_sphere_overlaps_table_aabb")
    if np.any(low_clearance):
        reasons.append("low_origin_clearance")
    if np.any(invalid):
        reasons.append("invalid_transform")

    if np.any(origin_penetration):
        table_status = "penetration"
    elif np.any(body_overlap):
        table_status = "possible_body_overlap"
    elif np.any(low_clearance) or np.any(invalid):
        table_status = "warning"
    else:
        table_status = "ok"

    robot_result = _probe_robot_penetration(
        extrinsic,
        robot_cache,
        robot_error=robot_error,
        collision_radius=robot_collision_radius,
        severe_radius=robot_severe_radius,
        probe_depth=robot_probe_depth,
        camera_radius=camera_radius,
    )
    if robot_result.status == "penetration":
        reasons.append("robot_probe_severe_intersection")
    elif robot_result.status == "possible_body_overlap":
        reasons.append("robot_probe_within_collision_radius")
    elif robot_result.error:
        reasons.append("robot_check_error")

    status = max((table_status, robot_result.status), key=_severity_rank)

    return CameraAudit(
        file=str(file_path),
        camera_id=camera_id,
        pose_source=pose_source,
        parent_link=parent_link,
        frame_count=int(extrinsic.shape[0]),
        table_z=float(table_z),
        table_size_x=sx,
        table_size_y=sy,
        min_origin_z=min_origin_z,
        min_top_clearance_m=min_top_clearance,
        worst_frame=worst_frame,
        origin_penetration_frames=int(np.count_nonzero(origin_penetration)),
        body_overlap_frames=int(np.count_nonzero(body_overlap)),
        low_clearance_frames=int(np.count_nonzero(low_clearance)),
        invalid_transform_frames=int(np.count_nonzero(invalid)),
        table_status=table_status,
        robot_status=robot_result.status,
        robot_min_distance_m=float(robot_result.min_distance_m),
        robot_min_clearance_m=float(robot_result.min_clearance_m),
        robot_worst_frame=int(robot_result.worst_frame),
        robot_penetration_frames=int(robot_result.penetration_frames),
        robot_severe_frames=int(robot_result.severe_frames),
        robot_nearest_link=robot_result.nearest_link,
        robot_error=robot_result.error,
        status=status,
        reasons=",".join(reasons),
    )


def audit_directory(args: argparse.Namespace) -> tuple[list[CameraAudit], list[FileError]]:
    root = Path(args.hdf5_dir).expanduser().resolve()
    files = _find_hdf5_files(root, recursive=not args.no_recursive)
    requested_cameras = None
    if args.camera_ids:
        requested_cameras = [camera_id.strip() for camera_id in args.camera_ids.split(",") if camera_id.strip()]

    scene_path = Path(args.scene).expanduser().resolve() if args.scene else None
    recam_config_path = Path(args.recam_config).expanduser().resolve() if args.recam else None
    table_size = args.table_size
    audits: list[CameraAudit] = []
    errors: list[FileError] = []
    recam_config_cache: dict[str | None, dict[str, Any]] = {}

    for file_path in files:
        try:
            with h5py.File(file_path, "r") as h5:
                file_table_size, file_table_z = _resolve_table_spec(
                    h5,
                    scene_path=scene_path,
                    table_z=args.table_z,
                    base_table_z=args.base_table_z,
                    table_size=table_size,
                    apply_height_offset=not args.no_height_offset,
                )
                new_configs: dict[str, Any] | None = None
                if args.recam:
                    robot_key = _robot_key_from_hdf5(h5)
                    if robot_key not in recam_config_cache:
                        recam_config_cache[robot_key] = _load_recam_camera_configs(recam_config_path, robot_key)
                    new_configs = recam_config_cache[robot_key]

                robot_cache: RobotKinematicCache | None = None
                robot_error = ""
                if not args.no_robot_check:
                    try:
                        robot_cache = _build_robot_kinematic_cache(
                            h5,
                            table_z=file_table_z,
                            robot_usd_override=args.robot_usd,
                            quiet_usd_warnings=not args.show_usd_warnings,
                        )
                    except Exception as exc:  # noqa: BLE001 - table audit can still run.
                        robot_error = f"{type(exc).__name__}: {exc}"

                camera_ids = _discover_wrist_cameras(
                    h5,
                    requested_cameras,
                    config_camera_ids=None if new_configs is None else set(new_configs.keys()),
                )
                if not camera_ids:
                    errors.append(FileError(str(file_path), "no wrist cameras found"))
                    continue
                for camera_id in camera_ids:
                    if args.recam:
                        assert new_configs is not None
                        if camera_id not in new_configs:
                            raise KeyError(f"{camera_id} is missing from recam config")
                        extrinsic, parent_link = _recam_extrinsic_from_config(
                            h5,
                            camera_id,
                            new_cam_cfg=new_configs[camera_id],
                            apply_new_camera_perturbation=not args.recam_no_camera_perturbation,
                        )
                        pose_source = "recam_config"
                    else:
                        extrinsic_path = f"cameras/{camera_id}/extrinsic_world_from_cam"
                        if extrinsic_path not in h5:
                            raise KeyError(f"missing {extrinsic_path}")
                        extrinsic = np.asarray(h5[extrinsic_path], dtype=np.float64)
                        saved_def = _hdf5_camera_definitions(h5).get(camera_id, {})
                        parent_link = str(saved_def.get("parent_link") or "")
                        pose_source = "hdf5_saved"
                    audits.append(
                        _audit_camera(
                            file_path,
                            camera_id,
                            extrinsic=extrinsic,
                            pose_source=pose_source,
                            parent_link=parent_link,
                            table_size=file_table_size,
                            table_z=file_table_z,
                            camera_radius=args.camera_radius,
                            min_clearance=args.min_clearance,
                            xy_margin=args.xy_margin,
                            z_epsilon=args.z_epsilon,
                            robot_cache=robot_cache,
                            robot_error=robot_error,
                            robot_collision_radius=args.robot_collision_radius,
                            robot_severe_radius=args.robot_severe_radius,
                            robot_probe_depth=args.robot_probe_depth,
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - keep auditing other files.
            errors.append(FileError(str(file_path), f"{type(exc).__name__}: {exc}"))

    return audits, errors


def _write_csv(path: Path, audits: list[CameraAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(audits[0]).keys()) if audits else [field.name for field in CameraAudit.__dataclass_fields__.values()]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            writer.writerow(asdict(audit))


def _write_json(path: Path, audits: list[CameraAudit], errors: list[FileError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audits": [asdict(audit) for audit in audits],
        "errors": [asdict(error) for error in errors],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _severity_rank(status: str) -> int:
    return {
        "skipped": 0,
        "ok": 0,
        "warning": 1,
        "possible_body_overlap": 2,
        "penetration": 3,
    }.get(status, 1)


def _should_fail(audits: list[CameraAudit], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    threshold = {
        "warning": 1,
        "body": 2,
        "penetration": 3,
    }[fail_on]
    return any(_severity_rank(audit.status) >= threshold for audit in audits)


def _fmt_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def _episode_id(file_path: str) -> str:
    stem = Path(file_path).stem
    if stem.startswith("episode_"):
        return stem[len("episode_") :]
    return stem


def _format_episode_ids(files: Iterable[str], *, max_items: int = 80) -> str:
    ids = sorted({_episode_id(file_path) for file_path in files})
    if not ids:
        return "none"
    shown = ids[:max_items]
    suffix = "" if len(ids) <= max_items else f", ... (+{len(ids) - max_items})"
    return ", ".join(shown) + suffix


def _print_report(args: argparse.Namespace, audits: list[CameraAudit], errors: list[FileError]) -> None:
    total_files = len({audit.file for audit in audits} | {error.file for error in errors})
    counts: dict[str, int] = {}
    table_counts: dict[str, int] = {}
    robot_counts: dict[str, int] = {}
    for audit in audits:
        counts[audit.status] = counts.get(audit.status, 0) + 1
        table_counts[audit.table_status] = table_counts.get(audit.table_status, 0) + 1
        robot_counts[audit.robot_status] = robot_counts.get(audit.robot_status, 0) + 1

    interesting = [audit for audit in audits if audit.status != "ok"]
    interesting.sort(key=lambda item: (-_severity_rank(item.status), item.file, item.camera_id))

    # ── Findings first (actionable information) ──────────────────────
    if interesting:
        print(f"Findings (showing up to {args.max_report}):")
        for audit in interesting[: args.max_report]:
            rel_file = os.path.relpath(audit.file, start=Path(args.hdf5_dir).expanduser().resolve())
            print(
                f"{rel_file} {audit.camera_id}: {audit.status}; "
                f"source={audit.pose_source}, "
                f"parent={audit.parent_link or '-'}, "
                f"table_status={audit.table_status}, "
                f"table_z={audit.table_z:.4f}, "
                f"min_z={audit.min_origin_z:.4f}, "
                f"min_clearance={audit.min_top_clearance_m:.4f} m, "
                f"worst_frame={audit.worst_frame}, "
                f"origin_frames={audit.origin_penetration_frames}, "
                f"body_frames={audit.body_overlap_frames}, "
                f"low_clearance_frames={audit.low_clearance_frames}, "
                f"robot_status={audit.robot_status}, "
                f"robot_min_dist={_fmt_float(audit.robot_min_distance_m)} m, "
                f"robot_clearance={_fmt_float(audit.robot_min_clearance_m)} m, "
                f"robot_worst_frame={audit.robot_worst_frame}, "
                f"robot_frames={audit.robot_penetration_frames}, "
                f"robot_severe_frames={audit.robot_severe_frames}, "
                f"robot_nearest={audit.robot_nearest_link or '-'}, "
                f"reasons={audit.reasons}"
            )
        if len(interesting) > args.max_report:
            print(f"... {len(interesting) - args.max_report} more findings hidden by --max-report")
    else:
        print("No wrist-camera table/robot penetration findings.")

    if errors and args.show_errors:
        print(f"\nErrors/skips (showing up to {args.max_report}):")
        for error in errors[: args.max_report]:
            rel_file = os.path.relpath(error.file, start=Path(args.hdf5_dir).expanduser().resolve())
            print(f"{rel_file}: {error.error}")

    severe_table_files = [audit.file for audit in audits if audit.origin_penetration_frames > 0]
    severe_robot_files = [audit.file for audit in audits if audit.robot_severe_frames > 0]
    severe_any_files = set(severe_table_files) | set(severe_robot_files)
    print("")
    print(f"Severe table penetration episode ids: {_format_episode_ids(severe_table_files)}")
    print(f"Severe robot penetration episode ids: {_format_episode_ids(severe_robot_files)}")
    print(f"Severe any penetration episode ids: {_format_episode_ids(severe_any_files)}")

    # ── Summary at the bottom ─────────────────────────────────────────
    print("")
    print(f"Scanned files: {total_files}")
    print(f"Camera audits: {len(audits)}")
    print(f"Pose source: {'recam_config' if args.recam else 'hdf5_saved'}")
    if args.recam:
        print(f"Recam config: {Path(args.recam_config).expanduser().resolve()}")
        print(
            "New camera perturbation: "
            f"{'disabled' if args.recam_no_camera_perturbation else 'reused from HDF5 sample'}"
        )
    print(
        "Thresholds: "
        f"camera_radius={args.camera_radius:.4f} m, "
        f"min_clearance={args.min_clearance:.4f} m, "
        f"xy_margin={args.xy_margin:.4f} m, "
        f"z_epsilon={args.z_epsilon:.4f} m, "
        f"robot_collision_radius={args.robot_collision_radius:.4f} m, "
        f"robot_severe_radius={args.robot_severe_radius:.4f} m, "
        f"robot_probe_depth={args.robot_probe_depth:.4f} m"
    )
    print(
        "Combined status counts: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in ("penetration", "possible_body_overlap", "warning", "ok"))
    )
    print(
        "Table status counts: "
        + ", ".join(f"{key}={table_counts.get(key, 0)}" for key in ("penetration", "possible_body_overlap", "warning", "ok"))
    )
    print(
        "Robot status counts: "
        + ", ".join(
            f"{key}={robot_counts.get(key, 0)}"
            for key in ("penetration", "possible_body_overlap", "warning", "ok", "skipped")
        )
    )
    if args.no_robot_check:
        print("Robot check: disabled by --no-robot-check")
    elif args.robot_usd:
        print(f"Robot USD override: {Path(args.robot_usd).expanduser().resolve()}")
    if errors:
        print(f"File errors/skips: {len(errors)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check all HDF5 files in a directory for wrist-camera table/robot penetration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5_dir", help="Directory containing .hdf5/.h5 episode files.")
    parser.add_argument("--camera-ids", default=None, help="Comma-separated camera ids. Defaults to wrist cameras.")
    parser.add_argument(
        "--recam",
        action="store_true",
        help=(
            "Estimate wrist-camera poses from current mount parameters instead of auditing saved camera poses. "
            "The script recovers the saved parent-link trajectory from old HDF5 camera extrinsics and old HDF5 "
            "mount offsets, then applies --recam-config."
        ),
    )
    parser.add_argument(
        "--recam-config",
        default=str(REPO_ROOT / "configs/collect/default.yaml"),
        help="Collect config used by --recam for new wrist-camera mount parameters.",
    )
    parser.add_argument(
        "--recam-no-camera-perturbation",
        action="store_true",
        help="With --recam, do not apply HDF5 camera offset/rpy generalization to the new mount offset.",
    )
    parser.add_argument("--scene", default=None, help="Scene YAML override used for table size/base height.")
    parser.add_argument("--table-z", type=float, default=None, help="Final table surface z override, after generalization.")
    parser.add_argument("--base-table-z", type=float, default=None, help="Base table z before HDF5 height_offset_m.")
    parser.add_argument(
        "--table-size",
        type=lambda value: _parse_float_triplet(value, default=DEFAULT_TABLE_SIZE),
        default=None,
        help="Table size as x,y or x,y,z in meters.",
    )
    parser.add_argument("--no-height-offset", action="store_true", help="Do not add HDF5 scene_generalization_sample table offset.")
    parser.add_argument("--camera-radius", type=float, default=0.03, help="Conservative wrist-camera body sphere radius in meters.")
    parser.add_argument("--min-clearance", type=float, default=0.02, help="Warn when optical center clearance above table is below this.")
    parser.add_argument("--xy-margin", type=float, default=0.0, help="Expand table XY footprint for origin/clearance checks.")
    parser.add_argument("--z-epsilon", type=float, default=1.0e-4, help="Tolerance for origin-at-or-below-table penetration.")
    parser.add_argument("--no-robot-check", action="store_true", help="Disable robot FK/proximity penetration checks.")
    parser.add_argument("--robot-usd", default=None, help="Robot USD override for FK-based robot penetration checks.")
    parser.add_argument(
        "--robot-collision-radius",
        type=float,
        default=0.035,
        help="Combined camera/link radius threshold for possible robot body overlap.",
    )
    parser.add_argument(
        "--robot-severe-radius",
        type=float,
        default=0.008,
        help="Distance threshold for severe robot penetration.",
    )
    parser.add_argument(
        "--robot-probe-depth",
        type=float,
        default=0.03,
        help="Depth of the near camera/view-axis probe used for robot penetration checks.",
    )
    parser.add_argument("--show-usd-warnings", action="store_true", help="Do not suppress USD warnings while loading robot FK.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan files directly under hdf5_dir.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV report path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-report", type=int, default=50, help="Maximum findings/errors printed to stdout.")
    parser.add_argument("--show-errors", action="store_true", help="Print skipped/error files.")
    parser.add_argument(
        "--fail-on",
        choices=("none", "warning", "body", "penetration"),
        default="body",
        help="Exit nonzero at or above this severity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.camera_radius) or args.camera_radius < 0.0:
        parser.error("--camera-radius must be a finite non-negative number")
    if not math.isfinite(args.min_clearance):
        parser.error("--min-clearance must be finite")
    if not math.isfinite(args.xy_margin) or args.xy_margin < 0.0:
        parser.error("--xy-margin must be a finite non-negative number")
    if not math.isfinite(args.z_epsilon) or args.z_epsilon < 0.0:
        parser.error("--z-epsilon must be a finite non-negative number")
    if not math.isfinite(args.robot_collision_radius) or args.robot_collision_radius < 0.0:
        parser.error("--robot-collision-radius must be a finite non-negative number")
    if not math.isfinite(args.robot_severe_radius) or args.robot_severe_radius < 0.0:
        parser.error("--robot-severe-radius must be a finite non-negative number")
    if args.robot_severe_radius > args.robot_collision_radius:
        parser.error("--robot-severe-radius must be <= --robot-collision-radius")
    if not math.isfinite(args.robot_probe_depth) or args.robot_probe_depth <= 0.0:
        parser.error("--robot-probe-depth must be a finite positive number")

    audits, errors = audit_directory(args)
    if args.csv_output:
        _write_csv(Path(args.csv_output).expanduser(), audits)
    if args.json_output:
        _write_json(Path(args.json_output).expanduser(), audits, errors)
    _print_report(args, audits, errors)
    return 1 if _should_fail(audits, args.fail_on) else 0


if __name__ == "__main__":
    sys.exit(main())
