"""HDF5 access and rendering helpers for the standalone episode viewer."""

from __future__ import annotations

import io
import json
import math
import os
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


DEFAULT_HDF5 = Path(
    "../test_teleop_data/hdf5/sharpa/episode_000000_replay.hdf5"
)

CAMERA_ORDER = [
    "cam_chest",
    "cam_overhead",
    "cam_stereo_left",
    "cam_stereo_right",
    "cam_wrist_left",
    "cam_wrist_right",
]

TACTILE_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

BOX_EDGES = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

CORNER_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

OBJECT_COLORS = [
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (120, 120, 120),
]


def occupancy_semantic_color(semantic_id: int) -> tuple[int, int, int]:
    if semantic_id <= 0:
        return (235, 235, 235)
    if semantic_id == 1:
        return (175, 178, 180)
    return OBJECT_COLORS[(semantic_id - 2) % len(OBJECT_COLORS)]


OCCUPANCY_LABEL_KEYS = ("occupancy_gt", "occupancy_tsdf", "occupancy")
OCCUPANCY_UNKNOWN = 0
OCCUPANCY_FREE = 1
OCCUPANCY_OCCUPIED = 2
MAX_OCCUPANCY_POINTS_PER_VIEW = 80000


class CameraFrame:
    """Minimal camera geometry pulled entirely from the HDF5 file."""

    def __init__(
        self,
        *,
        intrinsic: np.ndarray,
        extrinsic_world_from_cam: np.ndarray,
        image_shape: tuple[int, int],
        camera_model: str = "pinhole",
        fisheye_camera_matrix: np.ndarray | None = None,
        distortion_coefficients: np.ndarray | None = None,
    ) -> None:
        self.intrinsic = intrinsic
        self.extrinsic_world_from_cam = extrinsic_world_from_cam
        self.image_shape = image_shape
        self.camera_model = camera_model
        self.fisheye_camera_matrix = fisheye_camera_matrix
        self.distortion_coefficients = distortion_coefficients


def isaac_raw_camera_to_optical(points_raw_camera: np.ndarray) -> np.ndarray:
    """Convert Isaac camera-body axes (x forward, y left, z up) to CV optical axes."""
    points_optical = np.empty_like(points_raw_camera, dtype=np.float32)
    points_optical[:, 0] = -points_raw_camera[:, 1]
    points_optical[:, 1] = -points_raw_camera[:, 2]
    points_optical[:, 2] = points_raw_camera[:, 0]
    return points_optical


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def world_points_to_camera(
    points_world: np.ndarray,
    extrinsic_world_from_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    cam_pos_w = np.asarray(extrinsic_world_from_cam[:3, 3], dtype=np.float32)
    rot_world_from_cam = np.asarray(extrinsic_world_from_cam[:3, :3], dtype=np.float32)
    points_raw_camera = (points - cam_pos_w[None, :]) @ rot_world_from_cam
    points_cam = isaac_raw_camera_to_optical(points_raw_camera)
    valid = points_cam[:, 2] > 1.0e-6
    return points_cam.astype(np.float32), valid.astype(np.bool_)


def project_camera_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    z = points_cam[:, 2]
    valid = z > 1.0e-6
    if np.any(valid):
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        uv[valid, 0] = fx * points_cam[valid, 0] / z[valid] + cx
        uv[valid, 1] = fy * points_cam[valid, 1] / z[valid] + cy
    return uv


def project_camera_points_fisheye(
    points_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    valid = z > 1.0e-6
    if not np.any(valid):
        return uv

    xv, yv, zv = x[valid], y[valid], z[valid]
    r = np.sqrt(xv * xv + yv * yv)
    theta = np.arctan2(r, zv)
    k1, k2, k3, k4 = np.asarray(distortion_coefficients, dtype=np.float32)[:4]
    theta2 = theta * theta
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2 * theta2
        + k3 * theta2 * theta2 * theta2
        + k4 * theta2 * theta2 * theta2 * theta2
    )
    on_axis = r < 1.0e-8
    safe_r = np.where(on_axis, 1.0, r)
    scale = np.where(on_axis, 1.0 / zv, theta_d / safe_r)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    uv[valid, 0] = fx * scale * xv + cx
    uv[valid, 1] = fy * scale * yv + cy
    return uv


def project_world_points_to_image(
    points_world: np.ndarray,
    camera_frame: CameraFrame,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam, valid = world_points_to_camera(
        points_world,
        camera_frame.extrinsic_world_from_cam,
    )
    if (
        camera_frame.camera_model == "fisheye"
        and camera_frame.fisheye_camera_matrix is not None
        and camera_frame.distortion_coefficients is not None
    ):
        uv = project_camera_points_fisheye(
            points_cam,
            camera_frame.fisheye_camera_matrix,
            camera_frame.distortion_coefficients,
        )
    else:
        uv = project_camera_points(points_cam, camera_frame.intrinsic)
    return uv.astype(np.float32), valid.astype(np.bool_)


def decode_hdf5_value(value: Any) -> Any:
    """Convert common HDF5 scalar/string values to plain Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_hdf5_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_string_list(dataset: h5py.Dataset | None) -> list[str]:
    if dataset is None:
        return []
    value = dataset[()]
    decoded = decode_hdf5_value(value)
    if isinstance(decoded, str):
        stripped = decoded.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return [str(v) for v in parsed]
            except Exception:
                pass
        return [decoded] if decoded else []
    if isinstance(value, np.ndarray):
        return [str(decode_hdf5_value(v)) for v in value]
    if isinstance(decoded, (list, tuple)):
        return [str(v) for v in decoded]
    return [str(decoded)]


def read_meta(file: h5py.File, key: str, default: Any = None) -> Any:
    dataset = file.get(f"meta/{key}")
    if not isinstance(dataset, h5py.Dataset):
        return default
    return decode_hdf5_value(dataset[()])


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    decoded = decode_hdf5_value(value)
    if not isinstance(decoded, str):
        return decoded
    stripped = decoded.strip()
    if not stripped:
        return default
    try:
        return json.loads(stripped)
    except Exception:
        return default


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:
        return None


def dataset_bool(file: h5py.File, path: str, idx: int | None = None) -> bool | None:
    dataset = file.get(path)
    if not isinstance(dataset, h5py.Dataset):
        return None
    try:
        value = dataset[()] if idx is None or dataset.shape == () else dataset[idx]
    except Exception:
        return None
    return safe_bool(decode_hdf5_value(value))


def dataset_scalar(file: h5py.File, path: str, idx: int | None = None, default: Any = None) -> Any:
    dataset = file.get(path)
    if not isinstance(dataset, h5py.Dataset):
        return default
    try:
        value = dataset[()] if idx is None or dataset.shape == () else dataset[idx]
    except Exception:
        return default
    return decode_hdf5_value(value)


def group_scalar(group: h5py.Group, key: str, default: Any = None) -> Any:
    dataset = group.get(key)
    if not isinstance(dataset, h5py.Dataset):
        return default
    try:
        return decode_hdf5_value(dataset[()])
    except Exception:
        return default


def to_float_list(array: Any, digits: int = 5) -> list[float | None]:
    values = np.asarray(array).reshape(-1)
    out: list[float | None] = []
    for value in values:
        f = finite_float(value)
        out.append(None if f is None else round(f, digits))
    return out


def short_name(name: str) -> str:
    return (
        name.replace("multi_right_", "R.")
        .replace("right_", "R.")
        .replace("left_", "L.")
        .replace("multi_", "R.")
        .replace("_joint", "")
    )


def joint_group_name(name: str) -> str:
    if name.startswith("left_"):
        return "left_hand"
    if name.startswith("right_") or name.startswith("multi_right_"):
        return "right_hand"
    if name.startswith("multi_A"):
        return "right_arm"
    if name.startswith("A") and any(ch.isdigit() for ch in name):
        return "left_arm"
    return "other"


def group_label(group: str) -> str:
    return {
        "left_arm": "Left Arm",
        "right_arm": "Right Arm",
        "left_hand": "Left Hand",
        "right_hand": "Right Hand",
        "other": "Other",
    }.get(group, group)


def ordered_camera_ids(ids: list[str]) -> list[str]:
    known = [cam for cam in CAMERA_ORDER if cam in ids]
    extra = sorted(cam for cam in ids if cam not in CAMERA_ORDER)
    return known + extra


def tactile_sort_key(site: str) -> tuple[int, int, str]:
    side = 0 if site.startswith("left_") else 1 if site.startswith("right_") else 2
    finger_idx = len(TACTILE_FINGER_ORDER)
    for idx, finger in enumerate(TACTILE_FINGER_ORDER):
        if finger in site:
            finger_idx = idx
            break
    return (side, finger_idx, site)


def find_depth_key(camera_group: h5py.Group) -> str | None:
    if "depth" in camera_group:
        return "depth"
    if "depth_m" in camera_group:
        return "depth_m"
    return None


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def placeholder_png(title: str, detail: str = "", size: tuple[int, int] = (640, 480)) -> bytes:
    image = Image.new("RGB", size, (32, 34, 36))
    draw = ImageDraw.Draw(image)
    lines = [title]
    if detail:
        lines.extend(detail[i : i + 72] for i in range(0, len(detail), 72))
    y = max(20, size[1] // 2 - 14 * len(lines))
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        x = max(12, (size[0] - (bbox[2] - bbox[0])) // 2)
        draw.text((x, y), line, fill=(225, 230, 232))
        y += 24
    return image_bytes(image)


def colormap_array(
    values: np.ndarray,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    valid_mask: np.ndarray | None = None,
    zero_is_black: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if valid_mask is None:
        valid = np.isfinite(array)
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_) & np.isfinite(array)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.zeros(array.shape, dtype=np.float32)
    norm[valid] = np.clip((array[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    if zero_is_black:
        rgb[array <= 0] = 0
    return rgb


def robust_limits(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 32:
        lo = float(np.percentile(finite, lower))
        hi = float(np.percentile(finite, upper))
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


class EpisodeHDF5:
    """Lazy reader for one HDF5 episode."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._summary_cache: dict[str, Any] | None = None
        self._contact_cache: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _open(self) -> h5py.File:
        return h5py.File(self.path, "r")

    def _require_file(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

    def _camera_definitions(self, file: h5py.File) -> dict[str, dict[str, Any]]:
        definitions = parse_jsonish(read_meta(file, "camera_definitions", None), [])
        if not isinstance(definitions, list):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for item in definitions:
            if isinstance(item, dict) and item.get("camera_id"):
                out[str(item["camera_id"])] = item
        return out

    def _frame_count(self, file: h5py.File) -> int:
        meta_count = read_meta(file, "frame_count", None)
        if meta_count is not None:
            try:
                return int(meta_count)
            except Exception:
                pass
        for path in ("frame_valid", "robot/qpos", "action/commanded"):
            dataset = file.get(path)
            if isinstance(dataset, h5py.Dataset) and dataset.shape:
                return int(dataset.shape[0])
        cameras = file.get("cameras")
        if isinstance(cameras, h5py.Group):
            for cam_group in cameras.values():
                if isinstance(cam_group, h5py.Group):
                    for key in ("rgb", "depth", "depth_m"):
                        if key in cam_group and cam_group[key].shape:
                            return int(cam_group[key].shape[0])
        tactile = file.get("robot/tactile/tacmap")
        if isinstance(tactile, h5py.Group):
            for dataset in tactile.values():
                if isinstance(dataset, h5py.Dataset) and dataset.shape:
                    return int(dataset.shape[0])
        return 0

    def _clamp_idx(self, file: h5py.File, idx: int) -> int:
        frame_count = self._frame_count(file)
        if frame_count <= 0:
            return 0
        return max(0, min(int(idx), frame_count - 1))

    def _read_names(self, file: h5py.File, path: str, meta_key: str | None = None) -> list[str]:
        dataset = file.get(path)
        if isinstance(dataset, h5py.Dataset):
            names = read_string_list(dataset)
            if len(names) > 1:
                return names
        if meta_key is not None:
            parsed = parse_jsonish(read_meta(file, meta_key, None), [])
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        return []

    def _read_tactile_sites(self, file: h5py.File) -> list[str]:
        sites = self._read_names(file, "robot/tactile/meta/site_names")
        tacmap = file.get("robot/tactile/tacmap")
        if isinstance(tacmap, h5py.Group):
            present = set(tacmap.keys())
            sites = [site for site in sites if site in present]
            sites.extend(sorted(site for site in present if site not in sites))
        return sorted(sites, key=tactile_sort_key)

    def _occupancy_group(self, file: h5py.File) -> tuple[str | None, h5py.Group | None]:
        labels = file.get("labels")
        if not isinstance(labels, h5py.Group):
            return None, None
        for key in OCCUPANCY_LABEL_KEYS:
            group = labels.get(key)
            if isinstance(group, h5py.Group) and isinstance(group.get("state"), h5py.Dataset):
                return key, group
        return None, None

    def _occupancy_info(self, file: h5py.File) -> dict[str, Any]:
        key, group = self._occupancy_group(file)
        if group is None:
            return {"available": False}

        state = group.get("state")
        grid_shape_dataset = group.get("grid_shape")
        if isinstance(grid_shape_dataset, h5py.Dataset):
            grid_shape = [int(v) for v in np.asarray(grid_shape_dataset[()], dtype=np.int32).reshape(-1)[:3]]
        elif isinstance(state, h5py.Dataset) and len(state.shape) >= 4:
            grid_shape = [int(v) for v in state.shape[1:4]]
        else:
            grid_shape = []

        bounds_dataset = group.get("bounds")
        bounds = None
        if isinstance(bounds_dataset, h5py.Dataset):
            bounds_array = np.asarray(bounds_dataset[()], dtype=np.float32)
            if bounds_array.shape == (2, 3):
                bounds = [[round(float(v), 5) for v in row] for row in bounds_array]

        frame_valid = group.get("frame_valid")
        valid_count = None
        if isinstance(frame_valid, h5py.Dataset):
            valid_count = int(np.count_nonzero(frame_valid[:]))

        legend = parse_jsonish(group_scalar(group, "semantic_legend", None), {})
        if not isinstance(legend, dict):
            legend = {}

        return {
            "available": True,
            "label_key": key,
            "method": str(group_scalar(group, "method", "") or ""),
            "semantics": str(group_scalar(group, "semantics", "") or ""),
            "grid_shape": grid_shape,
            "bounds": bounds,
            "voxel_size": finite_float(group_scalar(group, "voxel_size", None)),
            "frame_count": int(state.shape[0]) if isinstance(state, h5py.Dataset) and state.shape else 0,
            "valid_frame_count": valid_count,
            "has_semantic_id": isinstance(group.get("semantic_id"), h5py.Dataset),
            "semantic_legend": {str(k): str(v) for k, v in legend.items()},
        }

    def _generalization_summary(self, file: h5py.File) -> list[dict[str, str]]:
        sample = parse_jsonish(read_meta(file, "scene_generalization_sample", None), {})
        if not isinstance(sample, dict):
            return []
        appearance = sample.get("appearance", {}) if isinstance(sample.get("appearance"), dict) else {}
        spatial = sample.get("spatial", {}) if isinstance(sample.get("spatial"), dict) else {}
        rows: list[dict[str, str]] = []

        background = appearance.get("background", {}) if isinstance(appearance.get("background"), dict) else {}
        if background:
            asset = background.get("asset_uri") or background.get("asset_name") or "n/a"
            category = background.get("asset_category") or background.get("asset_kind") or "background"
            rows.append(
                {
                    "label": "Background",
                    "value": f"{category} | {asset}",
                }
            )

        table = appearance.get("table_surface", {}) if isinstance(appearance.get("table_surface"), dict) else {}
        if table:
            rows.append(
                {
                    "label": "Table Texture",
                    "value": str(table.get("asset_uri") or table.get("asset_name") or "n/a"),
                }
            )

        light = appearance.get("light", {}) if isinstance(appearance.get("light"), dict) else {}
        if light:
            intensity = finite_float(light.get("intensity"))
            color = light.get("color")
            color_text = ""
            if isinstance(color, list) and len(color) >= 3:
                color_text = " color " + ", ".join(f"{float(c):.2f}" for c in color[:3])
            rows.append(
                {
                    "label": "Lighting",
                    "value": f"intensity {intensity:.1f}{color_text}" if intensity is not None else str(light),
                }
            )

        camera = spatial.get("camera", {}) if isinstance(spatial.get("camera"), dict) else {}
        cameras = camera.get("cameras", {}) if isinstance(camera.get("cameras"), dict) else {}
        if cameras:
            enabled = [cam for cam, cfg in cameras.items() if not isinstance(cfg, dict) or cfg.get("enabled", True)]
            rows.append(
                {
                    "label": "Camera Perturbation",
                    "value": f"{len(enabled)} cameras randomized: {', '.join(enabled[:6])}",
                }
            )

        replay_mode = sample.get("_replay_generalization_mode")
        if replay_mode:
            rows.append({"label": "Replay Mode", "value": str(replay_mode)})
        return rows

    def summary(self) -> dict[str, Any]:
        with self._lock:
            if self._summary_cache is not None:
                return self._summary_cache

        self._require_file()
        with self._open() as file:
            frame_count = self._frame_count(file)
            camera_group = file.get("cameras")
            camera_ids = ordered_camera_ids(list(camera_group.keys())) if isinstance(camera_group, h5py.Group) else []
            camera_definitions = self._camera_definitions(file)

            camera_info: dict[str, dict[str, Any]] = {}
            for cam_id in camera_ids:
                cam_group = file[f"cameras/{cam_id}"]
                definition = camera_definitions.get(cam_id, {})
                depth_key = find_depth_key(cam_group)
                camera_type = dataset_scalar(file, f"cameras/{cam_id}/camera_model", default=None)
                if camera_type is None:
                    camera_type = definition.get("camera_type", "pinhole")
                height = definition.get("height")
                width = definition.get("width")
                if depth_key is not None and len(cam_group[depth_key].shape) >= 3:
                    height, width = [int(v) for v in cam_group[depth_key].shape[1:3]]
                camera_info[cam_id] = {
                    "camera_type": str(camera_type),
                    "has_rgb": "rgb" in cam_group,
                    "has_depth": depth_key is not None,
                    "depth_key": depth_key,
                    "width": int(width) if width is not None else None,
                    "height": int(height) if height is not None else None,
                    "mount_type": definition.get("mount_type"),
                    "parent_link": definition.get("parent_link"),
                }

            object_ids = sorted(file["objects"].keys()) if isinstance(file.get("objects"), h5py.Group) else []
            object_roles = parse_jsonish(read_meta(file, "object_roles", None), {})
            object_body_types = parse_jsonish(read_meta(file, "object_body_types", None), {})
            if not isinstance(object_roles, dict):
                object_roles = {}
            if not isinstance(object_body_types, dict):
                object_body_types = {}

            joint_names = self._read_names(file, "robot/joint_names")
            action_names = self._read_names(file, "action/action_names", "action_names")
            tactile_sites = self._read_tactile_sites(file)
            occupancy_info = self._occupancy_info(file)

            object_tracks: dict[str, Any] = {}
            track_bounds = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
            step = max(1, frame_count // 300) if frame_count else 1
            objects_group = file.get("objects")
            if isinstance(objects_group, h5py.Group):
                all_x: list[float] = []
                all_y: list[float] = []
                for object_id in object_ids:
                    pose_dataset = objects_group[object_id].get("pose_world")
                    if not isinstance(pose_dataset, h5py.Dataset) or len(pose_dataset.shape) < 2:
                        continue
                    poses = np.asarray(pose_dataset[::step, :2], dtype=np.float32)
                    mask = np.isfinite(poses).all(axis=1)
                    xy = poses[mask]
                    object_tracks[object_id] = {
                        "step": step,
                        "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy],
                    }
                    if xy.size:
                        all_x.extend(float(v) for v in xy[:, 0])
                        all_y.extend(float(v) for v in xy[:, 1])
                if all_x and all_y:
                    track_bounds = {
                        "xmin": round(min(all_x), 4),
                        "xmax": round(max(all_x), 4),
                        "ymin": round(min(all_y), 4),
                        "ymax": round(max(all_y), 4),
                    }

            metadata = {
                "instruction": str(read_meta(file, "instruction", "")),
                "scene_name": str(read_meta(file, "scene_name", "")),
                "robot_key": str(read_meta(file, "robot_key", "")),
                "fps": int(read_meta(file, "fps", read_meta(file, "effective_fps", 0)) or 0),
                "effective_fps": finite_float(read_meta(file, "effective_fps", None)),
                "frame_count": frame_count,
                "schema_version": str(read_meta(file, "schema_version", "")),
                "success": safe_bool(read_meta(file, "success", None)),
                "task_name": str(read_meta(file, "task_name", "")),
                "dataset_version": str(read_meta(file, "dataset_version", "")),
                "source_domain": str(read_meta(file, "source_domain", "")),
                "created_at": str(read_meta(file, "created_at", "")),
            }

            frame_valid = file.get("frame_valid")
            valid_count = None
            if isinstance(frame_valid, h5py.Dataset):
                valid_count = int(np.count_nonzero(frame_valid[:]))

            summary = {
                "path": str(self.path),
                "file_name": self.path.name,
                "metadata": metadata,
                "frame_count": frame_count,
                "camera_ids": camera_ids,
                "camera_info": camera_info,
                "tactile_sites": tactile_sites,
                "object_ids": object_ids,
                "object_roles": object_roles,
                "object_body_types": object_body_types,
                "occupancy_info": occupancy_info,
                "box3d_object_ids": sorted(file["labels/box3d"].keys())
                if isinstance(file.get("labels/box3d"), h5py.Group)
                else [],
                "box2d_camera_ids": sorted(file["labels/box2d"].keys())
                if isinstance(file.get("labels/box2d"), h5py.Group)
                else [],
                "robot_joint_names": joint_names,
                "action_names": action_names,
                "joint_group_counts": self._group_counts(joint_names),
                "action_group_counts": self._group_counts(action_names),
                "generalization": self._generalization_summary(file),
                "object_tracks": object_tracks,
                "track_bounds": track_bounds,
                "valid_frame_count": valid_count,
                "has": {
                    "rgb": any(info["has_rgb"] for info in camera_info.values()),
                    "depth": any(info["has_depth"] for info in camera_info.values()),
                    "box3d": isinstance(file.get("labels/box3d"), h5py.Group),
                    "box2d": isinstance(file.get("labels/box2d"), h5py.Group),
                    "occupancy": bool(occupancy_info.get("available")),
                    "tactile": bool(tactile_sites),
                    "robot": isinstance(file.get("robot/qpos"), h5py.Dataset),
                    "action": isinstance(file.get("action/commanded"), h5py.Dataset),
                    "objects": bool(object_ids),
                },
            }

        with self._lock:
            self._summary_cache = summary
        return summary

    def _group_counts(self, names: list[str]) -> dict[str, int]:
        counts = {"left_arm": 0, "right_arm": 0, "left_hand": 0, "right_hand": 0, "other": 0}
        for name in names:
            counts[joint_group_name(name)] += 1
        return counts

    def _group_rows(
        self,
        names: list[str],
        *,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        qeffort: np.ndarray | None = None,
        action: np.ndarray | None = None,
        action_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        action_map: dict[str, float | None] = {}
        if action is not None and action_names:
            for idx, name in enumerate(action_names):
                if idx < len(action):
                    value = finite_float(action[idx])
                    action_map[name] = None if value is None else round(value, 5)

        grouped: dict[str, list[dict[str, Any]]] = {
            "left_arm": [],
            "right_arm": [],
            "left_hand": [],
            "right_hand": [],
            "other": [],
        }
        for idx, name in enumerate(names):
            row = {
                "index": idx,
                "name": short_name(name),
                "full_name": name,
                "qpos": None,
                "qvel": None,
                "qeffort": None,
                "action": action_map.get(name),
            }
            if qpos is not None and idx < len(qpos):
                row["qpos"] = finite_float(qpos[idx])
            if qvel is not None and idx < len(qvel):
                row["qvel"] = finite_float(qvel[idx])
            if qeffort is not None and idx < len(qeffort):
                row["qeffort"] = finite_float(qeffort[idx])
            for key in ("qpos", "qvel", "qeffort"):
                if row[key] is not None:
                    row[key] = round(float(row[key]), 5)
            grouped[joint_group_name(name)].append(row)

        return [
            {"id": key, "label": group_label(key), "rows": rows}
            for key, rows in grouped.items()
            if rows
        ]

    def frame(self, idx: int) -> dict[str, Any]:
        summary = self.summary()
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            frame_count = self._frame_count(file)
            object_roles = summary.get("object_roles", {})
            object_body_types = summary.get("object_body_types", {})

            objects: list[dict[str, Any]] = []
            objects_group = file.get("objects")
            if isinstance(objects_group, h5py.Group):
                for object_id in sorted(objects_group.keys()):
                    group = objects_group[object_id]
                    pose_dataset = group.get("pose_world")
                    if not isinstance(pose_dataset, h5py.Dataset):
                        continue
                    pose = np.asarray(pose_dataset[idx], dtype=np.float32)
                    if pose.shape[0] < 7:
                        continue
                    lin_vel = self._read_vector_dataset(group, idx, ["lin_vel_world", "linear_velocity"])
                    ang_vel = self._read_vector_dataset(group, idx, ["ang_vel_world", "angular_velocity"])
                    item: dict[str, Any] = {
                        "id": object_id,
                        "role": str(object_roles.get(object_id, "")),
                        "body_type": str(object_body_types.get(object_id, "")),
                        "xyz": to_float_list(pose[:3]),
                        "quat_xyzw": to_float_list(pose[3:7]),
                        "lin_vel": to_float_list(lin_vel) if lin_vel is not None else [],
                        "ang_vel": to_float_list(ang_vel) if ang_vel is not None else [],
                        "articulation": [],
                    }
                    qpos = group.get("qpos")
                    if isinstance(qpos, h5py.Dataset):
                        qpos_frame = np.asarray(qpos[idx], dtype=np.float32)
                        qvel_frame = np.asarray(group["qvel"][idx], dtype=np.float32) if "qvel" in group else None
                        qeffort_frame = (
                            np.asarray(group["qeffort"][idx], dtype=np.float32) if "qeffort" in group else None
                        )
                        joint_names = read_string_list(group.get("joint_names"))
                        if not joint_names:
                            joint_names = [f"joint_{i}" for i in range(len(qpos_frame))]
                        for j, name in enumerate(joint_names):
                            item["articulation"].append(
                                {
                                    "name": name,
                                    "qpos": round(float(qpos_frame[j]), 5) if j < len(qpos_frame) else None,
                                    "qvel": round(float(qvel_frame[j]), 5)
                                    if qvel_frame is not None and j < len(qvel_frame)
                                    else None,
                                    "qeffort": round(float(qeffort_frame[j]), 5)
                                    if qeffort_frame is not None and j < len(qeffort_frame)
                                    else None,
                                }
                            )
                    objects.append(item)

            joint_names = summary.get("robot_joint_names", [])
            action_names = summary.get("action_names", [])
            qpos = self._read_frame_array(file, "robot/qpos", idx)
            qvel = self._read_frame_array(file, "robot/qvel", idx)
            qeffort = self._read_frame_array(file, "robot/qeffort", idx)
            commanded = self._read_frame_array(file, "action/commanded", idx)

            # Fallback names for inference recordings that omit joint_names /
            # action_names (InferenceRecorder writes robot/joint_names but older
            # recordings may not).  Without a name list the grouped rows come
            # back empty even though qpos/action data exists.
            if not joint_names and qpos is not None:
                joint_names = [f"joint_{i}" for i in range(len(qpos))]
            if not action_names:
                if commanded is not None:
                    if joint_names and len(joint_names) == len(commanded):
                        action_names = joint_names  # action aligned to qpos (full-DOF model)
                    else:
                        action_names = [f"action_{i}" for i in range(len(commanded))]
                else:
                    action_names = list(joint_names)

            tactile_current: dict[str, Any] = {"any_contact": False, "sites": {}}
            tacmap_group = file.get("robot/tactile/tacmap")
            if isinstance(tacmap_group, h5py.Group):
                for site in summary.get("tactile_sites", []):
                    dataset = tacmap_group.get(site)
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    image = np.asarray(dataset[idx], dtype=np.uint8)
                    site_max = int(np.max(image)) if image.size else 0
                    site_mean = float(np.mean(image)) if image.size else 0.0
                    tactile_current["sites"][site] = {
                        "max": site_max,
                        "mean": round(site_mean, 5),
                        "nonzero": bool(site_max > 0),
                    }
                    tactile_current["any_contact"] = bool(tactile_current["any_contact"] or site_max > 0)

            depth_ranges = self._depth_ranges(file, idx)
            box_counts = self._box_counts(file, idx)
            occupancy_current = self._occupancy_counts(file, idx)
            series = self._series_window(file, idx, frame_count)

            frame_payload = {
                "idx": idx,
                "frame_count": frame_count,
                "time": {
                    "frame_index": dataset_scalar(file, "time/frame_index", idx, idx),
                    "sim_step": dataset_scalar(file, "time/sim_step", idx, None),
                    "timestamp_ns": dataset_scalar(file, "time/timestamp_ns", idx, None),
                },
                "status": {
                    "frame_valid": dataset_bool(file, "frame_valid", idx),
                    "action_valid": dataset_bool(file, "action/action_valid", idx),
                    "success": dataset_bool(file, "episode/success", idx)
                    if isinstance(file.get("episode/success"), h5py.Dataset)
                    else summary["metadata"].get("success"),
                    "done": dataset_bool(file, "episode/done", idx),
                    "is_last": dataset_bool(file, "episode/is_last", idx),
                    "frame_error": str(dataset_scalar(file, "frame_errors", idx, "") or ""),
                },
                "objects": objects,
                "robot": {
                    "groups": self._group_rows(
                        joint_names,
                        qpos=qpos,
                        qvel=qvel,
                        qeffort=qeffort,
                        action=commanded,
                        action_names=action_names,
                    ),
                },
                "action": {
                    "valid": dataset_bool(file, "action/action_valid", idx),
                    "groups": self._group_rows(action_names, qpos=commanded),
                    "source": str(dataset_scalar(file, "action/source", idx, "") or ""),
                    "control_mode": str(dataset_scalar(file, "action/control_mode", default="") or ""),
                    "action_type": str(dataset_scalar(file, "action/action_type", default="") or ""),
                },
                "depth_ranges": depth_ranges,
                "box_counts": box_counts,
                "occupancy_current": occupancy_current,
                "tactile_current": tactile_current,
                "series": series,
            }
        return frame_payload

    def _read_vector_dataset(self, group: h5py.Group, idx: int, keys: list[str]) -> np.ndarray | None:
        for key in keys:
            dataset = group.get(key)
            if isinstance(dataset, h5py.Dataset):
                return np.asarray(dataset[idx], dtype=np.float32)
        return None

    def _read_frame_array(self, file: h5py.File, path: str, idx: int) -> np.ndarray | None:
        dataset = file.get(path)
        if not isinstance(dataset, h5py.Dataset) or not dataset.shape:
            return None
        return np.asarray(dataset[idx], dtype=np.float32)

    def _depth_ranges(self, file: h5py.File, idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        cameras = file.get("cameras")
        if not isinstance(cameras, h5py.Group):
            return out
        for cam_id in ordered_camera_ids(list(cameras.keys())):
            cam_group = cameras[cam_id]
            depth_key = find_depth_key(cam_group)
            if depth_key is None:
                out[cam_id] = {"available": False}
                continue
            depth = np.asarray(cam_group[depth_key][idx], dtype=np.float32)
            finite = depth[np.isfinite(depth)]
            if finite.size == 0:
                out[cam_id] = {"available": True, "finite": 0, "invalid": int(depth.size)}
                continue
            p1, p99 = robust_limits(finite)
            out[cam_id] = {
                "available": True,
                "finite": int(finite.size),
                "invalid": int(depth.size - finite.size),
                "min": round(float(np.min(finite)), 5),
                "max": round(float(np.max(finite)), 5),
                "p1": round(p1, 5),
                "p99": round(p99, 5),
            }
        return out

    def _box_counts(self, file: h5py.File, idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        cameras = file.get("cameras")
        if not isinstance(cameras, h5py.Group):
            return out
        box3d_group = file.get("labels/box3d")
        box3d_objects = len(box3d_group.keys()) if isinstance(box3d_group, h5py.Group) else 0
        box2d_group = file.get("labels/box2d")
        for cam_id in ordered_camera_ids(list(cameras.keys())):
            visible_count = 0
            total_count = 0
            if isinstance(box2d_group, h5py.Group) and cam_id in box2d_group:
                cam_group = box2d_group[cam_id]
                for obj_group in cam_group.values():
                    if not isinstance(obj_group, h5py.Group):
                        continue
                    total_count += 1
                    visible = obj_group.get("visible")
                    xyxy = obj_group.get("xyxy")
                    if isinstance(visible, h5py.Dataset):
                        try:
                            if bool(visible[idx]):
                                visible_count += 1
                        except Exception:
                            pass
                    elif isinstance(xyxy, h5py.Dataset):
                        try:
                            box = np.asarray(xyxy[idx], dtype=np.int32)
                            if box.shape[0] == 4 and box[2] > box[0] and box[3] > box[1]:
                                visible_count += 1
                        except Exception:
                            pass
            out[cam_id] = {
                "box2d_visible": visible_count,
                "box2d_objects": total_count,
                "box3d_objects": box3d_objects,
            }
        return out

    def _occupancy_counts(self, file: h5py.File, idx: int) -> dict[str, Any]:
        key, group = self._occupancy_group(file)
        if group is None:
            return {"available": False}
        state_dataset = group.get("state")
        if not isinstance(state_dataset, h5py.Dataset) or not state_dataset.shape:
            return {"available": False}
        occ_idx = max(0, min(int(idx), int(state_dataset.shape[0]) - 1))
        state = np.asarray(state_dataset[occ_idx], dtype=np.uint8)
        unique, counts = np.unique(state, return_counts=True)
        state_counts = {int(k): int(v) for k, v in zip(unique, counts)}

        semantic_counts: list[dict[str, Any]] = []
        semantic_dataset = group.get("semantic_id")
        legend = parse_jsonish(group_scalar(group, "semantic_legend", None), {})
        if not isinstance(legend, dict):
            legend = {}
        occupied_mask = state == OCCUPANCY_OCCUPIED
        if isinstance(semantic_dataset, h5py.Dataset) and np.any(occupied_mask):
            semantic = np.asarray(semantic_dataset[occ_idx], dtype=np.uint16)
            semantic_unique, semantic_count_values = np.unique(semantic[occupied_mask], return_counts=True)
            order = np.argsort(-semantic_count_values)
            for item_idx in order[:12]:
                semantic_id = int(semantic_unique[item_idx])
                semantic_counts.append(
                    {
                        "id": semantic_id,
                        "label": str(legend.get(str(semantic_id), legend.get(semantic_id, f"id_{semantic_id}"))),
                        "count": int(semantic_count_values[item_idx]),
                    }
                )

        return {
            "available": True,
            "label_key": key,
            "idx": occ_idx,
            "frame_valid": dataset_bool(file, f"labels/{key}/frame_valid", occ_idx),
            "unknown": state_counts.get(OCCUPANCY_UNKNOWN, 0),
            "free": state_counts.get(OCCUPANCY_FREE, 0),
            "occupied": state_counts.get(OCCUPANCY_OCCUPIED, 0),
            "total": int(state.size),
            "state_counts": {str(k): v for k, v in state_counts.items()},
            "semantic_counts": semantic_counts,
        }

    def _series_window(self, file: h5py.File, idx: int, frame_count: int) -> dict[str, Any]:
        radius = min(80, max(10, frame_count // 12)) if frame_count else 10
        start = max(0, idx - radius)
        end = min(frame_count, idx + radius + 1)
        frames = list(range(start, end))

        def mean_abs(path: str) -> list[float | None]:
            dataset = file.get(path)
            if not isinstance(dataset, h5py.Dataset) or end <= start:
                return []
            data = np.asarray(dataset[start:end], dtype=np.float32)
            with np.errstate(invalid="ignore"):
                abs_data = np.abs(data)
                valid = np.isfinite(abs_data)
                counts = np.sum(valid, axis=1)
                sums = np.sum(np.where(valid, abs_data, 0.0), axis=1)
                values = np.full((data.shape[0],), np.nan, dtype=np.float32)
                np.divide(sums, counts, out=values, where=counts > 0)
            return [round(float(v), 5) if math.isfinite(float(v)) else None for v in values]

        return {
            "start": start,
            "end": end,
            "frames": frames,
            "qpos_mean_abs": mean_abs("robot/qpos"),
            "action_mean_abs": mean_abs("action/commanded"),
        }

    def contact_summary(self) -> dict[str, Any]:
        with self._lock:
            if self._contact_cache is not None:
                return self._contact_cache

        summary = self.summary()
        frame_count = int(summary.get("frame_count", 0))
        sites = summary.get("tactile_sites", [])
        result: dict[str, Any] = {
            "frame_count": frame_count,
            "sites": {},
            "total": {"max_values": [0] * frame_count, "nonzero_frames": [], "ranges": []},
        }
        if not sites or frame_count <= 0:
            with self._lock:
                self._contact_cache = result
            return result

        total_max = np.zeros(frame_count, dtype=np.uint8)
        with self._open() as file:
            tacmap_group = file.get("robot/tactile/tacmap")
            if not isinstance(tacmap_group, h5py.Group):
                return result
            for site in sites:
                dataset = tacmap_group.get(site)
                if not isinstance(dataset, h5py.Dataset):
                    continue
                max_values = np.zeros(frame_count, dtype=np.uint8)
                mean_values = np.zeros(frame_count, dtype=np.float32)
                for start in range(0, frame_count, 64):
                    end = min(frame_count, start + 64)
                    block = np.asarray(dataset[start:end], dtype=np.uint8)
                    flat = block.reshape(block.shape[0], -1)
                    max_values[start:end] = np.max(flat, axis=1)
                    mean_values[start:end] = np.mean(flat, axis=1)
                total_max = np.maximum(total_max, max_values)
                nonzero = np.flatnonzero(max_values > 0).astype(int).tolist()
                result["sites"][site] = {
                    "max_values": max_values.astype(int).tolist(),
                    "mean_values": [round(float(v), 5) for v in mean_values],
                    "nonzero_frames": nonzero,
                    "ranges": frame_ranges(nonzero),
                    "max": int(np.max(max_values)) if max_values.size else 0,
                }
        total_nonzero = np.flatnonzero(total_max > 0).astype(int).tolist()
        result["total"] = {
            "max_values": total_max.astype(int).tolist(),
            "nonzero_frames": total_nonzero,
            "ranges": frame_ranges(total_nonzero),
            "max": int(np.max(total_max)) if total_max.size else 0,
        }
        with self._lock:
            self._contact_cache = result
        return result

    def _camera_image_shape_from_group(self, cam_group: h5py.Group) -> tuple[int, int]:
        depth_key = find_depth_key(cam_group)
        if depth_key is not None and len(cam_group[depth_key].shape) >= 3:
            return int(cam_group[depth_key].shape[1]), int(cam_group[depth_key].shape[2])
        rgb = cam_group.get("rgb")
        if isinstance(rgb, h5py.Dataset) and len(rgb.shape) >= 4:
            return int(rgb.shape[1]), int(rgb.shape[2])
        return 480, 640

    def _occupied_voxel_points(
        self,
        group: h5py.Group,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray | None, str | None]:
        state_dataset = group.get("state")
        if not isinstance(state_dataset, h5py.Dataset) or not state_dataset.shape:
            return np.empty((0, 3), dtype=np.float32), None, "Occupancy state missing"
        occ_idx = max(0, min(int(idx), int(state_dataset.shape[0]) - 1))
        state = np.asarray(state_dataset[occ_idx], dtype=np.uint8)
        if state.ndim != 3:
            return np.empty((0, 3), dtype=np.float32), None, f"Bad occupancy shape {state.shape}"

        bounds_dataset = group.get("bounds")
        if not isinstance(bounds_dataset, h5py.Dataset):
            return np.empty((0, 3), dtype=np.float32), None, "Occupancy bounds missing"
        bounds = np.asarray(bounds_dataset[()], dtype=np.float32)
        if bounds.shape != (2, 3):
            return np.empty((0, 3), dtype=np.float32), None, f"Bad bounds shape {bounds.shape}"
        voxel_size = finite_float(group_scalar(group, "voxel_size", None))
        if voxel_size is None or voxel_size <= 0:
            return np.empty((0, 3), dtype=np.float32), None, "Occupancy voxel_size missing"

        mask = state == OCCUPANCY_OCCUPIED
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32), None, None

        ix, iy, iz = np.where(mask)
        semantics: np.ndarray | None = None
        semantic_dataset = group.get("semantic_id")
        if isinstance(semantic_dataset, h5py.Dataset) and semantic_dataset.shape[:1] == state_dataset.shape[:1]:
            semantic = np.asarray(semantic_dataset[occ_idx], dtype=np.uint16)
            if semantic.shape == state.shape:
                semantics = semantic[mask]

        if ix.size > MAX_OCCUPANCY_POINTS_PER_VIEW:
            keep = np.linspace(0, ix.size - 1, MAX_OCCUPANCY_POINTS_PER_VIEW, dtype=np.int64)
            ix = ix[keep]
            iy = iy[keep]
            iz = iz[keep]
            if semantics is not None:
                semantics = semantics[keep]

        points = np.stack(
            [
                bounds[0, 0] + (ix.astype(np.float32) + 0.5) * float(voxel_size),
                bounds[0, 1] + (iy.astype(np.float32) + 0.5) * float(voxel_size),
                bounds[0, 2] + (iz.astype(np.float32) + 0.5) * float(voxel_size),
            ],
            axis=1,
        ).astype(np.float32)
        return points, semantics, None

    def rgb_image(self, cam_id: str, idx: int) -> tuple[bytes, str]:
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            image = self._read_rgb_pil(file, cam_id, idx)
            if image is None:
                return placeholder_png("RGB missing", f"{cam_id} frame {idx}"), "image/png"
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            return buffer.getvalue(), "image/jpeg"

    def depth_image(self, cam_id: str, idx: int) -> tuple[bytes, str]:
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            cameras = file.get("cameras")
            if not isinstance(cameras, h5py.Group) or cam_id not in cameras:
                return placeholder_png("Camera missing", cam_id), "image/png"
            cam_group = cameras[cam_id]
            depth_key = find_depth_key(cam_group)
            if depth_key is None:
                return placeholder_png("Depth missing", cam_id), "image/png"
            depth = np.asarray(cam_group[depth_key][idx], dtype=np.float32)
            valid = np.isfinite(depth)
            if not np.any(valid):
                return placeholder_png("Depth has no finite values", cam_id), "image/png"
            lo, hi = robust_limits(depth[valid])
            rgb = colormap_array(depth, cmap_name="viridis", vmin=lo, vmax=hi, valid_mask=valid)
            return image_bytes(Image.fromarray(rgb, mode="RGB")), "image/png"

    def occupancy_image(self, cam_id: str, idx: int) -> tuple[bytes, str]:
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            cameras = file.get("cameras")
            if not isinstance(cameras, h5py.Group) or cam_id not in cameras:
                return placeholder_png("Camera missing", cam_id), "image/png"
            cam_group = cameras[cam_id]

            image = self._read_rgb_pil(file, cam_id, idx)
            if image is None:
                height, width = self._camera_image_shape_from_group(cam_group)
                image = Image.new("RGB", (width, height), (18, 20, 21))

            camera_frame = self._camera_frame(file, cam_id, idx, (image.height, image.width))
            if camera_frame is None:
                return placeholder_png("Camera geometry missing", cam_id, image.size), "image/png"

            key, group = self._occupancy_group(file)
            if group is None:
                return placeholder_png("Occupancy missing", "labels/occupancy_gt or labels/occupancy_tsdf"), "image/png"
            points_world, semantics, error = self._occupied_voxel_points(group, idx)
            if error:
                return placeholder_png("Occupancy render error", error, image.size), "image/png"

            image = self._draw_occupancy_overlay(image, camera_frame, points_world, semantics)
            draw = ImageDraw.Draw(image)
            label = f"{key}: {points_world.shape[0]} occupied voxels"
            bbox = draw.textbbox((8, 8), label)
            draw.rectangle((6, 6, bbox[2] + 12, bbox[3] + 10), fill=(20, 20, 20))
            draw.text((8, 8), label, fill=(240, 244, 242))
            return image_bytes(image), "image/png"

    def tactile_image(self, site: str, idx: int) -> tuple[bytes, str]:
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            dataset = file.get(f"robot/tactile/tacmap/{site}")
            if not isinstance(dataset, h5py.Dataset):
                return placeholder_png("TacMap missing", site, (240, 240)), "image/png"
            values = np.asarray(dataset[idx], dtype=np.uint8)
            rgb = colormap_array(
                values,
                cmap_name="inferno",
                vmin=0.0,
                vmax=255.0,
                valid_mask=np.ones(values.shape, dtype=np.bool_),
                zero_is_black=True,
            )
            return image_bytes(Image.fromarray(rgb, mode="RGB")), "image/png"

    def box_image(self, cam_id: str, idx: int, mode: str = "both") -> tuple[bytes, str]:
        with self._open() as file:
            idx = self._clamp_idx(file, idx)
            image = self._read_rgb_pil(file, cam_id, idx)
            if image is None:
                return placeholder_png("RGB missing for box overlay", f"{cam_id} frame {idx}"), "image/png"
            draw = ImageDraw.Draw(image)

            mode = str(mode or "both").lower()
            draw_box2d = mode in ("box2d", "2d", "both", "box_both", "all")
            draw_box3d = mode in ("box3d", "3d", "both", "box_both", "all")

            if draw_box3d:
                camera_frame = self._camera_frame(file, cam_id, idx, (image.height, image.width))
                box3d_group = file.get("labels/box3d")
                if camera_frame is not None and isinstance(box3d_group, h5py.Group):
                    self._draw_box3d_overlay(
                        draw,
                        camera_frame,
                        box3d_group,
                        idx,
                        draw_labels=not draw_box2d,
                    )

            if draw_box2d:
                box2d_group = file.get(f"labels/box2d/{cam_id}")
                if isinstance(box2d_group, h5py.Group):
                    self._draw_box2d_overlay(draw, box2d_group, idx)

            return image_bytes(image), "image/png"

    def _read_rgb_pil(self, file: h5py.File, cam_id: str, idx: int) -> Image.Image | None:
        dataset = file.get(f"cameras/{cam_id}/rgb")
        if not isinstance(dataset, h5py.Dataset):
            return None
        # Inference recordings down-sample camera frames (InferenceRecorder
        # stores RGB every DEX2BENCH_RECORD_STRIDE policy steps), so the RGB
        # dataset can be shorter than meta/frame_count.  Map the requested
        # qpos-frame index onto the nearest recorded RGB frame.
        rgb_len = int(dataset.shape[0]) if dataset.shape else 0
        frame_count = self._frame_count(file)
        if rgb_len > 0 and rgb_len < frame_count:
            idx = int(idx * rgb_len / frame_count)
            idx = max(0, min(idx, rgb_len - 1))
        value = dataset[idx]
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
        if isinstance(value, np.ndarray):
            data = value.astype(np.uint8, copy=False).tobytes()
        elif isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        elif isinstance(value, np.void):
            data = bytes(value)
        else:
            data = bytes(value)
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None

    def _draw_occupancy_overlay(
        self,
        image: Image.Image,
        camera_frame: CameraFrame,
        points_world: np.ndarray,
        semantics: np.ndarray | None,
    ) -> Image.Image:
        if points_world.size == 0:
            return image

        points_cam, valid = world_points_to_camera(points_world, camera_frame.extrinsic_world_from_cam)
        if (
            camera_frame.camera_model == "fisheye"
            and camera_frame.fisheye_camera_matrix is not None
            and camera_frame.distortion_coefficients is not None
        ):
            uv = project_camera_points_fisheye(
                points_cam,
                camera_frame.fisheye_camera_matrix,
                camera_frame.distortion_coefficients,
            )
        else:
            uv = project_camera_points(points_cam, camera_frame.intrinsic)

        height, width = camera_frame.image_shape
        in_bounds = (
            valid
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        if not np.any(in_bounds):
            return image

        uv = uv[in_bounds]
        depths = points_cam[in_bounds, 2]
        if semantics is not None:
            semantics = semantics[in_bounds]
        order = np.argsort(-depths)
        uv = uv[order]
        depths = depths[order]
        if semantics is not None:
            semantics = semantics[order]

        if semantics is None:
            lo, hi = robust_limits(depths)
            colors = colormap_array(depths.reshape(-1, 1), cmap_name="plasma", vmin=lo, vmax=hi).reshape(-1, 3)
        else:
            colors = np.asarray([occupancy_semantic_color(int(v)) for v in semantics], dtype=np.uint8)

        radius = max(1, min(width, height) // 360)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for (u, v), color in zip(uv, colors):
            x = int(round(float(u)))
            y = int(round(float(v)))
            r, g, b = [int(c) for c in color]
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(r, g, b, 170))

        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _camera_frame(
        self,
        file: h5py.File,
        cam_id: str,
        idx: int,
        image_shape: tuple[int, int],
    ) -> CameraFrame | None:
        cam_group = file.get(f"cameras/{cam_id}")
        if not isinstance(cam_group, h5py.Group):
            return None
        if "intrinsic" not in cam_group or "extrinsic_world_from_cam" not in cam_group:
            return None
        intrinsic = np.asarray(cam_group["intrinsic"][()], dtype=np.float32)
        extrinsic = np.asarray(cam_group["extrinsic_world_from_cam"][idx], dtype=np.float32)
        definitions = self._camera_definitions(file)
        definition = definitions.get(cam_id, {})
        extrinsic = fix_extrinsic_if_needed(extrinsic, definition)

        camera_model = dataset_scalar(file, f"cameras/{cam_id}/camera_model", default=None)
        if camera_model is None:
            camera_model = definition.get("camera_type", "pinhole")
        camera_model = str(camera_model)

        fisheye_matrix = None
        distortion = None
        if camera_model in ("fisheye", "opencv_fisheye", "isaacsim_fisheye"):
            if "fisheye_camera_matrix" in cam_group:
                fisheye_matrix = np.asarray(cam_group["fisheye_camera_matrix"][()], dtype=np.float32)
            elif definition.get("fisheye_camera_matrix") is not None:
                fisheye_matrix = np.asarray(definition["fisheye_camera_matrix"], dtype=np.float32)
            if "distortion_coefficients" in cam_group:
                distortion = np.asarray(cam_group["distortion_coefficients"][()], dtype=np.float32)
            elif definition.get("fisheye_distortion_coefficients") is not None:
                distortion = np.asarray(definition["fisheye_distortion_coefficients"], dtype=np.float32)
            if fisheye_matrix is not None and distortion is not None:
                camera_model = "fisheye"
            else:
                camera_model = "pinhole"

        return CameraFrame(
            intrinsic=intrinsic,
            extrinsic_world_from_cam=extrinsic,
            image_shape=image_shape,
            camera_model=camera_model,
            fisheye_camera_matrix=fisheye_matrix,
            distortion_coefficients=distortion,
        )

    def _draw_box3d_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        camera_frame: CameraFrame,
        box3d_group: h5py.Group,
        idx: int,
        *,
        draw_labels: bool,
    ) -> None:
        for object_idx, object_id in enumerate(sorted(box3d_group.keys())):
            group = box3d_group[object_id]
            if not isinstance(group, h5py.Group):
                continue
            if not all(key in group for key in ("center_world", "size_lwh", "quat_world")):
                continue
            center = np.asarray(group["center_world"][idx], dtype=np.float32)
            size = np.asarray(group["size_lwh"][idx], dtype=np.float32)
            quat = np.asarray(group["quat_world"][idx], dtype=np.float32)
            if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0)):
                continue
            color = OBJECT_COLORS[object_idx % len(OBJECT_COLORS)]
            corners = compute_box_corners(center, size, quat)
            self._draw_projected_box(draw, camera_frame, corners, color)
            if draw_labels:
                self._draw_projected_label(draw, camera_frame, center, object_id, color)

    def _draw_box2d_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        box2d_camera_group: h5py.Group,
        idx: int,
    ) -> None:
        for object_idx, object_id in enumerate(sorted(box2d_camera_group.keys())):
            group = box2d_camera_group[object_id]
            if not isinstance(group, h5py.Group) or "xyxy" not in group:
                continue
            visible = True
            if "visible" in group:
                try:
                    visible = bool(group["visible"][idx])
                except Exception:
                    visible = False
            xyxy = np.asarray(group["xyxy"][idx], dtype=np.int32)
            if xyxy.shape[0] != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            if not visible or x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0:
                continue
            color = OBJECT_COLORS[object_idx % len(OBJECT_COLORS)]
            self._draw_2d_rect(draw, (x1, y1, x2, y2), color)
            self._draw_2d_label(draw, (x1, y1), object_id, color)

    def _draw_2d_rect(
        self,
        draw: ImageDraw.ImageDraw,
        xyxy: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = xyxy
        for offset in range(3):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

    def _draw_2d_label(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[int, int],
        object_id: str,
        color: tuple[int, int, int],
    ) -> None:
        x, y = xy
        label = compact_object_label(object_id)
        bbox = draw.textbbox((x, y), label)
        pad = 3
        label_top = max(0, y - (bbox[3] - bbox[1]) - 2 * pad - 2)
        draw.rectangle(
            (x, label_top, x + (bbox[2] - bbox[0]) + 2 * pad, label_top + (bbox[3] - bbox[1]) + 2 * pad),
            fill=(20, 20, 20),
        )
        draw.text((x + pad, label_top + pad), label, fill=color)

    def _draw_projected_box(
        self,
        draw: ImageDraw.ImageDraw,
        camera_frame: CameraFrame,
        corners_world: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        height, width = camera_frame.image_shape
        for i, j in BOX_EDGES:
            n_subdiv = 20 if camera_frame.camera_model == "fisheye" else 1
            t = np.linspace(0.0, 1.0, n_subdiv + 1, dtype=np.float32)
            points = corners_world[i][None, :] * (1.0 - t[:, None]) + corners_world[j][None, :] * t[:, None]
            uv, valid = project_world_points_to_image(points, camera_frame)
            in_bounds = (
                valid
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= -width * 0.5)
                & (uv[:, 0] <= width * 1.5)
                & (uv[:, 1] >= -height * 0.5)
                & (uv[:, 1] <= height * 1.5)
            )
            for a in range(len(points) - 1):
                if in_bounds[a] and in_bounds[a + 1]:
                    draw.line(
                        [
                            (float(uv[a, 0]), float(uv[a, 1])),
                            (float(uv[a + 1, 0]), float(uv[a + 1, 1])),
                        ],
                        fill=color,
                        width=2,
                    )

    def _draw_projected_label(
        self,
        draw: ImageDraw.ImageDraw,
        camera_frame: CameraFrame,
        center_world: np.ndarray,
        object_id: str,
        color: tuple[int, int, int],
    ) -> None:
        height, width = camera_frame.image_shape
        uv, valid = project_world_points_to_image(center_world.reshape(1, 3), camera_frame)
        if not bool(valid[0]):
            return
        x, y = float(uv[0, 0]), float(uv[0, 1])
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        label = compact_object_label(object_id)
        bbox = draw.textbbox((x, y), label)
        pad = 3
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(20, 20, 20),
        )
        draw.text((x, y), label, fill=color)


def frame_ranges(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    ranges: list[list[int]] = []
    start = prev = int(frames[0])
    for frame in frames[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append([start, prev])
        start = prev = frame
    ranges.append([start, prev])
    return ranges


def compute_box_corners(center: np.ndarray, size_lwh: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    corners_local = CORNER_SIGNS * (0.5 * np.asarray(size_lwh, dtype=np.float32))
    rot = quat_xyzw_to_rot(np.asarray(quat_xyzw, dtype=np.float32))
    return (corners_local @ rot.T) + np.asarray(center, dtype=np.float32)


def compact_object_label(object_id: str) -> str:
    label = str(object_id)
    if label.startswith("obj_"):
        label = label[4:]
    return label


def fix_extrinsic_if_needed(extrinsic: np.ndarray, definition: dict[str, Any]) -> np.ndarray:
    if definition.get("mount_type") != "world":
        return extrinsic
    position = definition.get("position")
    target = definition.get("target")
    if position is None or target is None:
        return extrinsic
    expected = np.asarray(target, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    norm = float(np.linalg.norm(expected))
    if norm <= 1.0e-9:
        return extrinsic
    expected = expected / norm
    actual = np.asarray(extrinsic[:3, 0], dtype=np.float32)
    if float(np.dot(actual, expected)) >= -0.5:
        return extrinsic
    fixed = extrinsic.copy()
    fixed[:3, 0] = -extrinsic[:3, 0]
    fixed[:3, 2] = -extrinsic[:3, 2]
    return fixed
