"""Configuration utilities for dex2scene data collection."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import yaml


CAMERA_DEPENDENT_MODALITIES = {"rgb", "depth", "box2d"}
WRIST_CAMERA_IDS = ("cam_wrist_right", "cam_wrist_left")
CAMERA_MAX_WIDTH = 1920
CAMERA_MAX_HEIGHT = 1080

DEFAULT_OCCUPANCY_CONFIG: Dict[str, Any] = {
    "voxel_size": 0.02,
    "bounds": [[-0.8, -0.8, 0.0], [0.8, 0.8, 1.2]],
    "occupied_margin": None,
    "free_margin": None,
    "chunk_size": 262144,
    "max_voxel_count": 2000000,
    "require_all_cameras": True,
    "allow_depth_fallback": False,
    "label_version": "occupancy_tsdf_v1",
    "fusion_mode": "binary",
    "truncation_distance": None,
    "surface_threshold": None,
}


DEFAULT_COLLECT_CONFIG: Dict[str, Any] = {
    "profile": {
        "dataset_version": "bench_raw_v1",
        "sensor_profile_id": "benchmark_raw",
        "sensor_profile_version": "1.0",
    },
    "dataset": {
        "format": "hdf5_per_episode",
        "root": "./outputs",
        "name": "dex2scene_dataset",
    },
    "capture": {
        "fps": 100,
        "step_stride": 1,
        "max_buffer_mb": 2048,
    },
    "modalities": {
        "rgb": True,
        "depth": True,
        "object_pose": True,
        "joint_state": True,
        "box3d": True,
        "box2d": True,
        "tactile": False,
    },
    "camera_list": [
        {
            "id": "cam_chest",
            "mount_type": "world",
            "position": [0.0, -0.40, 1.50],
            "target": [0.0, 0.00, 0.82],
            "width": 640,
            "height": 480,
            "focal_length": 18.0,
            "horizontal_aperture": 20.955,
            "clipping_range": [0.01, 20.0],
        },
        {
            "id": "cam_overhead",
            "mount_type": "world",
            "position": [0.0, 0.0, 1.80],
            "target": [0.0, 0.0, 0.82],
            "width": 640,
            "height": 640,
            "focal_length": 15.0,
            "horizontal_aperture": 20.955,
            "clipping_range": [0.01, 20.0],
        },
        {
            "id": "cam_wrist_right",
            "mount_type": "robot_link",
            "camera_type": "fisheye",

            "fisheye_camera_matrix": [[169.7056, 0.0, 320.0], [0.0, 169.7056, 240.0], [0.0, 0.0, 1.0]],
            "fisheye_distortion_coefficients": [-0.0416667, 0.0005208, -0.0000031, 0.0000000108],
            "fisheye_max_fov_deg": 180.0,
            "parent_link": "wrist_3_link",
            "offset_xyz": [0.052620712, 0.057382115, 0.075768000],
            "offset_rpy": [-0.262750000, 0.084286000, 1.547360000],
            "width": 640,
            "height": 480,
            "clipping_range": [0.01, 20.0],
        },
        {
            "id": "cam_wrist_left",
            "mount_type": "robot_link",
            "camera_type": "fisheye",

            "fisheye_camera_matrix": [[169.7056, 0.0, 320.0], [0.0, 169.7056, 240.0], [0.0, 0.0, 1.0]],
            "fisheye_distortion_coefficients": [-0.0416667, 0.0005208, -0.0000031, 0.0000000108],
            "fisheye_max_fov_deg": 180.0,
            "parent_link": "L_arm_wrist_3_link",
            "offset_xyz": [-0.052529389, 0.057465743, 0.075768174],
            "offset_rpy": [0.261799551, 0.000000071, 1.569999956],
            "width": 640,
            "height": 480,
            "clipping_range": [0.01, 20.0],
        },
    ],
    "robot_wrist_camera_profiles": {
        "multi_panda_with_orca": {
            "cam_wrist_right": {
                "parent_link": "multi_panda_link7",
                "offset_xyz": [-0.06, 0.05, 0.27],
                "offset_rpy": [0.1571, -0.1571, 0.785],
            },
            "cam_wrist_left": {
                "parent_link": "panda_link7",
                "offset_xyz": [-0.06, 0.05, 0.27],
                "offset_rpy": [0.1571, 0.1571, 0.785],
            },
        },
        "multi_ur5_rh56dfx_with_flange": {
            "cam_wrist_right": {
                "parent_link": "wrist_3_link",
                "offset_xyz": [0.052620712, 0.057382115, 0.075768000],
                "offset_rpy": [-0.262750000, 0.084286000, 1.547360000],
            },
            "cam_wrist_left": {
                "parent_link": "L_arm_wrist_3_link",
                "offset_xyz": [-0.052529389, 0.057465743, 0.075768174],
                "offset_rpy": [0.261799551, 0.000000071, 1.569999956],
            },
        },
    },
    "scene_background": {
        "enabled": False,
        "uri": "../../../Dex2Assets/Background/Indoor/university_workshop_4k.hdr",
        "yaw_deg": 0.0,
        "physics_enabled": False,
        "fill_light_intensity": 800.0,
        "fill_light_jitter_ratio": 0.0,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_utf8(path: str) -> Dict[str, Any] | None:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_yaw_deg(value: Any) -> float:
    return float(value or 0.0) % 360.0


def _normalize_scene_background_config(
    config: Dict[str, Any] | None,
    *,
    config_path: str | None = None,
) -> Dict[str, Any]:
    raw = dict(config or {})
    enabled = bool(raw.get("enabled", False))
    uri = str(raw.get("uri", "") or "")
    bg_type = str(raw.get("type", "") or "").strip().lower()
    if not bg_type:
        ext = os.path.splitext(uri)[1].lower() if uri else ""
        bg_type = "usd_scene" if ext in {".usda", ".usd", ".usdc"} else "hdr"
    if uri and config_path and not os.path.isabs(uri) and "://" not in uri:
        uri = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), uri))
    result = {
        "enabled": enabled,
        "type": bg_type,
        "uri": uri.replace("\\", "/") if uri else "",
        "yaw_deg": _normalize_yaw_deg(raw.get("yaw_deg", 0.0)),
        "physics_enabled": bool(raw.get("physics_enabled", False)),
        "fill_light_intensity": float(raw.get("fill_light_intensity", 800.0)),
        "fill_light_jitter_ratio": float(raw.get("fill_light_jitter_ratio", 0.0)),
    }
    if "scene_offset" in raw:
        result["scene_offset"] = [float(v) for v in raw["scene_offset"]]
    return result


def load_collect_scene_background(config_path: str | None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_COLLECT_CONFIG.get("scene_background", {}))
    if config_path:
        cfg_dict = _load_yaml_utf8(os.path.abspath(config_path))
        if not isinstance(cfg_dict, dict):
            raise ValueError(f"Collect config must be a dict: {config_path}")
        raw_background = cfg_dict.get("scene_background", {})
        if raw_background is None:
            raw_background = {}
        if not isinstance(raw_background, dict):
            raise ValueError("scene_background must be a mapping if provided.")
        _deep_update(merged, raw_background)
    return _normalize_scene_background_config(merged, config_path=config_path)


def _apply_robot_wrist_camera_profile(config: Dict[str, Any], robot_key: str | None) -> Dict[str, Any]:
    camera_list = config.get("camera_list", [])
    if not isinstance(camera_list, list):
        raise ValueError("camera_list must be a list.")

    wrist_camera_present = False
    camera_by_id: Dict[str, Dict[str, Any]] = {}
    for item in camera_list:
        if not isinstance(item, dict):
            continue
        cam_id = str(item.get("id", ""))
        if not cam_id:
            continue
        camera_by_id[cam_id] = item
        if cam_id in WRIST_CAMERA_IDS:
            wrist_camera_present = True

    if not wrist_camera_present or not robot_key:
        return config

    profiles = config.get("robot_wrist_camera_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("robot_wrist_camera_profiles must be a mapping if provided.")

    profile = profiles.get(robot_key)
    if profile is None:
        raise ValueError(f"No robot_wrist_camera_profiles entry for robot key '{robot_key}'.")
    if not isinstance(profile, dict):
        raise ValueError(f"robot_wrist_camera_profiles['{robot_key}'] must be a mapping.")

    for cam_id in WRIST_CAMERA_IDS:
        override = profile.get(cam_id)
        if override is None:
            continue
        if not isinstance(override, dict):
            raise ValueError(f"robot_wrist_camera_profiles['{robot_key}']['{cam_id}'] must be a mapping.")
        base = camera_by_id.get(cam_id)
        if base is None:
            base = {"id": cam_id, "mount_type": "robot_link"}
            camera_list.append(base)
            camera_by_id[cam_id] = base
        _deep_update(base, override)

    return config


def normalize_occupancy_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_OCCUPANCY_CONFIG)
    if config:
        merged.update(config)
    merged["voxel_size"] = float(merged.get("voxel_size", DEFAULT_OCCUPANCY_CONFIG["voxel_size"]))
    merged["bounds"] = merged.get("bounds", deepcopy(DEFAULT_OCCUPANCY_CONFIG["bounds"]))

    occupied_margin_raw = None if config is None else config.get("occupied_margin")
    if occupied_margin_raw is None:
        occupied_margin = float(merged["voxel_size"])
    else:
        occupied_margin = float(occupied_margin_raw)

    free_margin_raw = None if config is None else config.get("free_margin")
    if free_margin_raw is None:
        free_margin = occupied_margin
    else:
        free_margin = float(free_margin_raw)

    merged["occupied_margin"] = occupied_margin
    merged["free_margin"] = free_margin
    merged["chunk_size"] = max(1, int(merged.get("chunk_size", DEFAULT_OCCUPANCY_CONFIG["chunk_size"])))
    merged["max_voxel_count"] = max(0, int(merged.get("max_voxel_count", DEFAULT_OCCUPANCY_CONFIG["max_voxel_count"])))
    merged["require_all_cameras"] = bool(
        merged.get("require_all_cameras", DEFAULT_OCCUPANCY_CONFIG["require_all_cameras"])
    )
    merged["allow_depth_fallback"] = bool(
        merged.get("allow_depth_fallback", DEFAULT_OCCUPANCY_CONFIG["allow_depth_fallback"])
    )
    merged["label_version"] = str(merged.get("label_version", DEFAULT_OCCUPANCY_CONFIG["label_version"]))

    merged["fusion_mode"] = str(merged.get("fusion_mode", "binary"))

    trunc_raw = None if config is None else config.get("truncation_distance")
    if trunc_raw is None:
        merged["truncation_distance"] = 4.0 * merged["voxel_size"]
    else:
        merged["truncation_distance"] = float(trunc_raw)

    surf_raw = None if config is None else config.get("surface_threshold")
    if surf_raw is None:
        merged["surface_threshold"] = 1.0 * merged["voxel_size"]
    else:
        merged["surface_threshold"] = float(surf_raw)

    return merged


DEFAULT_OCCUPANCY_GT_CONFIG: Dict[str, Any] = {
    "voxel_size": 0.01,
    "bounds": [[-1.15, -0.6, 0.74], [1.15, 0.6, 1.10]],
    "geometry_source": "collision",
    "label_version": "occupancy_gt_v1",
}


def normalize_occupancy_gt_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_OCCUPANCY_GT_CONFIG)
    if config:
        merged.update(config)
    merged["voxel_size"] = float(merged.get("voxel_size", 0.01))
    merged["bounds"] = merged.get("bounds", deepcopy(DEFAULT_OCCUPANCY_GT_CONFIG["bounds"]))
    merged["geometry_source"] = str(merged.get("geometry_source", "collision"))
    merged["label_version"] = str(merged.get("label_version", "occupancy_gt_v1"))
    return merged


@dataclass
class CameraCollectConfig:
    camera_id: str
    mount_type: str = "world"  # world | robot_link
    camera_type: str = "pinhole"  # pinhole | fisheye
    parent_link: str | None = None
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: tuple[float, float, float] = (1.5, -1.0, 1.2)
    target: tuple[float, float, float] | None = None
    quat_wxyz: tuple[float, float, float, float] | None = None
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    clipping_range: tuple[float, float] = (0.01, 20.0)

    # ---- fisheye calibration parameters ----
    fisheye_camera_matrix: tuple | None = None  # ((fx,0,cx),(0,fy,cy),(0,0,1))
    fisheye_distortion_coefficients: tuple | None = None  # (k1,k2,k3,k4)
    fisheye_max_fov_deg: float = 180.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CameraCollectConfig":
        cam_id = str(data.get("id", "camera"))
        mount_type = str(data.get("mount_type", "world")).lower()
        if mount_type not in {"world", "robot_link"}:
            raise ValueError(f"Unsupported camera mount_type '{mount_type}' for camera '{cam_id}'.")
        camera_type = str(data.get("camera_type", "pinhole")).lower()
        if camera_type not in {"pinhole", "fisheye"}:
            raise ValueError(f"Unsupported camera_type '{camera_type}' for camera '{cam_id}'.")
        parent_link_raw = data.get("parent_link")
        parent_link = str(parent_link_raw) if parent_link_raw is not None else None
        offset_xyz = tuple(float(v) for v in data.get("offset_xyz", [0.0, 0.0, 0.0]))
        offset_rpy = tuple(float(v) for v in data.get("offset_rpy", [0.0, 0.0, 0.0]))
        position = tuple(float(v) for v in data.get("position", [1.5, -1.0, 1.2]))
        target_raw = data.get("target")
        quat_raw = data.get("quat_wxyz")
        target = tuple(float(v) for v in target_raw) if target_raw is not None else None
        quat_wxyz = tuple(float(v) for v in quat_raw) if quat_raw is not None else None
        if mount_type == "world" and target is None and quat_wxyz is None:
            target = (0.0, 0.0, 0.75)
        if mount_type == "robot_link" and not parent_link:
            raise ValueError(f"camera '{cam_id}' uses robot_link mount but parent_link is not set.")
        # ---- parse fisheye calibration parameters ----
        fisheye_camera_matrix = None
        fisheye_distortion_coefficients = None
        fisheye_max_fov_deg = float(data.get("fisheye_max_fov_deg", 180.0))

        raw_matrix = data.get("fisheye_camera_matrix")
        if raw_matrix is not None:
            fisheye_camera_matrix = tuple(tuple(float(v) for v in row) for row in raw_matrix)

        raw_dist = data.get("fisheye_distortion_coefficients")
        if raw_dist is not None:
            fisheye_distortion_coefficients = tuple(float(v) for v in raw_dist)

        if camera_type == "fisheye":
            if fisheye_camera_matrix is None:
                raise ValueError(
                    f"camera '{cam_id}': camera_type='fisheye' requires fisheye_camera_matrix."
                )
            if fisheye_distortion_coefficients is None:
                raise ValueError(
                    f"camera '{cam_id}': camera_type='fisheye' requires fisheye_distortion_coefficients."
                )

        return CameraCollectConfig(
            camera_id=cam_id,
            mount_type=mount_type,
            camera_type=camera_type,
            parent_link=parent_link,
            offset_xyz=offset_xyz,
            offset_rpy=offset_rpy,
            position=position,
            target=target,
            quat_wxyz=quat_wxyz,
            width=min(int(data.get("width", 640)), CAMERA_MAX_WIDTH),
            height=min(int(data.get("height", 480)), CAMERA_MAX_HEIGHT),
            focal_length=float(data.get("focal_length", 24.0)),
            horizontal_aperture=float(data.get("horizontal_aperture", 20.955)),
            clipping_range=tuple(float(v) for v in data.get("clipping_range", [0.01, 20.0])),
            fisheye_camera_matrix=fisheye_camera_matrix,
            fisheye_distortion_coefficients=fisheye_distortion_coefficients,
            fisheye_max_fov_deg=fisheye_max_fov_deg,
        )


@dataclass
class CollectConfig:
    dataset_format: str = "hdf5_per_episode"
    dataset_root: str = "./outputs"
    dataset_name: str = "dex2scene_dataset"
    dataset_version: str = "bench_raw_v1"
    sensor_profile_id: str = "benchmark_raw"
    sensor_profile_version: str = "1.0"
    capture_fps: int = 100
    step_stride: int = 1
    capture_max_buffer_mb: int = 2048
    modalities: Dict[str, bool] = field(default_factory=dict)
    cameras: List[CameraCollectConfig] = field(default_factory=list)
    scene_background: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_root = os.path.abspath(str(self.dataset_root))
        self.capture_fps = int(self.capture_fps)
        self.step_stride = max(1, int(self.step_stride))
        self.capture_max_buffer_mb = max(0, int(self.capture_max_buffer_mb))
        self.modalities = {str(k): bool(v) for k, v in dict(self.modalities).items()}
        self.scene_background = _normalize_scene_background_config(self.scene_background)
        self._validate_camera_requirements()

    def _validate_camera_requirements(self) -> None:
        needs_cameras = any(self.modalities.get(modality, False) for modality in CAMERA_DEPENDENT_MODALITIES)
        if needs_cameras and not self.cameras:
            raise ValueError("camera_list must not be empty when camera-dependent modalities are enabled.")

    def enabled(self, modality: str) -> bool:
        return bool(self.modalities.get(modality, False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": {
                "dataset_version": self.dataset_version,
                "sensor_profile_id": self.sensor_profile_id,
                "sensor_profile_version": self.sensor_profile_version,
            },
            "dataset": {
                "format": self.dataset_format,
                "root": self.dataset_root,
                "name": self.dataset_name,
            },
            "capture": {
                "fps": self.capture_fps,
                "step_stride": self.step_stride,
                "max_buffer_mb": self.capture_max_buffer_mb,
            },
            "modalities": dict(self.modalities),
            "camera_list": [asdict(cam) for cam in self.cameras],
            "scene_background": dict(self.scene_background),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CollectConfig":
        profile = data.get("profile", {})
        dataset = data.get("dataset", {})
        capture = data.get("capture", {})
        modalities_cfg = data.get("modalities", {})
        if isinstance(modalities_cfg, list):
            modalities = {name: True for name in modalities_cfg}
        else:
            modalities = {str(k): bool(v) for k, v in modalities_cfg.items()}
        camera_pixel = data.get("camera_pixel", {})
        if not isinstance(camera_pixel, dict):
            camera_pixel = {}
        camera_list = data.get("camera_list", [])
        cameras = []
        for item in camera_list:
            cam = CameraCollectConfig.from_dict(item)
            pixel = camera_pixel.get(cam.camera_id)
            if pixel is not None and isinstance(pixel, (list, tuple)) and len(pixel) == 2:
                cam.width = min(int(pixel[0]), CAMERA_MAX_WIDTH)
                cam.height = min(int(pixel[1]), CAMERA_MAX_HEIGHT)
            cameras.append(cam)
        return CollectConfig(
            dataset_format=str(dataset.get("format", "hdf5_per_episode")),
            dataset_root=str(dataset.get("root", "./outputs")),
            dataset_name=str(dataset.get("name", "dex2scene_dataset")),
            dataset_version=str(profile.get("dataset_version", "bench_raw_v1")),
            sensor_profile_id=str(profile.get("sensor_profile_id", "benchmark_raw")),
            sensor_profile_version=str(profile.get("sensor_profile_version", "1.0")),
            capture_fps=int(capture.get("fps", 100)),
            step_stride=int(capture.get("step_stride", 1)),
            capture_max_buffer_mb=int(capture.get("max_buffer_mb", 2048)),
            modalities=modalities,
            cameras=cameras,
            scene_background=_normalize_scene_background_config(data.get("scene_background", {})),
        )


def load_collect_config(
    config_path: str | None,
    output_root_override: str | None = None,
    robot_key: str | None = None,
) -> CollectConfig:
    merged = deepcopy(DEFAULT_COLLECT_CONFIG)
    if config_path:
        cfg_dict = _load_yaml_utf8(os.path.abspath(config_path))
        if not isinstance(cfg_dict, dict):
            raise ValueError(f"Collect config must be a dict: {config_path}")
        _deep_update(merged, cfg_dict)
    if output_root_override:
        merged.setdefault("dataset", {})
        merged["dataset"]["root"] = output_root_override
    if config_path:
        merged["scene_background"] = _normalize_scene_background_config(
            merged.get("scene_background", {}),
            config_path=config_path,
        )
    _apply_robot_wrist_camera_profile(merged, robot_key=robot_key)
    return CollectConfig.from_dict(merged)
