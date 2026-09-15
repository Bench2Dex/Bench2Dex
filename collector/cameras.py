"""Camera creation and frame capture helpers.

Performance-optimised capture pipeline (P0–P3):
  P0 – Caller renders once after pose sync, then capture extracts all data.
  P1 – Cache static metadata (intrinsics, world-camera poses, config fields).
  P2 – CPU-side rotation from quaternion; avoid GPU matrix_from_quat roundtrip.
  P3 – GPU-side slice/cast before single .cpu() transfer; no redundant copies.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .camera_geometry import CameraFrame, quat_wxyz_to_rot
from .config import CameraCollectConfig
from utils.usd_prims import create_prim_compat, delete_prim_compat, delete_prims_and_wait, get_current_stage_compat


@dataclass
class _MountedCameraSpec:
    camera_id: str
    parent_link: str
    body_index: int
    offset_xyz: np.ndarray
    offset_quat_ros_wxyz: np.ndarray


# ---------------------------------------------------------------------------
# P1 helpers – per-camera static metadata that never changes within an episode
# ---------------------------------------------------------------------------

@dataclass
class _CameraStaticMeta:
    """Immutable metadata derived from config + first successful render."""
    intrinsic: np.ndarray              # [3,3] float32
    image_shape: tuple[int, int]
    camera_model: str                  # "pinhole" | "fisheye"
    clipping_range: tuple[float, float] | None
    distortion_coefficients: np.ndarray | None
    fisheye_camera_matrix: np.ndarray | None
    fisheye_valid_mask: np.ndarray | None
    # For world-mounted cameras only (pose never changes):
    extrinsic: np.ndarray | None       # [4,4] float32 – None for mounted cams
    cam_pos_w: np.ndarray | None       # [3] float32 – None for mounted cams
    cam_quat_wxyz: np.ndarray | None   # [4] float32 – None for mounted cams


class CameraRig:
    """Manage multiple IsaacLab camera sensors."""

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        cameras: list[CameraCollectConfig],
        enable_rgb: bool,
        enable_depth: bool,
        robot_articulation: object | None = None,
        camera_generalization_sample: Dict | None = None,
    ):
        self._sim = sim
        self._cfg = cameras
        self._enable_rgb = enable_rgb
        self._enable_depth = enable_depth
        self._robot_articulation = robot_articulation
        self._robot_prim_path = str(self._infer_robot_prim_path(robot_articulation) or "").rstrip("/")
        self._camera_generalization_sample = camera_generalization_sample or {}
        self._cameras: Dict[str, Camera] = {}

        self._cfg_by_id: Dict[str, CameraCollectConfig] = {
            cam_cfg.camera_id: cam_cfg for cam_cfg in cameras
        }
        self._fisheye_valid_mask_cache: Dict[str, np.ndarray] = {}
        self._body_name_to_index: Dict[str, int] = {}
        self._mounted_specs: Dict[str, _MountedCameraSpec] = {}
        self._native_mounted_camera_ids: set[str] = set()
        self._camera_base_prim_paths: Dict[str, str] = {}
        self._camera_sensor_prim_paths: Dict[str, str] = {}
        self._warned_missing_pose_data = False
        self._mounted_setup_errors: list[str] = []
        self._invalid_mounted_camera_ids: set[str] = set()
        self._blocked_camera_ids: set[str] = set()
        self._last_capture_errors: list[str] = []
        self._native_pose_overrides: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._warned_stale_native_camera_ids: set[str] = set()
        self._force_native_pose_camera_ids: set[str] = set()

        # ── P1 caches ────────────────────────────────────────────────
        self._static_meta: Dict[str, _CameraStaticMeta] = {}
        self._caches_warm: bool = False

        self._spawn_cameras()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def camera_prim_paths(self) -> Dict[str, str]:
        return dict(self._camera_sensor_prim_paths)

    @property
    def last_capture_errors(self) -> list[str]:
        return list(self._last_capture_errors)

    def set_robot_articulation(self, robot_articulation: object | None) -> None:
        self._robot_articulation = robot_articulation
        if not self._robot_prim_path:
            self._robot_prim_path = str(self._infer_robot_prim_path(robot_articulation) or "").rstrip("/")
        self._body_name_to_index = {}
        self._mounted_specs.clear()
        self._warned_missing_pose_data = False
        self._mounted_setup_errors = []
        self._invalid_mounted_camera_ids = set()
        self._blocked_camera_ids = set()
        self._native_pose_overrides = {}
        self._force_native_pose_camera_ids = set()

    def set_generalization_sample(self, camera_generalization_sample: Dict | None) -> None:
        self._camera_generalization_sample = camera_generalization_sample or {}

    def sync_mounted_camera_poses(self) -> None:
        """Synchronize robot-link cameras to the latest articulation link poses."""
        self._update_mounted_camera_poses()
        self._repair_stale_native_camera_poses(
            force=True,
            check_current=False,
            warn=False,
        )

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_robot_prim_path(robot_articulation: object | None) -> str | None:
        cfg = getattr(robot_articulation, "cfg", None)
        prim_path = getattr(cfg, "prim_path", None)
        return str(prim_path) if prim_path else None

    def _resolve_parent_link_prim_path(self, stage, parent_link: str) -> str | None:
        """Resolve a PhysX link prim below the configured robot root."""
        if not self._robot_prim_path:
            return None
        root = stage.GetPrimAtPath(self._robot_prim_path)
        if not root.IsValid():
            return None

        from pxr import Usd, UsdPhysics

        matches: list[str] = []
        for prim in Usd.PrimRange(root):
            if prim.GetName() != parent_link:
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                matches.append(str(prim.GetPath()))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple rigid-body prims named '{parent_link}' below {self._robot_prim_path}: {matches}"
            )
        return None

    def _effective_mounted_offset(
        self, cam_cfg: CameraCollectConfig
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        sample = self._camera_sample(cam_cfg.camera_id)
        offset_xyz = np.asarray(cam_cfg.offset_xyz, dtype=np.float32) + np.asarray(
            sample.get("offset_xyz_offset_m", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        rpy_offset_deg = tuple(float(v) for v in sample.get("rpy_offset_deg", [0.0, 0.0, 0.0]))
        offset_rpy = tuple(
            float(cam_cfg.offset_rpy[i]) + math.radians(rpy_offset_deg[i]) for i in range(3)
        )
        return offset_xyz, offset_rpy

    def _plan_camera_prim_paths(self, stage) -> None:
        self._native_mounted_camera_ids.clear()
        self._camera_base_prim_paths.clear()
        self._camera_sensor_prim_paths.clear()
        use_native_robot_link_cameras = os.environ.get(
            "DEX2BENCH_NATIVE_WRIST_CAMERAS", ""
        ).strip().lower() in {"1", "true", "yes"}
        for cam_cfg in self._cfg:
            base_prim = f"/World/Sensors/{cam_cfg.camera_id}"
            if (
                use_native_robot_link_cameras
                and cam_cfg.mount_type == "robot_link"
                and cam_cfg.parent_link
            ):
                parent_prim = self._resolve_parent_link_prim_path(stage, cam_cfg.parent_link)
                if parent_prim is not None:
                    base_prim = f"{parent_prim}/Dex2BenchCameras/{cam_cfg.camera_id}"
                    self._native_mounted_camera_ids.add(cam_cfg.camera_id)
                else:
                    raise RuntimeError(
                        f"Failed to resolve USD prim for camera '{cam_cfg.camera_id}' "
                        f"with parent_link='{cam_cfg.parent_link}' "
                        f"below robot prim path '{self._robot_prim_path or '<unset>'}'. "
                        f"Native USD mount is required — refusing to fall back."
                    )
            self._camera_base_prim_paths[cam_cfg.camera_id] = base_prim
            self._camera_sensor_prim_paths[cam_cfg.camera_id] = f"{base_prim}/Sensor"

    def _spawn_cameras(self) -> None:
        data_types: list[str] = []
        if self._enable_rgb:
            data_types.append("rgb")
        if self._enable_depth:
            data_types.append("distance_to_image_plane")
        if not data_types:
            return
        stage = get_current_stage_compat()
        self._plan_camera_prim_paths(stage)
        stage = self._delete_stale_camera_prims()
        for cam_cfg in self._cfg:
            base_prim = self._camera_base_prim_paths[cam_cfg.camera_id]
            sensor_prim_path = self._camera_sensor_prim_paths[cam_cfg.camera_id]
            create_prim_compat(base_prim, "Xform", stage=stage)
            if cam_cfg.camera_type == "fisheye":
                # Derive focal_length from OpenCV fx for base FisheyeCameraCfg
                K = np.asarray(cam_cfg.fisheye_camera_matrix, dtype=np.float64)
                fx_opencv = float(K[0, 0])
                approx_focal = fx_opencv * cam_cfg.horizontal_aperture / cam_cfg.width
                spawn_cfg = sim_utils.FisheyeCameraCfg(
                    projection_type="fisheyeKannalaBrandtK3",
                    focal_length=approx_focal,
                    horizontal_aperture=cam_cfg.horizontal_aperture,
                    clipping_range=cam_cfg.clipping_range,
                )
            else:
                spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=cam_cfg.focal_length,
                    horizontal_aperture=cam_cfg.horizontal_aperture,
                    clipping_range=cam_cfg.clipping_range,
                )
            offset_cfg = CameraCfg.OffsetCfg()
            if cam_cfg.camera_id in self._native_mounted_camera_ids:
                offset_xyz, offset_rpy = self._effective_mounted_offset(cam_cfg)
                print(f"[CameraRig] {cam_cfg.camera_id} offset: xyz={tuple(float(v) for v in offset_xyz)} rpy={offset_rpy}")
                offset_cfg = CameraCfg.OffsetCfg(
                    pos=tuple(float(v) for v in offset_xyz),
                    rot=tuple(float(v) for v in self._rpy_to_quat_wxyz(offset_rpy)),
                    convention="ros",
                )
            camera_cfg_kwargs = {
                "prim_path": sensor_prim_path,
                "update_period": 0,
                "width": cam_cfg.width,
                "height": cam_cfg.height,
                "data_types": data_types,
                "spawn": spawn_cfg,
                "offset": offset_cfg,
            }
            if "update_latest_camera_pose" in getattr(CameraCfg, "__annotations__", {}):
                camera_cfg_kwargs["update_latest_camera_pose"] = cam_cfg.mount_type == "robot_link"
            camera_cfg = CameraCfg(**camera_cfg_kwargs)
            cam = Camera(cfg=camera_cfg)
            # Isaac Lab Camera.__del__ accesses _rep_registry unconditionally;
            # ensure the attribute exists even for partially-initialized cameras.
            if not hasattr(cam, "_rep_registry"):
                cam._rep_registry = {}
            self._cameras[cam_cfg.camera_id] = cam
            if cam_cfg.camera_id in self._native_mounted_camera_ids:
                mount_kind = "native"
            elif cam_cfg.mount_type == "robot_link":
                mount_kind = "world-sync"
            else:
                mount_kind = "fallback"
            print(f"[CameraRig] {cam_cfg.camera_id}: {mount_kind} mount at {sensor_prim_path}")
            # Apply OpenCV fisheye distortion via USD prim attributes
            if cam_cfg.camera_type == "fisheye" and cam_cfg.fisheye_camera_matrix is not None:
                self._apply_opencv_fisheye_distortion(sensor_prim_path, cam_cfg, stage=stage)

    def _flush_stage_after_camera_delete(self) -> None:
        try:
            self._sim.render()
            return
        except Exception as exc:
            logger.debug("Camera prim deletion render flush failed: %s", exc)
        try:
            import omni.kit.app

            omni.kit.app.get_app().update()
        except Exception as exc:
            logger.debug("Camera prim deletion app update flush failed: %s", exc)

    def _delete_stale_camera_prims(self):
        """Remove stale sensor base prims before creating this rig."""
        stage = get_current_stage_compat()
        prim_paths = set(self._camera_base_prim_paths.values())
        prim_paths.update(f"/World/Sensors/{cam_cfg.camera_id}" for cam_cfg in self._cfg)
        try:
            deleted, remaining = delete_prims_and_wait(
                stage,
                lambda prim_path: delete_prim_compat(prim_path, stage=stage),
                sorted(prim_paths),
                self._flush_stage_after_camera_delete,
                attempts=3,
            )
            if remaining:
                raise RuntimeError(
                    "stale camera prims are still present after deletion: "
                    + ", ".join(remaining)
                )
            if deleted:
                logger.info("Deleted %d stale camera prim(s) before CameraRig spawn", deleted)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete stale camera prims before CameraRig spawn: {exc}") from exc
        return stage

    def _apply_opencv_fisheye_distortion(
        self, prim_path: str, cam_cfg: CameraCollectConfig, stage=None
    ) -> None:
        """Set OpenCV fisheye distortion attributes on a camera USD prim.

        Equivalent to isaacsim Camera.set_opencv_fisheye_properties()
        but works with Isaac Lab Camera (which doesn't expose that method).
        """
        import omni.usd
        if stage is None:
            stage = get_current_stage_compat()

        cam_prim = stage.GetPrimAtPath(prim_path)
        if not cam_prim.IsValid():
            raise RuntimeError(f"Camera prim not found: {prim_path}")

        K = np.asarray(cam_cfg.fisheye_camera_matrix, dtype=np.float64)
        D = np.asarray(cam_cfg.fisheye_distortion_coefficients, dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        k1 = float(D[0]) if len(D) > 0 else 0.0
        k2 = float(D[1]) if len(D) > 1 else 0.0
        k3 = float(D[2]) if len(D) > 2 else 0.0
        k4 = float(D[3]) if len(D) > 3 else 0.0

        # Apply OmniLensDistortionOpenCvFisheyeAPI schema, then set attributes.
        # Attribute names from official Omniverse OmniLensDistortion schemata docs.
        try:
            if not cam_prim.HasAPI("OmniLensDistortionOpenCvFisheyeAPI"):
                cam_prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI")
            prefix = "omni:lensdistortion:opencvFisheye:"
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}fx"), fx)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}fy"), fy)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}cx"), cx)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}cy"), cy)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}k1"), k1)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}k2"), k2)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}k3"), k3)
            omni.usd.set_prop_val(cam_prim.GetAttribute(f"{prefix}k4"), k4)
            omni.usd.set_prop_val(
                cam_prim.GetAttribute(f"{prefix}imageSize"),
                (cam_cfg.width, cam_cfg.height),
            )
        except Exception as exc:
            # Fallback: set ftheta attributes directly (older Isaac Sim
            # without OmniLensDistortionOpenCvFisheyeAPI).
            from pxr import Sdf

            def _set_typed_attr(attr_name: str, val, type_name) -> None:
                attr = cam_prim.GetAttribute(attr_name)
                if (not attr.IsValid()) or str(attr.GetTypeName()) == "":
                    attr = cam_prim.CreateAttribute(attr_name, type_name, custom=True)
                omni.usd.set_prop_val(attr, val)

            logger.warning(
                "OmniLensDistortionOpenCvFisheyeAPI unavailable for %s (%s), "
                "falling back to ftheta attributes with fisheyeKannalaBrandtK3",
                prim_path, exc,
            )
            _set_typed_attr("cameraProjectionType", "fisheyeKannalaBrandtK3", Sdf.ValueTypeNames.String)
            _set_typed_attr("fthetaWidth", float(cam_cfg.width), Sdf.ValueTypeNames.Float)
            _set_typed_attr("fthetaHeight", float(cam_cfg.height), Sdf.ValueTypeNames.Float)
            for attr, val in [
                ("fthetaCx", cx), ("fthetaCy", cy),
                ("fthetaFx", fx), ("fthetaFy", fy),
                ("fthetaK1", k1), ("fthetaK2", k2),
                ("fthetaK3", k3), ("fthetaK4", k4),
            ]:
                _set_typed_attr(attr, val, Sdf.ValueTypeNames.Float)

    # ------------------------------------------------------------------
    # Fisheye valid mask (already cached by design)
    # ------------------------------------------------------------------

    def _get_fisheye_valid_mask(
        self, cam_id: str, cam_cfg: CameraCollectConfig
    ) -> np.ndarray | None:
        if cam_id in self._fisheye_valid_mask_cache:
            return self._fisheye_valid_mask_cache[cam_id]
        if cam_cfg.fisheye_camera_matrix is None or cam_cfg.fisheye_distortion_coefficients is None:
            return None
        from .camera_geometry import compute_fisheye_valid_mask
        mask = compute_fisheye_valid_mask(
            height=cam_cfg.height,
            width=cam_cfg.width,
            camera_matrix=np.asarray(cam_cfg.fisheye_camera_matrix, dtype=np.float32),
            distortion_coefficients=np.asarray(cam_cfg.fisheye_distortion_coefficients, dtype=np.float32),
            max_fov_deg=cam_cfg.fisheye_max_fov_deg,
        )
        self._fisheye_valid_mask_cache[cam_id] = mask
        return mask

    # ------------------------------------------------------------------
    # Quaternion / rotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rpy_to_quat_wxyz(rpy_rad: tuple[float, float, float]) -> np.ndarray:
        r, p, y = rpy_rad
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        quat = np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        return quat / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1.astype(np.float64)
        w2, x2, y2, z2 = q2.astype(np.float64)
        quat = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        return quat / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _rpy_deg_to_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
        r, p, y = (math.radians(v) for v in rpy_deg)
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a wxyz quaternion (numerically stable)."""
        m = np.asarray(rot, dtype=np.float64)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float32)
        n = float(np.linalg.norm(q))
        return q / n if n > 1.0e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def _look_at_quat_wxyz(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        disable_blending: bool = False,
    ) -> np.ndarray:
        """Deterministic look-at -> world-frame camera quaternion (wxyz).

        Uses ROS camera convention: camera body x=right, y=down, z=forward.

        To avoid the classical look-at gimbal singularity (camera spinning
        around the optical axis when ``forward`` is nearly parallel to
        ``world_up`` — e.g. an overhead top-down camera), the up vector
        is blended with a fixed secondary up vector (world -Y) using a
        smooth weight that goes to 1 when forward is parallel to world_up.
        This eliminates the discontinuity that occurs when ``up_perp``
        shrinks toward zero (overhead cameras).

        If *disable_blending* is True, the standard world_up projection is
        used directly without any secondary-up blending.  This is needed
        for cameras whose forward vector lies close to the YZ plane
        (e.g. cam_chest looking from the front of the table): blending
        world_up=[0,0,1] with secondary_up=[0,-1,0] causes their Y/Z
        components to nearly cancel while X components add, making the
        resulting up vector dominated by noise after normalization.
        """
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = target - eye
        n = float(np.linalg.norm(forward))
        if n < 1.0e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        z_axis = forward / n  # +z in camera frame (ROS) points toward target
        up = np.asarray(world_up, dtype=np.float64)
        cos_fu = float(np.dot(up, z_axis))

        up_perp_primary = up - z_axis * cos_fu
        up_perp_primary_norm = float(np.linalg.norm(up_perp_primary))

        if disable_blending:
            # Pure world_up projection — no secondary-up blending.
            # Avoids the YZ-plane cancellation pathology for cameras
            # like cam_chest whose forward vector lies near the YZ plane.
            if up_perp_primary_norm < 1.0e-9:
                up_perp = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            else:
                up_perp = up_perp_primary / up_perp_primary_norm
            # When the camera looks downward (cos_fu < 0), the pure
            # world_up projection produces an up_perp whose y_axis
            # (= image "down" in ROS convention) points in the +Z
            # hemisphere, making the image appear upside-down.
            # Negate up_perp to restore the correct orientation.
            # This is a stable 180° rotation around the optical axis
            # — small perturbations cannot flip its sign because
            # cam_chest operates far from cos_fu ≈ 0.
            if cos_fu < 0:
                up_perp = -up_perp
        else:
            # Original behaviour: blend world_up with secondary_up
            # across the full cos_fu range to handle gimbal lock.
            alpha = cos_fu * cos_fu  # smooth in [0, 1]
            # Secondary up for overhead/bottom cameras: world -Y so the image
            # top points toward world +Y (robot forward), matching the original
            # GUI overhead view orientation.
            secondary_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(secondary_up, z_axis))) > 0.999:
                secondary_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            sec_proj = secondary_up - z_axis * float(np.dot(secondary_up, z_axis))
            up_perp = (1.0 - alpha) * up_perp_primary + alpha * sec_proj
            up_perp_norm = float(np.linalg.norm(up_perp))
            if up_perp_norm < 1.0e-9:
                up_perp = sec_proj
                up_perp_norm = float(np.linalg.norm(up_perp))
            up_perp /= up_perp_norm
        # ROS camera: y points "down" in image (so use -up_perp as down)
        x_axis = np.cross(z_axis, -up_perp)
        x_axis /= float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # cols = camera axes in world
        return cls._rot_to_quat_wxyz(rot)

    def _camera_sample(self, camera_id: str) -> Dict:
        sample = self._camera_generalization_sample or {}
        if "spatial" in sample:
            sample = sample.get("spatial", {}) or {}
        return (
            sample.get("camera", {})
            .get("cameras", {})
            .get(camera_id, {})
        )

    # ------------------------------------------------------------------
    # Robot body helpers (for mounted cameras)
    # ------------------------------------------------------------------

    def _build_body_name_to_index(self) -> Dict[str, int]:
        articulation = self._robot_articulation
        if articulation is None:
            return {}
        names_obj = None
        for attr in ("body_names", "link_names"):
            if hasattr(articulation, attr):
                names_obj = getattr(articulation, attr)
                if names_obj:
                    break
        if not names_obj and hasattr(articulation, "data"):
            for attr in ("body_names", "link_names"):
                if hasattr(articulation.data, attr):
                    names_obj = getattr(articulation.data, attr)
                    if names_obj:
                        break
        if not names_obj:
            return {}
        return {str(name): idx for idx, name in enumerate(names_obj)}

    def _resolve_mounted_specs(self) -> None:
        self._mounted_specs.clear()
        self._mounted_setup_errors = []
        self._invalid_mounted_camera_ids = set()
        self._blocked_camera_ids = set()
        body_map = self._build_body_name_to_index()
        missing_body: list[str] = []
        for cam_cfg in self._cfg:
            if cam_cfg.mount_type != "robot_link":
                continue
            parent_link = str(cam_cfg.parent_link or "")
            body_index = body_map.get(parent_link)
            if body_index is None:
                missing_body.append(f"{cam_cfg.camera_id}:{parent_link}")
                continue
            offset_xyz, offset_rpy = self._effective_mounted_offset(cam_cfg)
            self._mounted_specs[cam_cfg.camera_id] = _MountedCameraSpec(
                camera_id=cam_cfg.camera_id,
                parent_link=parent_link,
                body_index=int(body_index),
                offset_xyz=offset_xyz,
                offset_quat_ros_wxyz=self._rpy_to_quat_wxyz(offset_rpy),
            )
        if missing_body:
            raise RuntimeError(
                "Could not resolve articulation body indices for robot-link cameras: "
                + ", ".join(sorted(missing_body))
            )

    def _extract_body_vector(self, arr, body_index: int, expected_dim: int) -> np.ndarray | None:
        values = self._to_numpy(arr)
        if values.ndim == 3:
            if values.shape[1] <= body_index or values.shape[2] < expected_dim:
                return None
            return values[0, body_index].astype(np.float32)
        if values.ndim == 2:
            if values.shape[0] > body_index and values.shape[1] >= expected_dim:
                return values[body_index].astype(np.float32)
            if values.shape[1] > body_index and values.shape[0] >= expected_dim:
                return values[:, body_index].astype(np.float32)
        return None

    def _read_body_pose_w(self, body_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        articulation = self._robot_articulation
        if articulation is None:
            return None
        data = getattr(articulation, "data", None)
        if data is None:
            return None
        if hasattr(data, "body_state_w"):
            vec = self._extract_body_vector(getattr(data, "body_state_w"), body_index, expected_dim=7)
            if vec is not None:
                return vec[:3], vec[3:7]
        for pos_attr, quat_attr in (("body_pos_w", "body_quat_w"), ("body_link_pos_w", "body_link_quat_w"), ("link_pos_w", "link_quat_w"), ("body_com_pos_w", "body_com_quat_w")):
            if not hasattr(data, pos_attr) or not hasattr(data, quat_attr):
                continue
            pos = self._extract_body_vector(getattr(data, pos_attr), body_index, expected_dim=3)
            quat = self._extract_body_vector(getattr(data, quat_attr), body_index, expected_dim=4)
            if pos is not None and quat is not None:
                return pos, quat
        if not self._warned_missing_pose_data:
            print("[WARN] Cannot read robot body pose from articulation data for mounted cameras.")
            self._warned_missing_pose_data = True
        return None

    def _convert_camera_quat_convention(
        self,
        quat_wxyz: np.ndarray,
        *,
        origin: str,
        target: str,
    ) -> np.ndarray:
        quat_t = torch.from_numpy(np.asarray(quat_wxyz, dtype=np.float32)).unsqueeze(0).to(self._sim.device)
        converted = convert_camera_frame_orientation_convention(
            quat_t,
            origin=origin,
            target=target,
        )
        return converted[0].detach().cpu().numpy().astype(np.float32)

    def _quat_to_opengl_offset(self, quat_ros_wxyz: np.ndarray) -> np.ndarray:
        return self._convert_camera_quat_convention(quat_ros_wxyz, origin="ros", target="opengl")

    def _quat_to_world_camera(self, quat_opengl_wxyz: np.ndarray) -> np.ndarray:
        return self._convert_camera_quat_convention(quat_opengl_wxyz, origin="opengl", target="world")

    def _mounted_camera_pose_from_articulation(
        self, spec: _MountedCameraSpec
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pose = self._read_body_pose_w(spec.body_index)
        if pose is None:
            return None
        parent_pos_w, parent_quat_wxyz = pose
        parent_rot = quat_wxyz_to_rot(parent_quat_wxyz)
        cam_pos_w = parent_pos_w + parent_rot @ spec.offset_xyz
        cam_quat_wxyz = self._quat_mul_wxyz(parent_quat_wxyz, spec.offset_quat_ros_wxyz)
        return cam_pos_w.astype(np.float32), cam_quat_wxyz.astype(np.float32)

    def _mounted_camera_pose_from_articulation_native(
        self, spec: _MountedCameraSpec
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        pose = self._read_body_pose_w(spec.body_index)
        if pose is None:
            return None
        parent_pos_w, parent_quat_wxyz = pose
        parent_rot = quat_wxyz_to_rot(parent_quat_wxyz)
        cam_pos_w = parent_pos_w + parent_rot @ spec.offset_xyz
        offset_quat_opengl = self._quat_to_opengl_offset(spec.offset_quat_ros_wxyz)
        cam_quat_opengl_wxyz = self._quat_mul_wxyz(parent_quat_wxyz, offset_quat_opengl)
        cam_quat_world_wxyz = self._quat_to_world_camera(cam_quat_opengl_wxyz)
        return (
            cam_pos_w.astype(np.float32),
            cam_quat_opengl_wxyz.astype(np.float32),
            cam_quat_world_wxyz.astype(np.float32),
        )

    _mounted_diag_printed: set[str] = set()

    def _update_mounted_camera_poses(self) -> None:
        self._blocked_camera_ids = set()
        if not self._mounted_specs:
            return
        for cam_id, spec in self._mounted_specs.items():
            if cam_id in self._native_mounted_camera_ids:
                continue
            if cam_id in self._invalid_mounted_camera_ids:
                continue
            cam = self._cameras.get(cam_id)
            if cam is None:
                continue
            # ── one-time diagnostic: print effective mount params ──
            if cam_id not in CameraRig._mounted_diag_printed:
                CameraRig._mounted_diag_printed.add(cam_id)
                body_map = self._build_body_name_to_index()
                body_index = body_map.get(spec.parent_link)
                body_pose = self._read_body_pose_w(spec.body_index)
                body_pos = body_pose[0] if body_pose else None
                print(
                    f"[CameraRig DIAG] {cam_id}: "
                    f"parent_link='{spec.parent_link}' "
                    f"body_index={body_index} "
                    f"offset_xyz={spec.offset_xyz.tolist()} "
                    f"offset_quat_ros_wxyz={spec.offset_quat_ros_wxyz.round(4).tolist()} "
                    f"body_pos_w={None if body_pos is None else body_pos.round(4).tolist()}",
                    flush=True,
                )
            expected_pose = self._mounted_camera_pose_from_articulation(spec)
            if expected_pose is None:
                self._blocked_camera_ids.add(cam_id)
                self._last_capture_errors.append(f"{cam_id}: robot body pose unavailable")
                continue
            cam_pos_w, cam_quat_wxyz = expected_pose
            pos_t = torch.from_numpy(cam_pos_w).unsqueeze(0).to(self._sim.device)
            quat_t = torch.from_numpy(cam_quat_wxyz).unsqueeze(0).to(self._sim.device)
            cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="ros")

    def _repair_stale_native_camera_poses(
        self,
        threshold_m: float = 0.05,
        *,
        force: bool = False,
        check_current: bool = True,
        warn: bool = True,
    ) -> None:
        self._native_pose_overrides = {}
        for cam_id in self._native_mounted_camera_ids:
            spec = self._mounted_specs.get(cam_id)
            cam = self._cameras.get(cam_id)
            if spec is None or cam is None or cam_id in self._invalid_mounted_camera_ids:
                continue
            expected_pose = self._mounted_camera_pose_from_articulation_native(spec)
            if expected_pose is None:
                self._blocked_camera_ids.add(cam_id)
                self._last_capture_errors.append(f"{cam_id}: robot body pose unavailable")
                continue
            expected_pos, expected_quat_opengl, expected_quat_world = expected_pose
            current_pos = None
            if check_current:
                try:
                    data = cam.data
                    current_pos = data.pos_w[0].detach().cpu().numpy().astype(np.float32)
                except Exception:
                    current_pos = None
            distance = float("inf") if current_pos is None else float(np.linalg.norm(current_pos - expected_pos))
            force_pose = force or cam_id in self._force_native_pose_camera_ids
            if not force_pose and distance <= threshold_m:
                continue
            pos_t = torch.from_numpy(expected_pos).unsqueeze(0).to(self._sim.device)
            quat_t = torch.from_numpy(expected_quat_opengl).unsqueeze(0).to(self._sim.device)
            cam.set_world_poses(positions=pos_t, orientations=quat_t, convention="opengl")
            self._native_pose_overrides[cam_id] = (expected_pos, expected_quat_world)
            if not force_pose:
                self._force_native_pose_camera_ids.add(cam_id)
            if warn and cam_id not in self._warned_stale_native_camera_ids:
                print(
                    f"[WARN] {cam_id}: native camera pose was stale by {distance:.3f} m; "
                    "forcing pose from the articulation parent-link transform for the rest of this CameraRig."
                )
                self._warned_stale_native_camera_ids.add(cam_id)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_camera_initialized(cam_id: str, cam: Camera) -> None:
        """Initialize cameras created after the timeline PLAY callback."""
        if bool(getattr(cam, "is_initialized", False)) and hasattr(cam, "_ALL_INDICES"):
            return
        if bool(getattr(cam, "is_initialized", False)) and not hasattr(cam, "_ALL_INDICES"):
            cam._is_initialized = False
        initialize_impl = getattr(cam, "_initialize_impl", None)
        if not callable(initialize_impl):
            raise RuntimeError(f"{cam_id}: camera has no initialization hook")
        initialize_impl()
        if hasattr(cam, "_is_initialized"):
            cam._is_initialized = True
        if not hasattr(cam, "_rep_registry"):
            cam._rep_registry = {}
        if not hasattr(cam, "_ALL_INDICES"):
            view = getattr(cam, "_view", None)
            count = int(getattr(view, "count", 1) or 1)
            device = getattr(cam, "_device", "cpu")
            cam._ALL_INDICES = torch.arange(count, device=device, dtype=torch.long)

    def initialize_after_reset(self) -> None:
        self._resolve_mounted_specs()
        for cam_id, cam in self._cameras.items():
            self._ensure_camera_initialized(cam_id, cam)
        for cam_cfg in self._cfg:
            cam = self._cameras.get(cam_cfg.camera_id)
            if cam is None:
                continue
            if cam_cfg.mount_type != "world":
                continue
            sample = self._camera_sample(cam_cfg.camera_id)
            pos_offset = np.asarray(sample.get("position_offset_m", [0.0, 0.0, 0.0]), dtype=np.float32)
            target_offset = np.asarray(sample.get("target_offset_m", [0.0, 0.0, 0.0]), dtype=np.float32)
            distance_offset = float(sample.get("distance_offset_m", 0.0) or 0.0)
            rpy_offset_deg = tuple(float(v) for v in sample.get("rpy_offset_deg", [0.0, 0.0, 0.0]))
            eye = np.asarray(cam_cfg.position, dtype=np.float32)
            if cam_cfg.target is not None:
                target = np.asarray(cam_cfg.target, dtype=np.float32)
                forward = target - eye
                norm = float(np.linalg.norm(forward))
                if norm > 1.0e-8:
                    eye = eye - (forward / norm) * distance_offset
                eye = eye + pos_offset
                target = target + target_offset
                if any(abs(v) > 1.0e-8 for v in rpy_offset_deg):
                    rotated = self._rpy_deg_to_matrix(rpy_offset_deg) @ (target - eye)
                    target = eye + rotated.astype(np.float32)
                # FIX 2026-05-22: bypass Isaac Lab's set_world_poses_from_view
                # which suffers from look-at gimbal lock when forward is
                # (anti)parallel to world-up (e.g. cam_overhead). A tiny
                # target jitter could otherwise flip the camera 90/180 deg
                # around the optical axis. Compute deterministic look-at
                # quaternion ourselves and push via set_world_poses.
                #
                # cam_chest: disable the secondary-up blending because its
                # forward vector lies close to the YZ plane.  Blending
                # world_up=[0,0,1] with secondary_up=[0,-1,0] causes their
                # Y/Z components to nearly cancel while X components add,
                # amplifying tiny perturbation X components into large
                # camera tilts (~24° from ±2 cm offsets).
                quat = self._look_at_quat_wxyz(
                    eye, target,
                    disable_blending=(cam_cfg.camera_id == "cam_chest"),
                )
                cam.set_world_poses(
                    positions=torch.tensor(np.expand_dims(eye, 0), dtype=torch.float32, device=self._sim.device),
                    orientations=torch.tensor(np.expand_dims(quat, 0), dtype=torch.float32, device=self._sim.device),
                    convention="ros",
                )
            else:
                quat = np.asarray(cam_cfg.quat_wxyz, dtype=np.float32)
                if any(abs(v) > 1.0e-8 for v in rpy_offset_deg):
                    quat = self._quat_mul_wxyz(quat, self._rpy_to_quat_wxyz(tuple(math.radians(v) for v in rpy_offset_deg)))
                cam.set_world_poses(
                    positions=torch.tensor([eye + pos_offset], dtype=torch.float32, device=self._sim.device),
                    orientations=torch.tensor([quat], dtype=torch.float32, device=self._sim.device),
                    convention="world",
                )
        self._update_mounted_camera_poses()
        for cam in self._cameras.values():
            cam.reset()
        self.sync_mounted_camera_poses()

        # P1: Warm caches after first render so intrinsics & world poses are
        # available for all subsequent captures without GPU round-trips.
        self._warmup_caches()

    # ------------------------------------------------------------------
    # P1: Cache warming
    # ------------------------------------------------------------------

    def _warmup_caches(self) -> None:
        """Render every camera once and cache all static metadata."""
        self._static_meta.clear()
        self._caches_warm = False

        dt = self._sim.get_physics_dt()
        for cam_id, cam in self._cameras.items():
            if cam_id in self._invalid_mounted_camera_ids:
                continue
            try:
                cam.update(dt=dt, force_recompute=True)
            except Exception as exc:
                logger.warning("Cache warmup render failed for %s: %s", cam_id, exc)
                continue

            data = cam.data
            cam_cfg = self._cfg_by_id.get(cam_id)

            # Intrinsic (static for the lifetime of the camera)
            intrinsic = data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32)

            # Config-derived fields
            camera_model = "pinhole"
            distortion_coefficients = None
            fisheye_camera_matrix = None
            fisheye_valid_mask = None
            clipping_range = None
            if cam_cfg is not None:
                clipping_range = tuple(float(v) for v in cam_cfg.clipping_range)
                if cam_cfg.camera_type == "fisheye":
                    camera_model = "fisheye"
                    if cam_cfg.fisheye_camera_matrix is not None:
                        fisheye_camera_matrix = np.asarray(cam_cfg.fisheye_camera_matrix, dtype=np.float32)
                    if cam_cfg.fisheye_distortion_coefficients is not None:
                        distortion_coefficients = np.asarray(cam_cfg.fisheye_distortion_coefficients, dtype=np.float32)
                    fisheye_valid_mask = self._get_fisheye_valid_mask(cam_id, cam_cfg)

            # For world-mounted cameras, pose is static → cache full extrinsic + pos/quat
            extrinsic: np.ndarray | None = None
            cached_pos: np.ndarray | None = None
            cached_quat: np.ndarray | None = None
            is_world = cam_cfg is not None and cam_cfg.mount_type == "world"
            if is_world:
                cached_pos = data.pos_w[0].detach().cpu().numpy().astype(np.float32)
                cached_quat = data.quat_w_world[0].detach().cpu().numpy().astype(np.float32)
                # P2: CPU-side rotation
                rot = quat_wxyz_to_rot(cached_quat)
                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = rot
                extrinsic[:3, 3] = cached_pos

            self._static_meta[cam_id] = _CameraStaticMeta(
                intrinsic=intrinsic,
                image_shape=tuple(cam.image_shape),
                camera_model=camera_model,
                clipping_range=clipping_range,
                distortion_coefficients=distortion_coefficients,
                fisheye_camera_matrix=fisheye_camera_matrix,
                fisheye_valid_mask=fisheye_valid_mask,
                extrinsic=extrinsic,
                cam_pos_w=cached_pos,
                cam_quat_wxyz=cached_quat,
            )

        self._caches_warm = True
        logger.info("Camera caches warmed: %d cameras cached", len(self._static_meta))

    def _invalidate_caches(self) -> None:
        """Call when camera configuration changes (e.g. scene rebuild)."""
        self._static_meta.clear()
        self._caches_warm = False

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_camera_buffers(cam: Camera) -> None:
        data_obj = getattr(cam, "_data", None)
        if data_obj is None:
            return
        if hasattr(data_obj, "output"):
            data_obj.output = {}
        if hasattr(data_obj, "info") and isinstance(data_obj.info, list):
            data_types = list(getattr(getattr(cam, "cfg", None), "data_types", []))
            data_obj.info = [{name: None for name in data_types} for _ in data_obj.info]

    def _recover_camera_after_failure(self, cam_id: str, cam: Camera) -> None:
        self._reset_camera_buffers(cam)
        try:
            cam.reset()
        except Exception as exc:
            self._last_capture_errors.append(f"{cam_id}: recovery failed: {exc}")

    # ------------------------------------------------------------------
    # P3: Optimised single-frame data extraction (no redundant copies)
    # ------------------------------------------------------------------

    def _extract_rgb(self, data) -> np.ndarray | None:
        """Extract RGB from camera data with minimal copies (P3)."""
        if not self._enable_rgb or "rgb" not in data.output:
            return None
        rgb_output = data.output["rgb"]
        if rgb_output is None or rgb_output.numel() == 0 or rgb_output.ndim < 4 or rgb_output.shape[0] == 0:
            return None
        rgb_gpu = rgb_output[0]
        if rgb_gpu.numel() == 0 or rgb_gpu.ndim != 3 or rgb_gpu.shape[0] == 0 or rgb_gpu.shape[1] == 0:
            return None
        # Isaac Lab Camera stores RGBA; slice to RGB on GPU before transfer.
        if rgb_gpu.shape[-1] > 3:
            rgb_gpu = rgb_gpu[..., :3]
        if rgb_gpu.shape[-1] != 3:
            return None
        # Ensure contiguous layout, cast to uint8 on GPU, single .cpu() transfer.
        # .cpu() always allocates a fresh CPU tensor, so the GPU buffer can be
        # safely reused by Isaac Lab on the next update without data corruption.
        rgb_gpu = rgb_gpu.to(dtype=torch.uint8).contiguous()
        return rgb_gpu.cpu().numpy()

    def _extract_depth(self, data) -> tuple[np.ndarray | None, str | None]:
        """Extract depth from camera data with minimal copies (P3)."""
        if not self._enable_depth:
            return None, None
        depth_key = "distance_to_image_plane" if "distance_to_image_plane" in data.output else "depth"
        if depth_key not in data.output:
            return None, None
        depth_gpu = data.output[depth_key][0]
        if depth_gpu.dim() == 3 and depth_gpu.shape[-1] == 1:
            depth_gpu = depth_gpu.squeeze(-1)
        depth_gpu = depth_gpu.to(dtype=torch.float32).contiguous()
        return depth_gpu.cpu().numpy(), depth_key

    @staticmethod
    def _apply_fisheye_rgb_mask(rgb: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
        """Black out pixels outside the calibrated fisheye valid circle."""
        if rgb is None or mask is None:
            return rgb
        valid = np.asarray(mask, dtype=np.bool_)
        if valid.shape != rgb.shape[:2]:
            logger.warning(
                "Skipping fisheye RGB mask: mask shape %s does not match image shape %s",
                valid.shape,
                rgb.shape[:2],
            )
            return rgb
        if bool(valid.all()):
            return rgb
        masked = np.array(rgb, copy=True)
        masked[~valid] = 0
        return masked

    def _extract_dynamic_pose(self, cam_id: str, data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract pos / quat / extrinsic for a mounted (moving) camera.

        P2: rotation computed on CPU via numpy instead of GPU matrix_from_quat.
        Only two GPU→CPU syncs: pos_w and quat_w_world.
        """
        override = self._native_pose_overrides.get(cam_id)
        use_reported_pose = os.environ.get("DEX2BENCH_CAMERA_USE_REPORTED_POSE", "").strip().lower() in {"1", "true", "yes"}
        if override is not None and not use_reported_pose:
            cam_pos, cam_quat_wxyz = override
        else:
            cam_pos = data.pos_w[0].detach().cpu().numpy().astype(np.float32)
            cam_quat_wxyz = data.quat_w_world[0].detach().cpu().numpy().astype(np.float32)
        rot = quat_wxyz_to_rot(cam_quat_wxyz)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rot
        extrinsic[:3, 3] = cam_pos
        return cam_pos, cam_quat_wxyz, extrinsic

    # ------------------------------------------------------------------
    # P0: Two-phase capture — render all, then extract all
    # ------------------------------------------------------------------

    def _extract_camera_frame(self, cam_id: str, cam: Camera) -> CameraFrame:
        """Phase 2 of capture: extract data from an already-rendered camera.

        Uses cached static metadata (P1) and optimised extraction (P2/P3).
        """
        data = cam.data

        # P3: optimised image extraction
        rgb = self._extract_rgb(data)
        if self._enable_rgb and rgb is None:
            raw_rgb = data.output.get("rgb")
            raw_shape = tuple(raw_rgb.shape) if raw_rgb is not None else None
            raise RuntimeError(f"RGB output is empty or invalid (raw_shape={raw_shape})")
        depth_m, depth_semantics = self._extract_depth(data)

        # P1: retrieve cached static metadata
        meta = self._static_meta.get(cam_id)
        if meta is not None:
            intrinsic = meta.intrinsic
            image_shape = meta.image_shape
            camera_model = meta.camera_model
            clipping_range = meta.clipping_range
            distortion_coefficients = meta.distortion_coefficients
            fisheye_camera_matrix = meta.fisheye_camera_matrix
            fisheye_valid_mask = meta.fisheye_valid_mask
            # World cameras: fully cached extrinsic + pos + quat
            if meta.extrinsic is not None:
                extrinsic = meta.extrinsic
                cam_pos = meta.cam_pos_w
                cam_quat_wxyz = meta.cam_quat_wxyz
            else:
                # Mounted camera: dynamic pose (P2)
                cam_pos, cam_quat_wxyz, extrinsic = self._extract_dynamic_pose(cam_id, data)
        else:
            # Fallback: no cache available (should not happen after warmup)
            intrinsic = data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32)
            cam_pos, cam_quat_wxyz, extrinsic = self._extract_dynamic_pose(cam_id, data)
            image_shape = tuple(cam.image_shape)
            cam_cfg = self._cfg_by_id.get(cam_id)
            camera_model = "pinhole"
            distortion_coefficients = None
            fisheye_camera_matrix = None
            fisheye_valid_mask = None
            clipping_range = None
            if cam_cfg is not None:
                clipping_range = tuple(float(v) for v in cam_cfg.clipping_range)
                if cam_cfg.camera_type == "fisheye":
                    camera_model = "fisheye"
                    if cam_cfg.fisheye_camera_matrix is not None:
                        fisheye_camera_matrix = np.asarray(cam_cfg.fisheye_camera_matrix, dtype=np.float32)
                    if cam_cfg.fisheye_distortion_coefficients is not None:
                        distortion_coefficients = np.asarray(cam_cfg.fisheye_distortion_coefficients, dtype=np.float32)
                    fisheye_valid_mask = self._get_fisheye_valid_mask(cam_id, cam_cfg)

        # Fisheye depth correction: IsaacSim outputs ray distance (Euclidean
        # distance to camera origin) for fisheye cameras, even when the render
        # key is "distance_to_image_plane".  Record the true semantics so
        # downstream consumers (TSDF, depth visualisation) interpret correctly.
        if camera_model == "fisheye" and depth_semantics == "distance_to_image_plane":
            depth_semantics = "distance_to_camera"

        if camera_model == "fisheye":
            rgb = self._apply_fisheye_rgb_mask(rgb, fisheye_valid_mask)

        return CameraFrame(
            rgb=rgb,
            depth_m=depth_m,
            intrinsic=intrinsic,
            extrinsic_world_from_cam=extrinsic,
            cam_pos_w=cam_pos,
            cam_quat_wxyz=cam_quat_wxyz,
            image_shape=image_shape,
            depth_semantics=depth_semantics,
            camera_model=camera_model,
            distortion_coefficients=distortion_coefficients,
            fisheye_camera_matrix=fisheye_camera_matrix,
            fisheye_valid_mask=fisheye_valid_mask,
            clipping_range=clipping_range,
        )

    def capture(self, dt: float) -> Dict[str, CameraFrame]:
        """Extract camera data from the frame rendered by the caller."""
        self._last_capture_errors = list(self._mounted_setup_errors)

        # Do not mutate mounted camera poses here. IsaacLab Camera.update()
        # reads annotator outputs from the already-rendered frame; moving a
        # camera after sim.render() makes RGB and saved extrinsics disagree.
        # Call sync_mounted_camera_poses() before sim.render() instead.
        pre_render_blocked = set(self._blocked_camera_ids)
        self._blocked_camera_ids = pre_render_blocked

        # Build active camera list (skip invalid / blocked)
        skip = self._invalid_mounted_camera_ids | self._blocked_camera_ids
        active: list[tuple[str, Camera]] = [
            (cam_id, cam) for cam_id, cam in self._cameras.items()
            if cam_id not in skip
        ]

        # ── Phase 1: GPU render ALL cameras ──────────────────────────
        rendered: list[tuple[str, Camera]] = []
        for cam_id, cam in active:
            try:
                cam.update(dt=dt, force_recompute=True)
                rendered.append((cam_id, cam))
            except Exception as exc:
                self._last_capture_errors.append(f"{cam_id}: render failed: {exc}")
                self._recover_camera_after_failure(cam_id, cam)

        # ── Phase 2: extract data from ALL rendered cameras ──────────
        frames: Dict[str, CameraFrame] = {}
        for cam_id, cam in rendered:
            try:
                frames[cam_id] = self._extract_camera_frame(cam_id, cam)
            except Exception as exc:
                self._last_capture_errors.append(f"{cam_id}: extract failed: {exc}")
                self._recover_camera_after_failure(cam_id, cam)

        return frames

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        render_product_paths: set[str] = set()
        for cam_id, cam in list(self._cameras.items()):
            # SensorBase.reset() only clears frame buffers and raises when the
            # sensor has already been invalidated by a timeline STOP. Scene
            # rebuild needs actual resource teardown instead.
            render_paths = list(getattr(cam, "_render_product_paths", []) or [])
            render_product_paths.update(str(path) for path in render_paths if path)
            registry = getattr(cam, "_rep_registry", None)
            if registry is not None:
                for annotators in list(registry.values()):
                    for annotator, render_path in zip(list(annotators), render_paths):
                        try:
                            annotator.detach([render_path])
                        except Exception as exc:
                            logger.debug("Failed to detach camera annotator %s: %s", cam_id, exc)
                registry.clear()
            # Always ensure _rep_registry exists so Camera.__del__ doesn't crash
            cam._rep_registry = {}
            try:
                clear_callbacks = getattr(cam, "_clear_callbacks", None)
                if callable(clear_callbacks):
                    clear_callbacks()
            except Exception as exc:
                print(f"[WARN] camera '{cam_id}' cleanup failed: {exc}")
        self._cameras.clear()

        if render_product_paths:
            stage = get_current_stage_compat()
            for render_product_path in sorted(render_product_paths):
                try:
                    delete_prim_compat(render_product_path, stage=stage)
                except Exception as exc:
                    logger.debug("Failed to delete camera render product %s: %s", render_product_path, exc)
            self._flush_stage_after_camera_delete()

        # Delete camera USD prims so they don't block parent prim deletion
        # during full scene rebuild.  Native-mounted cameras are parented
        # under robot links; without this, clear_scene_prims_preserve_robot
        # cannot delete the GlobalRobot prim between episodes.
        #
        # Use raw USD stage.RemovePrim() because DeletePrimsCommand (used
        # by isaaclab delete_prim) cannot delete prims under an articulation
        # while the articulation is still loaded.
        if self._camera_base_prim_paths:
            from pxr import Sdf

            stage = get_current_stage_compat()
            for base_prim_path in self._camera_base_prim_paths.values():
                try:
                    stage.RemovePrim(Sdf.Path(base_prim_path))
                except Exception as exc:
                    logger.warning("Failed to delete camera prim %s: %s", base_prim_path, exc)

        self._mounted_specs.clear()
        self._body_name_to_index = {}
        self._robot_articulation = None
        self._mounted_setup_errors = []
        self._invalid_mounted_camera_ids = set()
        self._blocked_camera_ids = set()
        self._last_capture_errors = []
        self._native_pose_overrides = {}
        self._force_native_pose_camera_ids = set()
        self._invalidate_caches()
