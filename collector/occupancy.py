"""Multi-view occupancy labels derived from depth."""

from __future__ import annotations

from enum import IntEnum
from typing import Dict

import numpy as np

from .camera_geometry import is_geometry_compatible, project_camera_points_dispatch, world_points_to_camera


DEFAULT_CHUNK_SIZE = 262144
DEFAULT_MAX_VOXEL_COUNT = 2000000
DEFAULT_REQUIRE_ALL_CAMERAS = True


class OccupancyState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


def compute_occupancy_grid_shape(bounds: np.ndarray, voxel_size: float) -> np.ndarray:
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    if bounds_arr.shape != (2, 3):
        raise ValueError(f"occupancy.bounds must have shape (2, 3), got {bounds_arr.shape}")
    voxel_size_value = float(voxel_size)
    if voxel_size_value <= 0.0:
        raise ValueError("occupancy.voxel_size must be > 0")

    extent = bounds_arr[1] - bounds_arr[0]
    if np.any(extent <= 0.0):
        raise ValueError(f"occupancy.bounds must be strictly increasing, got {bounds_arr.tolist()}")

    steps = extent / voxel_size_value
    rounded = np.rint(steps)
    snapped = np.where(np.isclose(steps, rounded, rtol=0.0, atol=1.0e-6), rounded, np.ceil(steps))
    return np.maximum(1, snapped.astype(np.int32))


def compute_occupancy_voxel_count(bounds: np.ndarray, voxel_size: float) -> int:
    grid_shape = compute_occupancy_grid_shape(bounds, voxel_size).astype(np.int64)
    return int(np.prod(grid_shape, dtype=np.int64))


class OccupancyLabeler:
    def __init__(self, enabled: bool, config: Dict):
        self.enabled = enabled
        self.config = dict(config)
        self._voxel_size = float(config.get("voxel_size", 0.02))
        bounds = config.get("bounds", [[-0.8, -0.8, 0.0], [0.8, 0.8, 1.2]])
        self._bounds = np.asarray(bounds, dtype=np.float64)
        self._occupied_margin = float(config.get("occupied_margin", 0.02))
        self._free_margin = float(config.get("free_margin", self._occupied_margin))
        self._chunk_size = max(1, int(config.get("chunk_size", DEFAULT_CHUNK_SIZE)))
        self._max_voxel_count = max(0, int(config.get("max_voxel_count", DEFAULT_MAX_VOXEL_COUNT)))
        self._require_all_cameras = bool(config.get("require_all_cameras", DEFAULT_REQUIRE_ALL_CAMERAS))
        self._allow_depth_fallback = bool(config.get("allow_depth_fallback", False))
        expected_camera_ids = config.get("expected_camera_ids", [])
        self._expected_camera_ids = tuple(str(cam_id) for cam_id in expected_camera_ids)
        self._fusion_mode = str(config.get("fusion_mode", "binary"))
        self._truncation_distance = float(config.get("truncation_distance", 3.0 * self._voxel_size))
        self._surface_threshold = float(config.get("surface_threshold", 0.5 * self._voxel_size))
        self._grid_shape: np.ndarray | None = None
        self._voxel_centers: np.ndarray | None = None
        self._voxel_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _ensure_grid(self) -> None:
        if self._grid_shape is not None and self._voxel_axes is not None:
            return
        self._grid_shape = compute_occupancy_grid_shape(self._bounds, self._voxel_size)
        voxel_count = int(np.prod(self._grid_shape.astype(np.int64), dtype=np.int64))
        if self._max_voxel_count > 0 and voxel_count > self._max_voxel_count:
            raise ValueError(
                f"occupancy voxel count {voxel_count} exceeds occupancy.max_voxel_count={self._max_voxel_count}."
            )
        self._voxel_axes = self._build_voxel_axes()

    def _build_voxel_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._grid_shape is None:
            raise RuntimeError("occupancy grid shape is not initialized")
        return tuple(
            (
                self._bounds[0, i]
                + (np.arange(int(self._grid_shape[i]), dtype=np.float32) + 0.5) * self._voxel_size
            ).astype(np.float32)
            for i in range(3)
        )

    @staticmethod
    def _chunk_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [bounds_min[0], bounds_min[1], bounds_min[2]],
                [bounds_min[0], bounds_min[1], bounds_max[2]],
                [bounds_min[0], bounds_max[1], bounds_min[2]],
                [bounds_min[0], bounds_max[1], bounds_max[2]],
                [bounds_max[0], bounds_min[1], bounds_min[2]],
                [bounds_max[0], bounds_min[1], bounds_max[2]],
                [bounds_max[0], bounds_max[1], bounds_min[2]],
                [bounds_max[0], bounds_max[1], bounds_max[2]],
            ],
            dtype=np.float32,
        )

    def _camera_may_see_chunk(self, camera_frame, bounds_min: np.ndarray, bounds_max: np.ndarray) -> bool:
        corners_world = self._chunk_corners(bounds_min, bounds_max)
        points_cam, positive_z = world_points_to_camera(corners_world, camera_frame.extrinsic_world_from_cam)
        if not np.any(positive_z):
            return False
        if not np.all(positive_z):
            return True
        uv = project_camera_points_dispatch(points_cam, camera_frame)
        height, width = camera_frame.image_shape
        return not (
            float(np.max(uv[:, 0])) < 0.0
            or float(np.min(uv[:, 0])) >= float(width)
            or float(np.max(uv[:, 1])) < 0.0
            or float(np.min(uv[:, 1])) >= float(height)
        )

    def _iter_voxel_chunks(self):
        if self._grid_shape is None or self._voxel_axes is None:
            raise RuntimeError("occupancy grid is not initialized")
        grid_shape = self._grid_shape.astype(np.int64)
        total_voxels = int(np.prod(grid_shape, dtype=np.int64))
        yz_stride = int(grid_shape[1] * grid_shape[2])
        z_stride = int(grid_shape[2])
        half_voxel = np.float32(self._voxel_size * 0.5)
        axes_x, axes_y, axes_z = self._voxel_axes

        for start in range(0, total_voxels, self._chunk_size):
            stop = min(total_voxels, start + self._chunk_size)
            flat_index = np.arange(start, stop, dtype=np.int64)
            x_idx = flat_index // yz_stride
            yz_index = flat_index % yz_stride
            y_idx = yz_index // z_stride
            z_idx = yz_index % z_stride

            centers = np.empty((flat_index.shape[0], 3), dtype=np.float32)
            centers[:, 0] = axes_x[x_idx]
            centers[:, 1] = axes_y[y_idx]
            centers[:, 2] = axes_z[z_idx]

            bounds_min = centers.min(axis=0) - half_voxel
            bounds_max = centers.max(axis=0) + half_voxel
            yield start, stop, centers, bounds_min.astype(np.float32), bounds_max.astype(np.float32)

    @staticmethod
    def _voxel_depth(points_cam: np.ndarray, camera_frame) -> np.ndarray:
        """Compute voxel depth metric matching the camera's depth_m convention.

        Pinhole: Z-depth (distance along optical axis).
        Fisheye: ray distance (Euclidean), since wide-FOV depth maps
                 store ray distance even under distance_to_image_plane semantics.
        """
        if getattr(camera_frame, "camera_model", "pinhole") == "fisheye":
            return np.sqrt(
                points_cam[:, 0] ** 2 + points_cam[:, 1] ** 2 + points_cam[:, 2] ** 2
            ).astype(np.float32)
        return points_cam[:, 2]

    @staticmethod
    def _has_valid_depth_observation(camera_frame) -> bool:
        depth_m = getattr(camera_frame, "depth_m", None)
        if depth_m is None:
            return False
        depth_arr = np.asarray(depth_m)
        return bool(np.any(np.isfinite(depth_arr)))

    @classmethod
    def _usable_camera_frames(cls, camera_frames: Dict) -> Dict:
        usable: Dict[str, object] = {}
        for cam_id, camera_frame in (camera_frames or {}).items():
            if camera_frame is None:
                continue
            if not is_geometry_compatible(camera_frame):
                continue
            if not cls._has_valid_depth_observation(camera_frame):
                continue
            usable[str(cam_id)] = camera_frame
        return usable
    def _validate_depth_semantics(self, usable_camera_frames: Dict[str, object]) -> None:
        for cam_id, camera_frame in usable_camera_frames.items():
            depth_semantics = getattr(camera_frame, "depth_semantics", None) or "distance_to_image_plane"
            if depth_semantics in ("distance_to_image_plane", "distance_to_camera"):
                continue
            if depth_semantics == "depth" and self._allow_depth_fallback:
                continue
            raise ValueError(
                f"camera '{cam_id}' depth semantics '{depth_semantics}' are not supported without allow_depth_fallback=true"
            )

    def on_step(self, *args, **kwargs) -> Dict:
        if not self.enabled:
            return {}

        self._ensure_grid()
        assert self._grid_shape is not None

        camera_frames = kwargs.get("camera_frames", {}) or {}
        usable_camera_frames = self._usable_camera_frames(camera_frames)
        self._validate_depth_semantics(usable_camera_frames)
        if not usable_camera_frames:
            return {}
        if self._require_all_cameras and self._expected_camera_ids:
            missing_camera_ids = [cam_id for cam_id in self._expected_camera_ids if cam_id not in usable_camera_frames]
            if missing_camera_ids:
                return {}

        if self._fusion_mode == "tsdf":
            return self._fuse_tsdf(usable_camera_frames)
        return self._fuse_binary(usable_camera_frames)

    # ------------------------------------------------------------------
    # Binary fusion (original algorithm)
    # ------------------------------------------------------------------

    def _fuse_binary(self, usable_camera_frames: Dict) -> Dict:
        state = np.full(tuple(int(v) for v in self._grid_shape), OccupancyState.UNKNOWN, dtype=np.uint8)
        total_voxels = int(np.prod(self._grid_shape.astype(np.int64), dtype=np.int64))
        free_any = np.zeros((total_voxels,), dtype=np.bool_)
        occupied_any = np.zeros((total_voxels,), dtype=np.bool_)

        for camera_frame in usable_camera_frames.values():
            for start, stop, centers_world, bounds_min, bounds_max in self._iter_voxel_chunks():
                if not self._camera_may_see_chunk(camera_frame, bounds_min, bounds_max):
                    continue
                points_cam, positive_z = world_points_to_camera(centers_world, camera_frame.extrinsic_world_from_cam)
                uv = project_camera_points_dispatch(points_cam, camera_frame)
                height, width = camera_frame.image_shape
                in_frame = (
                    positive_z
                    & (uv[:, 0] >= 0.0)
                    & (uv[:, 0] < float(width))
                    & (uv[:, 1] >= 0.0)
                    & (uv[:, 1] < float(height))
                )
                if not np.any(in_frame):
                    continue
                u_idx = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, width - 1)
                v_idx = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, height - 1)
                # For fisheye cameras, exclude pixels outside the valid circle
                valid_mask = getattr(camera_frame, "fisheye_valid_mask", None)
                if valid_mask is not None:
                    in_frame = in_frame & valid_mask[v_idx, u_idx]
                    if not np.any(in_frame):
                        continue
                depth_samples = camera_frame.depth_m[v_idx, u_idx].astype(np.float32)
                finite_depth = np.isfinite(depth_samples)
                visible = in_frame & finite_depth
                voxel_depth = self._voxel_depth(points_cam, camera_frame)
                free_mask = visible & (voxel_depth < depth_samples - self._free_margin)
                occupied_mask = visible & (np.abs(voxel_depth - depth_samples) <= self._occupied_margin)

                free_any[start:stop] |= free_mask
                occupied_any[start:stop] |= occupied_mask

        flat_state = state.reshape(-1)
        flat_state[free_any] = OccupancyState.FREE
        # Occupancy evidence is stronger than free-space evidence from another view.
        flat_state[occupied_any] = OccupancyState.OCCUPIED

        return {
            "grid_shape": self._grid_shape.astype(np.int32),
            "voxel_size": float(self._voxel_size),
            "bounds": self._bounds.astype(np.float32),
            "state": flat_state.reshape(state.shape).astype(np.uint8),
        }

    # ------------------------------------------------------------------
    # TSDF fusion (weighted signed-distance averaging + zero-crossing)
    # ------------------------------------------------------------------

    def _fuse_tsdf(self, usable_camera_frames: Dict) -> Dict:
        total_voxels = int(np.prod(self._grid_shape.astype(np.int64), dtype=np.int64))
        tsdf_sum = np.zeros(total_voxels, dtype=np.float64)
        weight_sum = np.zeros(total_voxels, dtype=np.float64)
        mu = self._truncation_distance

        for camera_frame in usable_camera_frames.values():
            for start, stop, centers_world, bounds_min, bounds_max in self._iter_voxel_chunks():
                if not self._camera_may_see_chunk(camera_frame, bounds_min, bounds_max):
                    continue
                points_cam, positive_z = world_points_to_camera(
                    centers_world, camera_frame.extrinsic_world_from_cam
                )
                uv = project_camera_points_dispatch(points_cam, camera_frame)
                height, width = camera_frame.image_shape
                in_frame = (
                    positive_z
                    & (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width))
                    & (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
                )
                if not np.any(in_frame):
                    continue
                u_idx = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, width - 1)
                v_idx = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, height - 1)
                valid_mask = getattr(camera_frame, "fisheye_valid_mask", None)
                if valid_mask is not None:
                    in_frame = in_frame & valid_mask[v_idx, u_idx]
                    if not np.any(in_frame):
                        continue
                depth_samples = camera_frame.depth_m[v_idx, u_idx].astype(np.float32)
                visible = in_frame & np.isfinite(depth_samples)

                # SDF: positive = voxel in front of surface (free),
                #       zero = at surface, negative = behind surface.
                voxel_depth = self._voxel_depth(points_cam, camera_frame)
                sdf = depth_samples - voxel_depth
                within_trunc = visible & (sdf > -mu)
                sdf_clamped = np.clip(sdf, -mu, mu)

                # NaN-safe accumulation: np.where avoids 0.0 * NaN = NaN.
                contribution = np.where(within_trunc, sdf_clamped, 0.0)
                tsdf_sum[start:stop] += contribution
                weight_sum[start:stop] += np.where(within_trunc, 1.0, 0.0)

        # Classification from averaged TSDF.
        observed = weight_sum > 0
        tsdf_avg = np.zeros(total_voxels, dtype=np.float64)
        np.divide(tsdf_sum, weight_sum, where=observed, out=tsdf_avg)

        state = np.full(tuple(int(v) for v in self._grid_shape),
                        OccupancyState.UNKNOWN, dtype=np.uint8)
        flat_state = state.reshape(-1)
        th = self._surface_threshold
        flat_state[observed & (tsdf_avg > th)] = OccupancyState.FREE
        flat_state[observed & (np.abs(tsdf_avg) <= th)] = OccupancyState.OCCUPIED
        # observed & (tsdf_avg < -th) → behind surface → stays UNKNOWN

        return {
            "grid_shape": self._grid_shape.astype(np.int32),
            "voxel_size": float(self._voxel_size),
            "bounds": self._bounds.astype(np.float32),
            "state": flat_state.reshape(state.shape).astype(np.uint8),
        }


