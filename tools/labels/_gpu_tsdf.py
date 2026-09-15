"""GPU-accelerated TSDF depth fusion for offline occupancy label generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tools.labels._accel import CameraStaticData, torch_available, to_torch, to_numpy
from collector.occupancy import OccupancyState, compute_occupancy_grid_shape

if TYPE_CHECKING:
    import torch

if torch_available():
    import torch as _torch
else:
    _torch = None  # type: ignore[assignment]


class TSDFFuser:
    """GPU TSDF fusion — processes all voxels at once per camera (no chunk loop).

    Pre-computes voxel center coordinates on device once and reuses them
    across all frames and cameras.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        voxel_size: float,
        device: "torch.device",
        truncation_distance: float,
        surface_threshold: float,
    ) -> None:
        if _torch is None:
            raise RuntimeError("torch is not installed")

        self._device = device
        self._mu = truncation_distance
        self._threshold = surface_threshold
        self._voxel_size = voxel_size
        self._bounds = np.asarray(bounds, dtype=np.float32)

        grid_shape = compute_occupancy_grid_shape(bounds, voxel_size)
        self._grid_shape = tuple(int(v) for v in grid_shape)
        nx, ny, nz = self._grid_shape
        self._V = nx * ny * nz

        # Pre-compute voxel centers on device (reused across ALL frames)
        xs = self._bounds[0, 0] + (_torch.arange(nx, device=device, dtype=_torch.float32) + 0.5) * voxel_size
        ys = self._bounds[0, 1] + (_torch.arange(ny, device=device, dtype=_torch.float32) + 0.5) * voxel_size
        zs = self._bounds[0, 2] + (_torch.arange(nz, device=device, dtype=_torch.float32) + 0.5) * voxel_size
        gx, gy, gz = _torch.meshgrid(xs, ys, zs, indexing="ij")
        self._voxel_centers = _torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

        # Running accumulators
        self._tsdf_sum = _torch.zeros(self._V, dtype=_torch.float32, device=device)
        self._weight_sum = _torch.zeros(self._V, dtype=_torch.float32, device=device)

    def reset(self) -> None:
        """Zero accumulators for processing a new episode."""
        self._tsdf_sum.zero_()
        self._weight_sum.zero_()

    def fuse_frame(
        self,
        depth_maps: dict[str, np.ndarray],
        extrinsics: dict[str, np.ndarray],
        camera_static: dict[str, CameraStaticData],
    ) -> None:
        """Fuse depth observations from all cameras for one frame."""
        for cam_id, depth_np in depth_maps.items():
            static = camera_static.get(cam_id)
            if static is None:
                continue
            extr_np = extrinsics.get(cam_id)
            if extr_np is None:
                continue

            depth_t = to_torch(depth_np, self._device)  # (H, W)
            H, W = depth_t.shape

            extr_t = to_torch(extr_np, self._device)  # (4, 4)
            cam_pos = extr_t[:3, 3]  # (3,)
            R = extr_t[:3, :3]  # (3, 3)

            # World to camera
            diff = self._voxel_centers - cam_pos.unsqueeze(0)  # (V, 3)
            raw_cam = diff @ R  # (V, 3)

            # Isaac-to-optical: x=-y_raw, y=-z_raw, z=x_raw
            points_cam = _torch.stack(
                [-raw_cam[:, 1], -raw_cam[:, 2], raw_cam[:, 0]], dim=-1
            )  # (V, 3)

            X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

            # Project to pixel coordinates
            if static.camera_model == "fisheye" and static.fisheye_matrix is not None:
                uv, voxel_depth = self._project_fisheye(X, Y, Z, static)
            else:
                uv, voxel_depth = self._project_pinhole(X, Y, Z, static)

            u, v = uv[:, 0], uv[:, 1]

            # Valid mask
            in_frame = (Z > 1.0e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            # Fisheye valid circle mask
            if static.valid_mask is not None:
                mask_t = to_torch(static.valid_mask.astype(np.uint8), self._device)
                u_int = u.round().clamp(0, W - 1).long()
                v_int = v.round().clamp(0, H - 1).long()
                in_frame = in_frame & mask_t[v_int, u_int].bool()

            # Gather depth
            u_idx = u.round().clamp(0, W - 1).long()
            v_idx = v.round().clamp(0, H - 1).long()
            depth_samples = depth_t[v_idx, u_idx]  # (V,)

            visible = in_frame & _torch.isfinite(depth_samples)

            # SDF computation
            sdf = depth_samples - voxel_depth
            within_trunc = visible & (sdf > -self._mu)
            sdf_clamped = sdf.clamp(-self._mu, self._mu)

            # Accumulate (multiply by mask avoids temporary tensor allocations)
            within_f = within_trunc.float()
            self._tsdf_sum += sdf_clamped * within_f
            self._weight_sum += within_f

    def classify(self) -> np.ndarray:
        """Classify voxels from accumulated TSDF. Returns ``(nx, ny, nz)`` uint8."""
        observed = self._weight_sum > 0
        tsdf_avg = _torch.where(
            observed,
            self._tsdf_sum / self._weight_sum.clamp(min=1.0e-8),
            _torch.zeros_like(self._tsdf_sum),
        )

        state = _torch.zeros(self._V, dtype=_torch.uint8, device=self._device)
        state[observed & (tsdf_avg > self._threshold)] = OccupancyState.FREE
        state[observed & (tsdf_avg.abs() <= self._threshold)] = OccupancyState.OCCUPIED

        return to_numpy(state).reshape(self._grid_shape)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self._grid_shape

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def voxel_size(self) -> float:
        return self._voxel_size

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_pinhole(
        self, X: "torch.Tensor", Y: "torch.Tensor", Z: "torch.Tensor",
        static: CameraStaticData,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        z_safe = Z.clamp(min=1.0e-6)
        fx = float(static.intrinsic[0, 0])
        fy = float(static.intrinsic[1, 1])
        cx = float(static.intrinsic[0, 2])
        cy = float(static.intrinsic[1, 2])
        u = fx * X / z_safe + cx
        v = fy * Y / z_safe + cy
        uv = _torch.stack([u, v], dim=-1)
        return uv, Z  # voxel_depth = Z for pinhole

    def _project_fisheye(
        self, X: "torch.Tensor", Y: "torch.Tensor", Z: "torch.Tensor",
        static: CameraStaticData,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r = _torch.sqrt(X * X + Y * Y)
        theta = _torch.atan2(r, Z)

        k1, k2, k3, k4 = (float(v) for v in static.distortion[:4])
        theta2 = theta * theta
        theta_d = theta * (
            1.0 + k1 * theta2 + k2 * theta2 ** 2 + k3 * theta2 ** 3 + k4 * theta2 ** 4
        )

        on_axis = r < 1.0e-8
        safe_r = _torch.where(on_axis, _torch.ones_like(r), r)
        scale = _torch.where(on_axis, 1.0 / Z.clamp(min=1.0e-6), theta_d / safe_r)

        K = static.fisheye_matrix
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        u = fx * scale * X + cx
        v = fy * scale * Y + cy
        uv = _torch.stack([u, v], dim=-1)

        # Fisheye depth = ray distance (Euclidean)
        voxel_depth = _torch.sqrt(X * X + Y * Y + Z * Z)
        return uv, voxel_depth
