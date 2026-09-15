"""Batch box3d / box2d label generation and camera geometry across all frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tools.labels._accel import (
    EpisodeBatchData,
    CameraStaticData,
    torch_available,
    to_torch,
    to_numpy,
)

if TYPE_CHECKING:
    import torch

if torch_available():
    import torch as _torch
else:
    _torch = None  # type: ignore[assignment]


# ===================================================================
# Batch camera geometry (torch)
# ===================================================================

def batch_quat_xyzw_to_rot(q: "torch.Tensor") -> "torch.Tensor":
    """Vectorized quaternion (xyzw) to rotation matrix.

    Args:
        q: ``(N, 4)`` tensor with ``[x, y, z, w]`` ordering.

    Returns:
        ``(N, 3, 3)`` rotation matrices (float32).
    """
    q = q.float()
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = _torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=_torch.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


# Pre-computed sign matrix for 8 box corners: shape (8, 3)
_CORNER_SIGNS = np.array(
    [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ],
    dtype=np.float32,
)

_INVISIBLE_XYXY = np.array([-1, -1, -1, -1], dtype=np.int32)


# ---------------------------------------------------------------------------
# GPU batch box3d
# ---------------------------------------------------------------------------

def batch_box3d_all_frames(
    batch_data: EpisodeBatchData,
    device: "torch.device",
) -> dict[str, dict]:
    """Compute box3d labels for ALL frames at once on *device*.

    Returns ``{obj_id: {"center_world": (F,3), "size_lwh": (3,),
    "quat_world": (F,4), "corners_world": (F,8,3)}}`` as numpy arrays.
    """
    if batch_data.local_bboxes is None:
        return {}

    results: dict[str, dict] = {}
    corner_signs_t = to_torch(_CORNER_SIGNS, device)  # (8, 3)

    for obj_id, poses_np in batch_data.poses.items():
        bbox = batch_data.local_bboxes.get(obj_id)
        if bbox is None:
            continue

        lo = np.asarray(bbox[0], dtype=np.float32)
        hi = np.asarray(bbox[1], dtype=np.float32)
        size = hi - lo  # (3,)
        center_local = 0.5 * (lo + hi)  # (3,)
        corners_local = _CORNER_SIGNS * (0.5 * size)  # (8, 3)

        poses_t = to_torch(poses_np, device)  # (F, 7)
        positions = poses_t[:, :3]  # (F, 3)
        quats = poses_t[:, 3:7]  # (F, 4)  xyzw

        # Mask NaN frames: replace with valid dummy values for torch computation,
        # then overwrite results with NaN afterwards.
        finite_mask = _torch.isfinite(poses_t).all(dim=1)  # (F,)
        if not finite_mask.all():
            positions = positions.clone()
            quats = quats.clone()
            positions[~finite_mask] = 0.0
            quats[~finite_mask] = _torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

        R = batch_quat_xyzw_to_rot(quats)  # (F, 3, 3)

        center_local_t = to_torch(center_local, device)  # (3,)
        corners_local_t = to_torch(corners_local, device)  # (8, 3)

        # center_world[f] = positions[f] + R[f] @ center_local
        center_world = positions + _torch.einsum("fij,j->fi", R, center_local_t)  # (F, 3)

        # corners_world[f, c] = corners_local[c] @ R[f].T + center_world[f]
        corners_world = (
            _torch.einsum("cj,fkj->fck", corners_local_t, R) + center_world.unsqueeze(1)
        )  # (F, 8, 3)

        results[obj_id] = {
            "center_world": to_numpy(center_world),
            "size_lwh": size,
            "quat_world": to_numpy(quats),
            "corners_world": to_numpy(corners_world),
        }
        # Restore NaN for frames with invalid poses
        if not finite_mask.all():
            nan_mask = to_numpy(~finite_mask)
            results[obj_id]["center_world"][nan_mask] = np.nan
            results[obj_id]["quat_world"][nan_mask] = np.nan
            results[obj_id]["corners_world"][nan_mask] = np.nan

    return results


# ---------------------------------------------------------------------------
# GPU batch box2d
# ---------------------------------------------------------------------------

def batch_box2d_all_frames(
    box3d_results: dict[str, dict],
    batch_data: EpisodeBatchData,
    device: "torch.device",
) -> dict[str, dict[str, dict]]:
    """Project all 3D box corners to 2D for all cameras/frames at once.

    Returns ``{cam_id: {obj_id: {"xyxy": (F,4) int32, "visible": (F,) bool}}}``.
    """
    out: dict[str, dict[str, dict]] = {}

    for cam_id, static in batch_data.camera_static.items():
        if static.camera_model == "fisheye" and (
            static.fisheye_matrix is None or static.distortion is None
        ):
            continue

        extr_np = batch_data.extrinsics[cam_id]  # (F, 4, 4)
        F_frames = extr_np.shape[0]
        H, W = static.image_shape

        cam_labels: dict[str, dict] = {}

        for obj_id, box3d in box3d_results.items():
            corners_np = box3d["corners_world"]  # (F, 8, 3)
            xyxy, visible = _project_corners_to_xyxy(
                corners_np, extr_np, static, device, H, W,
            )
            cam_labels[obj_id] = {"xyxy": xyxy, "visible": visible}

        out[cam_id] = cam_labels

    return out


def _project_corners_to_xyxy(
    corners_np: np.ndarray,
    extr_np: np.ndarray,
    static: CameraStaticData,
    device: "torch.device",
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (F,8,3) world corners through per-frame extrinsics → (F,4) xyxy."""
    corners_t = to_torch(corners_np, device)  # (F, 8, 3)
    extr_t = to_torch(extr_np, device)  # (F, 4, 4)
    F_frames = corners_t.shape[0]

    # World to camera per frame
    cam_pos = extr_t[:, :3, 3]  # (F, 3)
    R = extr_t[:, :3, :3]  # (F, 3, 3)
    diff = corners_t - cam_pos[:, None, :]  # (F, 8, 3)
    raw_cam = _torch.einsum("fvj,fjk->fvk", diff, R)  # (F, 8, 3)

    # Isaac-to-optical
    points_cam = _torch.stack(
        [-raw_cam[..., 1], -raw_cam[..., 2], raw_cam[..., 0]], dim=-1
    )  # (F, 8, 3)

    z_valid = points_cam[..., 2] > 1.0e-6  # (F, 8)

    # Project
    if static.camera_model == "fisheye":
        uv = _project_fisheye_flat(points_cam, static, device)
    else:
        uv = _project_pinhole_flat(points_cam, static, device)
    # uv: (F, 8, 2)

    # A corner is valid if it is in front of the camera (z>0). We deliberately
    # do NOT require it to land inside the image rectangle: corners that project
    # outside the frame still bound the box and are clamped to the image edge
    # afterwards. Dropping them (the old `z_valid & in_frame` behavior) shrank
    # the 2D box whenever an object extended beyond the image edge -- e.g. a
    # tall microwave whose top corners project above the frame got a box that
    # only spanned its in-frame corners, missing the part reaching the edge.
    # This matches the CPU path (compute_box2d_labels): valid = z_valid, take
    # min/max over all of them, then clamp to the image bounds.
    valid = z_valid  # (F, 8)

    # Fisheye valid mask: keep only corners inside the circular imaging region.
    if static.valid_mask is not None:
        mask_t = to_torch(static.valid_mask.astype(np.uint8), device)  # (H, W)
        u_int = uv[..., 0].round().clamp(0, W - 1).long()
        v_int = uv[..., 1].round().clamp(0, H - 1).long()
        in_circle = mask_t[v_int, u_int].bool()
        valid = valid & in_circle

    # Compute xyxy per frame
    INF = float("inf")
    u_vals = _torch.where(valid, uv[..., 0], _torch.tensor(INF, device=device))
    v_vals = _torch.where(valid, uv[..., 1], _torch.tensor(INF, device=device))
    u_neg = _torch.where(valid, uv[..., 0], _torch.tensor(-INF, device=device))
    v_neg = _torch.where(valid, uv[..., 1], _torch.tensor(-INF, device=device))

    x1 = u_vals.min(dim=1).values  # (F,)
    y1 = v_vals.min(dim=1).values
    x2 = u_neg.max(dim=1).values
    y2 = v_neg.max(dim=1).values

    any_valid = valid.any(dim=1)  # (F,)

    # Out-of-bounds check
    oob = (x2 < 0) | (y2 < 0) | (x1 >= W) | (y1 >= H)
    vis = any_valid & ~oob  # (F,)

    # Clamp
    x1 = x1.clamp(0, W - 1)
    y1 = y1.clamp(0, H - 1)
    x2 = x2.clamp(0, W - 1)
    y2 = y2.clamp(0, H - 1)

    xyxy_t = _torch.stack([x1.floor(), y1.floor(), x2.ceil(), y2.ceil()], dim=-1)  # (F, 4)
    xyxy_np = to_numpy(xyxy_t).astype(np.int32)
    vis_np = to_numpy(vis)

    # Mark invisible frames
    xyxy_np[~vis_np] = _INVISIBLE_XYXY
    return xyxy_np, vis_np


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_pinhole_flat(
    points_cam: "torch.Tensor",
    static: CameraStaticData,
    device: "torch.device",
) -> "torch.Tensor":
    """Pinhole projection for (F, 8, 3) → (F, 8, 2)."""
    z = points_cam[..., 2].clamp(min=1.0e-6)
    fx = float(static.intrinsic[0, 0])
    fy = float(static.intrinsic[1, 1])
    cx = float(static.intrinsic[0, 2])
    cy = float(static.intrinsic[1, 2])
    u = fx * points_cam[..., 0] / z + cx
    v = fy * points_cam[..., 1] / z + cy
    return _torch.stack([u, v], dim=-1)


def _project_fisheye_flat(
    points_cam: "torch.Tensor",
    static: CameraStaticData,
    device: "torch.device",
) -> "torch.Tensor":
    """Fisheye Kannala-Brandt projection for (F, 8, 3) → (F, 8, 2)."""
    X = points_cam[..., 0]
    Y = points_cam[..., 1]
    Z = points_cam[..., 2]

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
    return _torch.stack([u, v], dim=-1)


