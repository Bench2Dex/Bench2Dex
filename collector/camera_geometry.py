"""Pure numpy camera geometry helpers shared by online and offline pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _isaac_raw_camera_to_optical(points_raw_camera: np.ndarray) -> np.ndarray:
    """Convert Isaac camera-body axes (x forward, y left, z up) to CV optical axes."""
    points_optical = np.empty_like(points_raw_camera, dtype=np.float32)
    points_optical[:, 0] = -points_raw_camera[:, 1]
    points_optical[:, 1] = -points_raw_camera[:, 2]
    points_optical[:, 2] = points_raw_camera[:, 0]
    return points_optical


@dataclass
class CameraFrame:
    rgb: np.ndarray | None
    depth_m: np.ndarray | None
    intrinsic: np.ndarray
    extrinsic_world_from_cam: np.ndarray
    cam_pos_w: np.ndarray
    cam_quat_wxyz: np.ndarray
    image_shape: tuple[int, int]
    depth_semantics: str | None = None
    camera_model: str = "pinhole"  # "pinhole" | "fisheye"
    distortion_coefficients: np.ndarray | None = None  # [k1,k2,k3,k4]
    fisheye_camera_matrix: np.ndarray | None = None  # 3x3
    fisheye_valid_mask: np.ndarray | None = None  # bool [H,W]
    clipping_range: tuple[float, float] | None = None


def is_geometry_compatible(camera_frame: CameraFrame) -> bool:
    model = str(getattr(camera_frame, "camera_model", "pinhole") or "pinhole")
    if model == "pinhole":
        return True
    if model == "fisheye":
        return (
            getattr(camera_frame, "fisheye_camera_matrix", None) is not None
            and getattr(camera_frame, "distortion_coefficients", None) is not None
        )
    return False


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0e-8))
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1.0e-8))
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1.0e-8))
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def world_points_to_camera(
    points_world: np.ndarray,
    extrinsic_world_from_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cam_pos_w = np.asarray(extrinsic_world_from_cam[:3, 3], dtype=np.float32)
    rot_world_from_cam = np.asarray(extrinsic_world_from_cam[:3, :3], dtype=np.float32)
    points_raw_camera = (points_world - cam_pos_w[None, :]) @ rot_world_from_cam
    points_cam = _isaac_raw_camera_to_optical(points_raw_camera)
    valid = points_cam[:, 2] > 1.0e-6
    return points_cam.astype(np.float32), valid


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


def project_world_points_to_image(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_from_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam, valid = world_points_to_camera(points_world, extrinsic_world_from_cam)
    uv = project_camera_points(points_cam, intrinsic)
    return uv.astype(np.float32), valid.astype(np.bool_)


# ---------------------------------------------------------------------------
# Fisheye (Kannala-Brandt equidistant) projection
# ---------------------------------------------------------------------------

def project_camera_points_fisheye(
    points_cam: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    """OpenCV fisheye equidistant (Kannala-Brandt) forward projection.

    theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    u = fx * (theta_d / r) * X + cx
    v = fy * (theta_d / r) * Y + cy

    When r->0: theta_d/r -> 1/Z (L'Hopital), degenerates to pinhole.
    Fully vectorized numpy, no iteration.

    Args:
        points_cam: (N, 3) in CV optical frame (Z forward, X right, Y down).
        camera_matrix: (3, 3) OpenCV camera matrix.
        distortion_coefficients: (4,) [k1, k2, k3, k4].
    Returns:
        uv: (N, 2) pixel coordinates.
    """
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = Z > 1.0e-6
    if not np.any(valid):
        return uv

    Xv, Yv, Zv = X[valid], Y[valid], Z[valid]
    r = np.sqrt(Xv * Xv + Yv * Yv)
    theta = np.arctan2(r, Zv)

    k1, k2, k3, k4 = distortion_coefficients[:4]
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
    scale = np.where(on_axis, 1.0 / Zv, theta_d / safe_r)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    uv[valid, 0] = fx * scale * Xv + cx
    uv[valid, 1] = fy * scale * Yv + cy
    return uv


def project_camera_points_dispatch(
    points_cam: np.ndarray,
    camera_frame: CameraFrame,
) -> np.ndarray:
    """Route to fisheye or pinhole projection based on camera_model."""
    if camera_frame.camera_model == "fisheye":
        return project_camera_points_fisheye(
            points_cam,
            camera_frame.fisheye_camera_matrix,
            camera_frame.distortion_coefficients,
        )
    return project_camera_points(points_cam, camera_frame.intrinsic)


def project_world_points_to_image_dispatch(
    points_world: np.ndarray,
    camera_frame: CameraFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """World -> image projection with automatic model dispatch."""
    points_cam, valid = world_points_to_camera(
        points_world, camera_frame.extrinsic_world_from_cam
    )
    uv = project_camera_points_dispatch(points_cam, camera_frame)
    return uv.astype(np.float32), valid.astype(np.bool_)


def compute_fisheye_valid_mask(
    height: int,
    width: int,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    max_fov_deg: float = 190.0,
) -> np.ndarray:
    """Compute the circular valid-pixel mask for a fisheye image.

    Valid region: pixels whose incident angle theta < theta_max.
    theta_max = max_fov_deg / 2 (half-angle).
    """
    theta_max = np.radians(max_fov_deg / 2.0)
    k1, k2, k3, k4 = distortion_coefficients[:4]
    tm2 = theta_max * theta_max
    r_d_max = theta_max * (1.0 + k1 * tm2 + k2 * tm2**2 + k3 * tm2**3 + k4 * tm2**4)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    v_coords, u_coords = np.mgrid[0:height, 0:width]
    r_norm = np.sqrt(((u_coords - cx) / fx) ** 2 + ((v_coords - cy) / fy) ** 2)
    return (r_norm <= float(r_d_max)).astype(np.bool_)
