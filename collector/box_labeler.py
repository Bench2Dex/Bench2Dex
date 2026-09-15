"""3D/2D box label generation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .camera_geometry import CameraFrame, is_geometry_compatible, project_world_points_to_image_dispatch, quat_xyzw_to_rot


# Pre-computed sign matrix for 8 box corners: shape (8, 3)
_CORNER_SIGNS = np.array([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
], dtype=np.float32)


def _box_corners_local(size_lwh: np.ndarray) -> np.ndarray:
    return _CORNER_SIGNS * (0.5 * size_lwh)


def compute_box3d_labels(
    object_states: Dict[str, Dict],
    local_bboxes: Dict[str, tuple],
    _bbox_cache: dict = {},
) -> Dict[str, Dict]:
    labels: Dict[str, Dict] = {}
    for obj_id, state in object_states.items():
        bbox = local_bboxes.get(obj_id)
        if bbox is None:
            continue
        # Cache pre-computed bbox arrays (static per episode)
        if obj_id not in _bbox_cache or _bbox_cache[obj_id][0] is not bbox:
            lo = np.asarray(bbox[0], dtype=np.float32)
            hi = np.asarray(bbox[1], dtype=np.float32)
            size = hi - lo
            center_local = 0.5 * (lo + hi)
            corners_local = _box_corners_local(size)
            _bbox_cache[obj_id] = (bbox, size, center_local, corners_local)
        _, size, center_local, corners_local = _bbox_cache[obj_id]

        pose = state["pose_world"]
        rot = quat_xyzw_to_rot(pose[3:7])
        center_world = pose[:3] + rot @ center_local
        corners_world = (corners_local @ rot.T) + center_world
        labels[obj_id] = {
            "center_world": center_world,
            "size_lwh": size,
            "quat_world": pose[3:7],
            "corners_world": corners_world,
        }
    return labels


_INVISIBLE_XYXY = np.array([-1, -1, -1, -1], dtype=np.int32)


def compute_box2d_labels(
    box3d_labels: Dict[str, Dict],
    camera_frames: Dict[str, CameraFrame],
) -> Dict[str, Dict[str, Dict]]:
    out: Dict[str, Dict[str, Dict]] = {}
    for cam_id, cam in camera_frames.items():
        if not is_geometry_compatible(cam):
            continue
        cam_labels: Dict[str, Dict] = {}
        height, width = cam.image_shape
        for obj_id, box in box3d_labels.items():
            uv, valid = project_world_points_to_image_dispatch(
                points_world=box["corners_world"],
                camera_frame=cam,
            )

            # For fisheye cameras, exclude projected points outside the
            # circular valid imaging region to prevent inflated 2D boxes.
            fisheye_mask = getattr(cam, "fisheye_valid_mask", None)
            if fisheye_mask is not None and np.any(valid):
                u_int = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, width - 1)
                v_int = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, height - 1)
                valid = valid & fisheye_mask[v_int, u_int]

            if not np.any(valid):
                cam_labels[obj_id] = {
                    "xyxy": _INVISIBLE_XYXY.copy(),
                    "visible": False,
                }
                continue

            u = uv[valid, 0]
            v = uv[valid, 1]
            x1 = float(np.min(u))
            y1 = float(np.min(v))
            x2 = float(np.max(u))
            y2 = float(np.max(v))

            if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
                cam_labels[obj_id] = {
                    "xyxy": _INVISIBLE_XYXY.copy(),
                    "visible": False,
                }
                continue

            x1 = max(0.0, min(x1, width - 1))
            y1 = max(0.0, min(y1, height - 1))
            x2 = max(0.0, min(x2, width - 1))
            y2 = max(0.0, min(y2, height - 1))

            cam_labels[obj_id] = {
                "xyxy": np.array(
                    [int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))],
                    dtype=np.int32,
                ),
                "visible": True,
            }
        out[cam_id] = cam_labels
    return out
