"""State readers for objects and articulations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to xyzw convention via numpy index swap."""
    return quat_wxyz[..., [1, 2, 3, 0]].astype(np.float32)


def _has_articulation_interface(obj: object | None) -> bool:
    return (
        obj is not None
        and hasattr(obj, "write_joint_state_to_sim")
        and hasattr(obj, "set_joint_position_target")
    )


def _articulation_data(obj: object | None) -> Any | None:
    if obj is None:
        return None
    try:
        return obj.data
    except AttributeError:
        return None


def select_robot_articulation(interactive_objects: Dict[str, object]) -> object | None:
    """Return the global robot articulation, or None.

    Only ``global_robot`` is considered a valid robot.  Falling back to an
    arbitrary articulation (e.g. a task drawer) would silently produce wrong
    joint-state data.
    """
    candidate = interactive_objects.get("global_robot")
    if _has_articulation_interface(candidate):
        return candidate
    return None


def read_joint_state(articulation: object | None, _cache: dict = {}) -> Dict | None:
    if articulation is None:
        return None
    data = _articulation_data(articulation)
    if data is None:
        return None
    # Batch joint_pos and joint_vel into a single GPU→CPU transfer
    stacked = torch.stack([data.joint_pos[0], data.joint_vel[0]], dim=0)
    stacked_np = stacked.detach().cpu().numpy()
    qpos = stacked_np[0].astype(np.float32)
    qvel = stacked_np[1].astype(np.float32)
    qeffort = None
    if hasattr(data, "applied_torque") and data.applied_torque is not None:
        qeffort = _to_numpy(data.applied_torque[0]).astype(np.float32)
    # Cache joint_names (immutable per articulation instance)
    art_id = id(articulation)
    if art_id not in _cache:
        _cache[art_id] = list(articulation.joint_names)
    return {
        "joint_names": _cache[art_id],
        "qpos": qpos,
        "qvel": qvel,
        "qeffort": qeffort,
    }


def _read_joint_payload(obj: object) -> Dict | None:
    if not _has_articulation_interface(obj):
        return None
    data = _articulation_data(obj)
    if data is None or not hasattr(data, "joint_pos") or not hasattr(data, "joint_vel"):
        return None
    payload = {
        "joint_names": list(getattr(obj, "joint_names", [])),
        "qpos": _to_numpy(data.joint_pos[0]).astype(np.float32),
        "qvel": _to_numpy(data.joint_vel[0]).astype(np.float32),
        "qeffort": None,
    }
    if hasattr(data, "applied_torque") and data.applied_torque is not None:
        payload["qeffort"] = _to_numpy(data.applied_torque[0]).astype(np.float32)
    return payload


def read_object_state(obj: object) -> Dict | None:
    data = getattr(obj, "data", None)
    if data is None:
        return None

    if hasattr(data, "root_state_w"):
        root_state = _to_numpy(data.root_state_w[0])
        if root_state.dtype != np.float32:
            root_state = root_state.astype(np.float32)
        pos = root_state[:3]
        quat_xyzw = _quat_wxyz_to_xyzw(root_state[3:7])
        lin_vel = root_state[7:10]
        ang_vel = root_state[10:13]
        pose_world = np.empty(7, dtype=np.float32)
        pose_world[:3] = pos
        pose_world[3:7] = quat_xyzw
        state = {
            "pose_world": pose_world,
            "lin_vel_world": lin_vel,
            "ang_vel_world": ang_vel,
        }
        joint_payload = _read_joint_payload(obj)
        if joint_payload is not None:
            state.update(joint_payload)
        return state

    # Fallback for deformable bodies: use nodal centroid and identity orientation.
    if hasattr(data, "nodal_pos_w") and hasattr(data, "nodal_vel_w"):
        nodal_pos = _to_numpy(data.nodal_pos_w[0]).astype(np.float32)
        nodal_vel = _to_numpy(data.nodal_vel_w[0]).astype(np.float32)
        pos = nodal_pos.mean(axis=0)
        lin_vel = nodal_vel.mean(axis=0)
        pose_world = np.concatenate([pos, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)], axis=0)
        return {
            "pose_world": pose_world,
            "lin_vel_world": lin_vel.astype(np.float32),
            "ang_vel_world": np.zeros(3, dtype=np.float32),
        }
    return None


def read_joint_limits(articulation: object | None) -> Dict | None:
    """Read runtime soft and physical joint limits without conflating them.

    ``soft_joint_pos_limits`` are controller operating limits.  Isaac Lab
    versions that expose ``joint_pos_limits`` additionally provide the asset
    physical limits.  The returned mapping keeps both sources explicit;
    callers must not label a soft limit as a mechanical hard limit.
    """
    if articulation is None:
        return None
    data = _articulation_data(articulation)
    if data is None:
        return None
    def _pair(attr_name: str):
        limits = getattr(data, attr_name, None)
        if limits is None:
            return None
        limits_np = _to_numpy(limits[0])  # shape (num_joints, 2)
        if limits_np.ndim == 2 and limits_np.shape[1] == 2:
            return (limits_np[:, 0].astype(np.float32), limits_np[:, 1].astype(np.float32))
        return None

    try:
        soft = _pair("soft_joint_pos_limits")
        hard = _pair("joint_pos_limits")
    except Exception:
        return None
    if soft is None and hard is None:
        return None
    return {
        "soft": soft or hard,
        "hard": hard,
        "soft_source": "runtime.soft_joint_pos_limits" if soft is not None else "runtime.joint_pos_limits",
        "hard_source": "runtime.joint_pos_limits" if hard is not None else None,
    }


def read_object_states(interactive_objects: Dict[str, object], object_ids: list[str]) -> Dict[str, Dict]:
    states: Dict[str, Dict] = {}
    for obj_id in object_ids:
        obj = interactive_objects.get(obj_id)
        if obj is None:
            continue
        state = read_object_state(obj)
        if state is not None:
            states[obj_id] = state
    return states





