from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def _coerce_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> Sequence[float]:
    if isinstance(pose_or_state, dict):
        return pose_or_state["pose_world"]
    return pose_or_state


def pose_quat_wxyz(pose_or_state: Sequence[float] | dict[str, Any]) -> tuple[float, float, float, float]:
    pose = _coerce_pose(pose_or_state)
    return float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5])


def world_axis_from_pose(
    pose_or_state: Sequence[float] | dict[str, Any],
    local_axis: str = "z",
) -> tuple[float, float, float]:
    w, x, y, z = pose_quat_wxyz(pose_or_state)
    if local_axis == "x":
        return (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        )
    if local_axis == "y":
        return (
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        )
    return (
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    )


def up_dot_from_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> float:
    return float(world_axis_from_pose(pose_or_state, "z")[2])


def tilt_deg_from_pose(pose_or_state: Sequence[float] | dict[str, Any]) -> float:
    up_dot = max(-1.0, min(1.0, up_dot_from_pose(pose_or_state)))
    return float(math.degrees(math.acos(up_dot)))


def quat_rotate_vector(
    pose_world: Sequence[float] | dict[str, Any],
    local_offset: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a local-frame offset by the quaternion in *pose_world*.

    pose_world: [x, y, z, qx, qy, qz, qw] or a state dict with "pose_world".
    local_offset: [dx, dy, dz] in the object's local frame.

    Returns the world-frame position: pose_world[:3] + quat * local_offset.
    """
    pose = _coerce_pose(pose_world)
    px, py, pz = float(local_offset[0]), float(local_offset[1]), float(local_offset[2])
    qx, qy, qz, qw = float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])
    # Hamilton product: q * p * q^{-1} (pure-vector rotation)
    t0 = 2.0 * (qy * pz - qz * py)
    t1 = 2.0 * (qz * px - qx * pz)
    t2 = 2.0 * (qx * py - qy * px)
    wx = px + qw * t0 + qy * t2 - qz * t1
    wy = py + qw * t1 + qz * t0 - qx * t2
    wz = pz + qw * t2 + qx * t1 - qy * t0
    return (float(pose[0]) + wx, float(pose[1]) + wy, float(pose[2]) + wz)


def get_joint_positions(state: dict[str, Any]) -> dict[str, float]:
    qpos = state.get("qpos")
    if qpos is None:
        return {}
    if isinstance(qpos, dict):
        return {str(name): float(value) for name, value in qpos.items()}

    values = list(qpos)
    joint_names = list(state.get("joint_names") or [])
    if joint_names:
        return {
            str(name): float(values[idx])
            for idx, name in enumerate(joint_names)
            if idx < len(values)
        }
    return {f"joint_{idx}": float(value) for idx, value in enumerate(values)}


def get_joint_position(state: dict[str, Any], joint_name: str, default: float = 0.0) -> float:
    joint_positions = get_joint_positions(state)
    return float(joint_positions.get(str(joint_name), default))
