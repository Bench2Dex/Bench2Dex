from __future__ import annotations

import math
from typing import Any, Dict

from .state_utils import get_joint_position, quat_rotate_vector, up_dot_from_pose, world_axis_from_pose


def _get_pos(states: Dict[str, Dict[str, Any]], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _get_offset_point(
    states: Dict[str, Dict[str, Any]],
    obj_id: str,
    *,
    center_key: str = "center_offset",
    local_key: str = "local_offset",
    node: Dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    if node is None:
        return _get_pos(states, obj_id)
    offset = node.get(center_key)
    if offset is None:
        offset = node.get(local_key)
    if offset is not None:
        return quat_rotate_vector(states[obj_id], offset)
    return _get_pos(states, obj_id)


def _world_delta_to_local(
    state: Dict[str, Any],
    delta: tuple[float, float, float],
) -> tuple[float, float, float]:
    pose = state["pose_world"]
    inverse_pose = [0.0, 0.0, 0.0, -float(pose[3]), -float(pose[4]), -float(pose[5]), float(pose[6])]
    return quat_rotate_vector(inverse_pose, delta)


def _dist_3d(states: Dict[str, Dict[str, Any]], a: str, b: str) -> float:
    ax, ay, az = _get_pos(states, a)
    bx, by, bz = _get_pos(states, b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _check_object_inside(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    ctr = node["container"]
    tolerance = float(node.get("tolerance", 0.12))
    z_offset = float(node.get("z_offset", 0.05))
    _ox, _oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    cx, cy, cz = _get_offset_point(
        states,
        ctr,
        center_key="container_center_offset",
        local_key="container_local_offset",
        node=node,
    )
    ok = math.hypot(_ox - cx, _oy - cy) < tolerance and oz > cz - z_offset
    max_z = node.get("max_z")
    if ok and max_z is not None:
        ok = oz < float(max_z)
    return ok


def _check_object_on(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    target = node["target"]
    tolerance_xy = float(node.get("tolerance_xy", 0.12))
    tolerance_z = float(node.get("tolerance_z", 0.10))
    ox, oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    tx, ty, tz = _get_offset_point(
        states,
        target,
        center_key="target_center_offset",
        local_key="target_local_offset",
        node=node,
    )
    return math.hypot(ox - tx, oy - ty) < tolerance_xy and 0 < (oz - tz) < tolerance_z


def _check_object_near(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    return _dist_3d(states, node["object"], node["target"]) < float(node.get("threshold", 0.25))


def _check_object_in_zone(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    ox, oy, oz = _get_pos(states, obj)
    lo, hi = node["aabb"]
    return lo[0] <= ox <= hi[0] and lo[1] <= oy <= hi[1] and lo[2] <= oz <= hi[2]


def _check_object_upright(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    tol_deg = float(node.get("tolerance_deg", 25))
    local_axis = str(node.get("local_axis", "z"))
    sign = -1.0 if local_axis.startswith("-") else 1.0
    local_axis = local_axis[1:] if local_axis.startswith("-") else local_axis
    if local_axis == "z" and sign > 0:
        up_dot = up_dot_from_pose(states[obj])
    else:
        _x, _y, up_dot = world_axis_from_pose(states[obj], local_axis)
        up_dot *= sign
    return up_dot > math.cos(math.radians(tol_deg))


def _check_object_flipped(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    tol_deg = float(node.get("tolerance_deg", 25))
    return up_dot_from_pose(states[obj]) < -math.cos(math.radians(tol_deg))


def _check_object_orientation(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    tol_deg = float(node.get("tolerance_deg", 30))
    lx, ly, lz = world_axis_from_pose(states[obj], str(node.get("local_axis", "z")))
    tx, ty, tz = node.get("target_direction", [0, 0, -1])
    tn = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
    dot = (lx * tx + ly * ty + lz * tz) / tn
    return dot > math.cos(math.radians(tol_deg))


def _check_relative_position(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    ref = node["reference"]
    relation = node["relation"]
    margin = float(node.get("margin", 0.02))
    ox, oy, oz = _get_pos(states, obj)
    rx, ry, rz = _get_pos(states, ref)
    if relation == "left":
        return oy > ry + margin
    if relation == "right":
        return oy < ry - margin
    if relation == "front":
        return ox < rx - margin
    if relation == "behind":
        return ox > rx + margin
    if relation == "above":
        return oz > rz + margin
    if relation == "below":
        return oz < rz - margin
    raise ValueError(f"Unknown relation: {relation!r}")


def _check_joint_state(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    obj = node["object"]
    joint = str(node.get("joint", "joint_0"))
    target = node["target"]
    tol = float(node.get("tolerance", 0.15))
    current = get_joint_position(states[obj], joint, 0.0)
    if target == "open":
        return abs(current) > tol
    if target == "closed":
        return abs(current) < tol
    return abs(current - float(target)) < tol


def _check_object_lifted(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    _ox, _oy, oz = _get_pos(states, node["object"])
    return oz > float(node.get("min_z", 0.80))


def _check_height_order(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    zs = [_get_pos(states, obj_id)[2] for obj_id in node["objects"]]
    return all(a <= b + 0.005 for a, b in zip(zs, zs[1:]))


def _check_object_static(
    node: Dict[str, Any],
    states: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any] | None = None,
) -> bool:
    threshold = float(node.get("threshold", 0.05))
    ang_threshold = float(node.get("ang_threshold", threshold * 5))
    obj = node["object"]
    state = states[obj]

    lv = state.get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))

    if bool(node.get("use_position_delta", False)) and ctx is not None:
        pos = _get_pos(states, obj)
        key = f"_static_prev_pos_{obj}"
        dt = float(ctx.get("dt", 1.0 / 60.0))
        prev = ctx.get(key)
        ctx[key] = pos
        if prev is not None and dt > 0:
            speed = math.sqrt(
                (pos[0] - prev[0]) ** 2
                + (pos[1] - prev[1]) ** 2
                + (pos[2] - prev[2]) ** 2
            ) / dt

    av = state.get("ang_vel_world", [0.0, 0.0, 0.0])
    ang_speed = math.sqrt(sum(float(v) * float(v) for v in av))
    if not bool(node.get("check_angular", True)):
        return speed < threshold
    return speed < threshold and ang_speed < ang_threshold


def _check_wipe_table_motion(
    node: Dict[str, Any],
    states: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
) -> bool:
    """Require sustained sponge motion close to the tabletop.

    This is a trajectory-based proxy for wiping: the sponge must remain near
    the table surface while accumulating enough XY travel.  The condition is
    evaluated only after the dispenser stage has completed.
    """
    obj = node["object"]
    x, y, z = _get_pos(states, obj)
    table_z = float(node.get("table_z", 0.75))
    min_z = float(node.get("min_z", table_z))
    max_z = float(node.get("max_z", table_z + 0.25))
    if min_z <= z <= max_z:
        key = f"_wipe_prev_xy_{obj}"
        prev = ctx.get(key)
        if prev is not None:
            ctx["_wipe_path"] = float(ctx.get("_wipe_path", 0.0)) + math.hypot(x - prev[0], y - prev[1])
        ctx[key] = (x, y)
    else:
        ctx.pop(f"_wipe_prev_xy_{obj}", None)
    return float(ctx.get("_wipe_path", 0.0)) >= float(node.get("min_path", 0.60))


def _check_tool_part_near(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    """Check if a tool's working end is near a target object.

    Unlike object_near which uses center-of-mass distance, this computes
    the world position of a tool's working end via local_offset + quaternion
    rotation, then checks 3D distance to the target.
    """
    tool_id = node["tool"]
    target_id = node["target"]
    local_offset = node["local_offset"]
    threshold = float(node.get("threshold", 0.15))
    tool_state = states[tool_id]
    target_pos = _get_pos(states, target_id)
    tool_part_pos = quat_rotate_vector(tool_state["pose_world"], local_offset)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(tool_part_pos, target_pos)))
    return dist < threshold


def _check_impact_count(node: Dict[str, Any], states: Dict[str, Dict[str, Any]], ctx: Dict[str, Any]) -> bool:
    """Detect repeated impact: tool working end near target AND approaching
    AND (optionally) PhysX contact force, with rising-edge + cooldown counting.

    Conditions checked (all must be true for a hit):
      1. Tool part near target (local_offset + distance)
      2. Tool velocity directed toward target (approach_speed > speed * approach_ratio)
      3. PhysX contact force between tool and target (if contact_force_threshold > 0
         and contact_forces data is available)

    Counting uses rising-edge detection with a cooldown to avoid double-counting
    a single physical collision due to force signal vibration.
    """
    tool_id = node["tool"]
    target_id = node["target"]
    local_offset = node.get("local_offset", [0, 0, 0])
    speed_threshold = float(node.get("speed_threshold", 0.4))
    dist_threshold = float(node.get("dist_threshold", 0.15))
    approach_ratio = float(node.get("approach_ratio", 0.3))
    contact_force_threshold = float(node.get("contact_force_threshold", 0.0))
    required_hits = int(node.get("count", 1))
    cooldown_frames = int(node.get("cooldown_frames", 10))
    key = f"_impact_{tool_id}_{target_id}"

    tool_state = states[tool_id]

    # 1. Tool part world position
    tool_part_pos = quat_rotate_vector(tool_state["pose_world"], local_offset)
    target_pos = _get_pos(states, target_id)

    # 2. Distance from tool part to target
    diff = [tool_part_pos[i] - target_pos[i] for i in range(3)]
    dist = math.sqrt(sum(d * d for d in diff))
    near = dist < dist_threshold

    # 3. Approach speed: velocity component toward target
    vel = tool_state.get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) ** 2 for v in vel))
    if dist > 1e-6 and speed > 1e-6:
        # Unit vector from tool_part toward target
        direction = [-d / dist for d in diff]
        approach_speed = sum(float(vel[i]) * direction[i] for i in range(3))
    else:
        approach_speed = 0.0
    approaching = speed > speed_threshold and approach_speed > speed * approach_ratio

    # 4. PhysX contact force check (optional, degrades gracefully)
    in_contact = True  # default: skip contact check
    if contact_force_threshold > 0:
        contact_forces = tool_state.get("contact_forces")
        if contact_forces is not None:
            pair_force = contact_forces.get(target_id, [0.0, 0.0, 0.0])
            force_norm = math.sqrt(sum(float(f) ** 2 for f in pair_force))
            in_contact = force_norm > contact_force_threshold
        else:
            # No contact data available — warn once and skip
            if not ctx.get(f"{key}_no_contact_warned"):
                import warnings
                warnings.warn(
                    f"impact_count: contact_force_threshold={contact_force_threshold} "
                    f"but no contact_forces data for {tool_id}. "
                    f"Skipping contact check. Set up ContactSensor to enable it.",
                    stacklevel=2,
                )
                ctx[f"{key}_no_contact_warned"] = True

    # 5. Rising edge + cooldown
    hitting = near and approaching and in_contact
    prev_hit = bool(ctx.get(f"{key}_hit", False))
    cooldown = int(ctx.get(f"{key}_cooldown", 0))
    if hitting and not prev_hit and cooldown <= 0:
        ctx[f"{key}_count"] = int(ctx.get(f"{key}_count", 0)) + 1
        ctx[f"{key}_cooldown"] = cooldown_frames
    else:
        ctx[f"{key}_cooldown"] = max(0, cooldown - 1)
    ctx[f"{key}_hit"] = hitting

    return int(ctx.get(f"{key}_count", 0)) >= required_hits


def _check_object_in_container_zone(node: Dict[str, Any], states: Dict[str, Dict[str, Any]]) -> bool:
    """Check if *object* is inside a zone defined relative to *container* centre.

    ``zone_lo`` / ``zone_hi`` are [dx, dy, dz] offsets from the container's
    centre.  By default the delta is interpreted in world axes; with
    ``container_frame: true`` it is rotated into the container local frame.
    This lets the zone move with the container at runtime (table-height
    changes, physics settling, generalisation).
    """
    obj = node["object"]
    container = node["container"]
    zone_lo = node["zone_lo"]
    zone_hi = node["zone_hi"]
    ox, oy, oz = _get_offset_point(
        states,
        obj,
        center_key="object_center_offset",
        local_key="object_local_offset",
        node=node,
    )
    cx, cy, cz = _get_offset_point(
        states,
        container,
        center_key="container_center_offset",
        local_key="container_local_offset",
        node=node,
    )
    dx, dy, dz = ox - cx, oy - cy, oz - cz
    if bool(node.get("container_frame", False)):
        dx, dy, dz = _world_delta_to_local(states[container], (dx, dy, dz))
    return (
        float(zone_lo[0]) <= dx <= float(zone_hi[0])
        and float(zone_lo[1]) <= dy <= float(zone_hi[1])
        and float(zone_lo[2]) <= dz <= float(zone_hi[2])
    )


def evaluate_condition_tree(node: Dict[str, Any], states: Dict[str, Dict[str, Any]], ctx: Dict[str, Any]) -> bool:
    ctype = node["type"]

    if ctype == "all":
        return all(evaluate_condition_tree(child, states, ctx) for child in node["conditions"])
    if ctype == "any":
        return any(evaluate_condition_tree(child, states, ctx) for child in node["conditions"])
    if ctype == "sequence":
        key = f"_seq_{id(node)}"
        step_idx = int(ctx.get(key, 0))
        steps = node["steps"]
        if step_idx >= len(steps):
            return True
        if evaluate_condition_tree(steps[step_idx], states, ctx):
            ctx[key] = step_idx + 1
        return int(ctx.get(key, 0)) >= len(steps)
    if ctype == "not":
        return not evaluate_condition_tree(node["condition"], states, ctx)
    if ctype == "hold_duration":
        timer_key = f"_hold_{id(node)}"
        if evaluate_condition_tree(node["condition"], states, ctx):
            ctx[timer_key] = float(ctx.get(timer_key, 0.0)) + float(ctx.get("dt", 1 / 60))
        else:
            ctx[timer_key] = 0.0
        return float(ctx[timer_key]) >= float(node["seconds"])
    if ctype == "object_inside":
        return _check_object_inside(node, states)
    if ctype == "object_on":
        return _check_object_on(node, states)
    if ctype == "object_near":
        return _check_object_near(node, states)
    if ctype == "object_in_zone":
        return _check_object_in_zone(node, states)
    if ctype == "object_in_container_zone":
        return _check_object_in_container_zone(node, states)
    if ctype == "object_upright":
        return _check_object_upright(node, states)
    if ctype == "object_flipped":
        return _check_object_flipped(node, states)
    if ctype == "object_orientation":
        return _check_object_orientation(node, states)
    if ctype == "relative_position":
        return _check_relative_position(node, states)
    if ctype == "joint_state":
        return _check_joint_state(node, states)
    if ctype == "object_lifted":
        return _check_object_lifted(node, states)
    if ctype == "height_order":
        return _check_height_order(node, states)
    if ctype == "object_static":
        return _check_object_static(node, states, ctx)
    if ctype == "wipe_table_motion":
        return _check_wipe_table_motion(node, states, ctx)
    if ctype == "tool_part_near":
        return _check_tool_part_near(node, states)
    if ctype == "impact_count":
        return _check_impact_count(node, states, ctx)
    raise ValueError(f"Unknown condition type: {ctype!r}")


def evaluate_conditions(conditions: list[Dict[str, Any]], states: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    return all(evaluate_condition_tree(node, states, ctx) for node in conditions)
