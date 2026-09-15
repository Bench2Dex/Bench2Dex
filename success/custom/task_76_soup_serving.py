"""Success evaluator for 76_soup_serving.

Two ladle scoops from the pot into the bowl, ladle returned to the pot,
bowl placed upright in the serving zone.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.state_utils import quat_rotate_vector, world_axis_from_pose


StateMap = Dict[str, Any]


_LADLE_HEAD_OFFSET = (0.0224, -0.00004, 0.00148)


_POT_CENTER_XY = (0.0, 0.24)


_POT_RADIUS_XY = 0.10


_POT_Z_REL_LO = -0.02


_POT_Z_REL_HI = 0.20


_BOWL_UPRIGHT_TOL_DEG = 30.0


_BOWL_SERVE_X = (-0.56, -0.38)


_BOWL_SERVE_Y = (0.20, 0.42)


_BOWL_SERVE_Z = (0.68, 0.85)


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(object_id in states and states[object_id].get("pose_world") is not None for object_id in object_ids)


def _pos(states: StateMap, object_id: str) -> tuple[float, float, float]:
    pose = states[object_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _ladle_head_pos(states: StateMap) -> tuple:
    """World position of the ladle spoon-head centre."""
    ladle = "obj_330_ladle_3"
    state = states.get(ladle)
    if state is None:
        return None
    return quat_rotate_vector(state["pose_world"], _LADLE_HEAD_OFFSET)


def _pot_z(states: StateMap) -> float:
    """Actual pot origin Z (varies with table height generalization)."""
    return _pos(states, "obj_330_pot_1")[2]


def _head_in_pot_zone(states: StateMap) -> bool:
    hp = _ladle_head_pos(states)
    if hp is None:
        return False
    dx = hp[0] - _POT_CENTER_XY[0]
    dy = hp[1] - _POT_CENTER_XY[1]
    pz = _pot_z(states)
    return (dx * dx + dy * dy) < _POT_RADIUS_XY * _POT_RADIUS_XY and pz + _POT_Z_REL_LO <= hp[2] <= pz + _POT_Z_REL_HI


def _head_above_bowl(states: StateMap) -> bool:
    """Ladle head is horizontally near bowl centre and above it."""
    hp = _ladle_head_pos(states)
    if hp is None:
        return False
    bx, by, bz = _pos(states, "obj_330_bowl_2")
    d_xy = math.hypot(hp[0] - bx, hp[1] - by)
    return d_xy < 0.15 and hp[2] > bz + 0.02


def _bowl_upright_76(states: StateMap) -> bool:
    bowl = "obj_330_bowl_2"
    s = states.get(bowl)
    if s is None:
        return False
    _x, _y, dot = world_axis_from_pose(s, "z")
    return dot > math.cos(math.radians(_BOWL_UPRIGHT_TOL_DEG))


def _bowl_in_serve_zone(states: StateMap) -> bool:
    """Bowl is on the table in the front-right serving area."""
    bx, by, bz = _pos(states, "obj_330_bowl_2")
    return (_BOWL_SERVE_X[0] <= bx <= _BOWL_SERVE_X[1]
            and _BOWL_SERVE_Y[0] <= by <= _BOWL_SERVE_Y[1]
            and _BOWL_SERVE_Z[0] <= bz <= _BOWL_SERVE_Z[1])


def check_success(states: StateMap, ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str((params or {}).get("phase", "terminal"))
    pot = "obj_330_pot_1"
    bowl = "obj_330_bowl_2"
    if not _has(states, "obj_333_stove_1", pot, bowl, "obj_330_ladle_3"):
        return False

    # Named phases for YAML stages
    if phase == "ladle_returned":
        return _head_in_pot_zone(states)
    if phase == "bowl_placed":
        return _bowl_in_serve_zone(states)
    if phase == "bowl_upright":
        return _bowl_upright_76(states)

    # State machine (scoop1 / scoop2 / terminal)
    scoop1_done = bool(ctx.get("scoop1_done", False))
    scoop2_done = bool(ctx.get("scoop2_done", False))
    was_in_pot = bool(ctx.get("was_in_pot", _head_in_pot_zone(states)))

    if not scoop1_done:
        if was_in_pot and _head_above_bowl(states):
            scoop1_done = True
            was_in_pot = False
        elif _head_in_pot_zone(states):
            was_in_pot = True

    elif not scoop2_done:
        if was_in_pot and _head_above_bowl(states):
            scoop2_done = True
            was_in_pot = False
        elif _head_in_pot_zone(states):
            was_in_pot = True

    ctx["scoop1_done"] = scoop1_done
    ctx["scoop2_done"] = scoop2_done
    ctx["was_in_pot"] = was_in_pot

    if phase == "scoop1":
        return scoop1_done
    if phase == "scoop2":
        return scoop2_done

    # Terminal: both scoops done, ladle returned, bowl placed, bowl upright
    return (scoop1_done and scoop2_done
            and _head_in_pot_zone(states)
            and _bowl_in_serve_zone(states)
            and _bowl_upright_76(states))
