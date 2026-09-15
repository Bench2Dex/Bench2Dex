"""Task-specific success evaluator for 08 — frypan on stand + pour.

Success sequence:
  1. frypan placed on display stand
  2. soy sauce tilted toward frypan (pour)
  3. olive oil tilted toward frypan (pour)
  4. both bottles returned upright + static on the table, bread still in pan

Modes (via params.mode):
  - ``pour_soy``  — soy sauce tilted far enough (used by pour_soy_sauce stage)
  - ``pour_oil``  — olive oil tilted far enough (used by pour_olive_oil stage)
  - ``final``     — full sequence incl. bottles upright + static + bread in pan
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions
from success.state_utils import world_axis_from_pose, quat_rotate_vector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRYPAN_ID = "obj_223_chefmate_frypan_1"
STAND_ID = "obj_333_portable_stove_6"
BREAD_ID = "obj_182_bread_2"
SOY_ID = "obj_172_soy_sauce_1"
OIL_ID = "obj_146_olive_oil_1"

# Both bottles spawn upright with their local *y* axis aligned to world +z.
# tilt_deg_from_pose uses local z, which would be wrong for these assets, so
# we compute the tilt of local y instead.
POUR_TILT_MIN_DEG = 50.0      # bottle must be tilted at least this far
UPRIGHT_MAX_TILT_DEG = 15.0   # bottle considered "returned upright" below this

# Bottle mouth (highest point) in the bottle local frame.  Bottles spawn with
# local Y = world +Z (rpy [90,0,0]), so the mouth sits at local Y = height.
#   soy sauce: mesh Y range [0, 0.150]
#   olive oil: mesh Y range [0, 0.300]
SOY_MOUTH_OFFSET = (0.0, 0.150, 0.0)
OIL_MOUTH_OFFSET = (0.0, 0.300, 0.0)
STATIC_LIN_THRESHOLD = 0.05
# Bottles must be back on the table after pouring, not held in the air.
# The stove is kinematic and sits on the table, so its height tracks
# table-height generalization.  A bottle returned to the table ends near the
# stove height; one held aloft sits well above it.
BOTTLE_ON_TABLE_DZ_MAX = 0.05

# Frypan center is offset from the object origin (the handle position) along
# the frypan local z axis by ~0.27 m.  Bread is "in the pan" when its xy
# distance to that center is within the pan radius (~0.125 m).
FRYPAN_CENTER_OFFSET_Z = 0.27
FRYPAN_RADIUS = 0.125

FRYPAN_ON_STAND = {
    "type": "all",
    "conditions": [{
        "type": "object_on",
        "object": FRYPAN_ID,
        "target": STAND_ID,
        "tolerance_xy": 0.10,
        "tolerance_z": 0.18,
        "target_center_offset": [-1.42e-05, 0.039681, 0.0],
        "object_center_offset": [0.0038325, 0.0176274, 0.1674452],
    }],
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _tilt_y(state: Dict[str, Any]) -> float:
    """Angle (deg) between local *y* axis and world up (+z)."""
    ax, ay, az = world_axis_from_pose(state, "y")
    return math.degrees(math.acos(max(-1.0, min(1.0, az))))


def _pos(states: Dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _mouth_pos(states: Dict[str, Any], obj_id: str, mouth_offset: tuple) -> tuple:
    """World position of the bottle mouth (highest point of the bottle)."""
    return quat_rotate_vector(states[obj_id]["pose_world"], mouth_offset)


def _bottle_pouring(states: Dict[str, Any], obj_id: str, mouth_offset: tuple) -> bool:
    """A pour counts only when the bottle is tilted ≥50° AND its mouth stays
    within the frypan XY footprint (so the liquid lands in the pan)."""
    if obj_id not in states or FRYPAN_ID not in states:
        return False
    if _tilt_y(states[obj_id]) < POUR_TILT_MIN_DEG:
        return False
    mx, my, _mz = _mouth_pos(states, obj_id, mouth_offset)
    cx, cy = _frypan_center(states)
    return math.hypot(mx - cx, my - cy) < FRYPAN_RADIUS


def _frypan_center(states: Dict[str, Any]) -> tuple[float, float]:
    """Return (x, y) of the frypan's cooking surface centre in world frame.

    The object origin is the pan handle; the pan body is offset along the
    frypan local z axis to its cooking-surface centre.
    """
    fx, fy, _fz = _pos(states, FRYPAN_ID)
    zx, zy, _zz = world_axis_from_pose(states[FRYPAN_ID], "z")
    return fx + zx * FRYPAN_CENTER_OFFSET_Z, fy + zy * FRYPAN_CENTER_OFFSET_Z


def _stove_center(states: Dict[str, Any]) -> tuple[float, float]:
    """Stove centre in world frame (stove origin + target offset)."""
    ox, oy, _oz = _pos(states, STAND_ID)
    dx, dy, _dz = quat_rotate_vector(states[STAND_ID]["pose_world"], [-1.42e-05, 0.039681, 0.0])
    return ox + dx, oy + dy


def _frypan_on_stove(states: Dict[str, Any]) -> bool:
    """Frypan cooking-surface centre within 0.10 m of the stove centre."""
    if FRYPAN_ID not in states or STAND_ID not in states:
        return False
    fx, fy = _frypan_center(states)
    sx, sy = _stove_center(states)
    return math.hypot(fx - sx, fy - sy) < 0.10


def _bread_in_pan(states: Dict[str, Any]) -> bool:
    """Bread xy must stay within the frypan radius around the pan centre."""
    if BREAD_ID not in states or FRYPAN_ID not in states:
        return False
    bx, by, _bz = _pos(states, BREAD_ID)
    cx, cy = _frypan_center(states)
    return math.hypot(bx - cx, by - cy) < FRYPAN_RADIUS


def _bottle_upright(states: Dict[str, Any], obj_id: str) -> bool:
    return obj_id in states and _tilt_y(states[obj_id]) < UPRIGHT_MAX_TILT_DEG


def _bottle_static(states: Dict[str, Any], obj_id: str) -> bool:
    state = states.get(obj_id)
    if state is None:
        return False
    lv = state.get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))
    return speed < STATIC_LIN_THRESHOLD


def _bottle_on_table(states: Dict[str, Any], obj_id: str) -> bool:
    """Bottle must be back at the table surface (relative to the stand).

    The stand is kinematic and its height follows table-height generalisation,
    so comparing the bottle z against the stand z automatically adapts.
    """
    if obj_id not in states or STAND_ID not in states:
        return False
    _ox, _oy, oz = _pos(states, obj_id)
    _sx, _sy, sz = _pos(states, STAND_ID)
    return (oz - sz) < BOTTLE_ON_TABLE_DZ_MAX


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    mode = str(params.get("mode", "final"))

    # -- stage modes: single bottle tilted toward the pan -----------------
    if mode == "pour_soy":
        return _bottle_pouring(states, SOY_ID, SOY_MOUTH_OFFSET)
    if mode == "pour_oil":
        return _bottle_pouring(states, OIL_ID, OIL_MOUTH_OFFSET)

    # -- final: full sequence ---------------------------------------------
    step = int(ctx.get("_t08_step", 0))

    if step == 0:
        if _frypan_on_stove(states):
            ctx["_t08_step"] = 1

    if step == 1:  # soy poured
        if _bottle_pouring(states, SOY_ID, SOY_MOUTH_OFFSET):
            ctx["_t08_step"] = 2

    if step == 2:  # oil poured
        if _bottle_pouring(states, OIL_ID, OIL_MOUTH_OFFSET):
            ctx["_t08_step"] = 3

    if step == 3:  # final: both returned upright + static + on table + bread in pan
        soy_ok = (
            _bottle_upright(states, SOY_ID)
            and _bottle_static(states, SOY_ID)
            and _bottle_on_table(states, SOY_ID)
        )
        oil_ok = (
            _bottle_upright(states, OIL_ID)
            and _bottle_static(states, OIL_ID)
            and _bottle_on_table(states, OIL_ID)
        )
        if soy_ok and oil_ok and _bread_in_pan(states):
            ctx["_t08_step"] = 4

    return int(ctx.get("_t08_step", 0)) >= 4
