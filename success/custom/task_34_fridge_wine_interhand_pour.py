"""Task-specific success evaluator for 34_fridge_wine_interhand_pour.

Left hand opens fridge → takes out wine bottle → hands to right hand in the air →
right hand pours wine into glass → hands back to left hand → left hand returns
bottle to fridge and closes the door.

Two modes (controlled via ``params.mode``):

- ``pour_only`` — wine bottle is tilted toward the glass (used by the
  *pour_wine* stage).  Detection is latched.

- ``full`` (default) — sequence:
  1. open fridge
  2. take bottle out (lifted above table)
  3. pour wine toward glass
  4. return bottle to fridge
  5. close fridge
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions
from success.state_utils import get_joint_position, tilt_deg_from_pose, world_axis_from_pose


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECEIVER_DIST_THRESHOLD = 0.35
RECEIVER_ALIGN_TOL_DEG = 50.0
POUR_TILT_MIN_DEG = 50.0
LIFT_MIN_Z = 0.82
INSIDE_TOLERANCE = 0.25
JOINT_OPEN_TOLERANCE = 1.22
JOINT_CLOSED_TOLERANCE = 0.25

# When the bottle is returned to the fridge it must not be toppled over.
# It may be slightly tilted but not beyond this angle from world-up.
BOTTLE_UPRIGHT_MAX_TILT_DEG = 60.0

FRIDGE_ID = "obj_111_refrigerator_1"
BOTTLE_ID = "obj_224_wine_bottle_1"
GLASS_IDS = ("obj_194_wineglass_2",)

BOTTLE_IN_FRIDGE_CONDITION = {
    "type": "object_in_container_zone",
    "object": BOTTLE_ID,
    "container": FRIDGE_ID,
    "container_frame": True,
    "container_center_offset": [-0.046, -0.019, -0.336],
    "zone_lo": [-0.10, -0.08, -0.04],
    "zone_hi": [0.10, 0.11, 0.05],
}


# ---------------------------------------------------------------------------
# Low-level checks
# ---------------------------------------------------------------------------


def _get_pos(states: Dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _check_fridge_open(states: Dict[str, Any]) -> bool:
    fridge = states.get(FRIDGE_ID)
    if fridge is None:
        return False
    pos = get_joint_position(fridge, "joint_0", 0.0)
    return abs(pos) > JOINT_OPEN_TOLERANCE


def _check_fridge_closed(states: Dict[str, Any]) -> bool:
    fridge = states.get(FRIDGE_ID)
    if fridge is None:
        return False
    pos = get_joint_position(fridge, "joint_0", 0.0)
    return abs(pos) < JOINT_CLOSED_TOLERANCE


def _check_bottle_lifted(states: Dict[str, Any]) -> bool:
    if BOTTLE_ID not in states:
        return False
    _ox, _oy, oz = _get_pos(states, BOTTLE_ID)
    return oz > LIFT_MIN_Z


def _check_bottle_in_fridge(states: Dict[str, Any]) -> bool:
    if BOTTLE_ID not in states or FRIDGE_ID not in states:
        return False
    return evaluate_conditions([BOTTLE_IN_FRIDGE_CONDITION], states, {})


def _check_bottle_upright(states: Dict[str, Any]) -> bool:
    """Bottle local z must stay within *BOTTLE_UPRIGHT_MAX_TILT_DEG* of world up.

    The bottle is spawned upright with its local z axis aligned to world +z
    (verified from initial-frame poses).  After returning it to the fridge it
    must not be toppled over — a small tilt is tolerated but a clearly fallen
    bottle (~90 deg) fails.
    """
    if BOTTLE_ID not in states:
        return False
    tilt = tilt_deg_from_pose(states[BOTTLE_ID])
    return tilt <= BOTTLE_UPRIGHT_MAX_TILT_DEG


def _faces_target(
    source: Dict[str, Any],
    target: Dict[str, Any],
    tol_deg: float = RECEIVER_ALIGN_TOL_DEG,
) -> bool:
    sp = source["pose_world"]
    tp = target["pose_world"]
    dx = float(tp[0] - sp[0])
    dy = float(tp[1] - sp[1])
    dist_xy = math.hypot(dx, dy)
    if dist_xy <= 1.0e-6 or dist_xy > RECEIVER_DIST_THRESHOLD:
        return False

    # Try both x and z axes — different bottle orientations may use different axes
    for axis_name in ("x", "z", "-x", "-z"):
        ax, ay, _az = world_axis_from_pose(source, axis_name.lstrip("-"))
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_xy_norm = math.hypot(ax, ay)
        if axis_xy_norm <= 1.0e-6:
            continue
        dot = sign * (ax * dx + ay * dy) / (axis_xy_norm * dist_xy)
        if dot > math.cos(math.radians(tol_deg)):
            return True
    return False


def _check_pour(states: Dict[str, Any]) -> bool:
    """Return True when the bottle is tilted toward the glass."""
    bottle = states.get(BOTTLE_ID)
    if bottle is None:
        return False

    tilt = tilt_deg_from_pose(bottle)
    if tilt < POUR_TILT_MIN_DEG:
        return False

    for glass_id in GLASS_IDS:
        glass = states.get(glass_id)
        if glass is not None and _faces_target(bottle, glass):
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    """Evaluate success.

    Parameters
    ----------
    states:
        Object-id → state-dict mapping (must contain *pose_world*).
    ctx:
        Mutable context dict shared across calls for the same evaluation scope.
        Used to track sequence progress and latch pour detection.
    params:
        YAML-specified ``params`` dict.  ``mode`` can be ``"pour_only"``
        (used by the *pour_wine* stage), ``"returned"`` (used by the
        *return_bottle* stage), or omitted for the full terminal check.
    """
    mode = params.get("mode", "full")

    # -- pour_only: used by the pour_wine stage ---------------------------
    if mode == "pour_only":
        if not ctx.get("_pour_detected", False):
            if _check_pour(states):
                ctx["_pour_detected"] = True
        return bool(ctx.get("_pour_detected", False))

    # -- returned: used by the return_bottle stage ------------------------
    if mode == "returned":
        return _check_bottle_in_fridge(states) and _check_bottle_upright(states)

    # -- full: sequence --------------------------------------------
    step = int(ctx.get("_step", 0))

    if step == 0 and _check_fridge_open(states):
        ctx["_step"] = 1

    if step == 1 and _check_bottle_lifted(states):
        ctx["_step"] = 2

    if step == 2:
        if _check_pour(states):
            ctx["_step"] = 3

    if step == 3 and _check_bottle_in_fridge(states) and _check_bottle_upright(states):
        ctx["_step"] = 4

    if step == 4 and _check_fridge_closed(states):
        ctx["_step"] = 5

    return int(ctx.get("_step", 0)) >= 5
