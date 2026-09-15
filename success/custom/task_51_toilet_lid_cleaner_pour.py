"""Task-specific success evaluator for 51_toilet_lid_cleaner_pour.

Success: open toilet lid → pour cleaner into toilet (tilted toward toilet
while lid stays open) → close toilet lid.

The pour step requires: cleaner lifted + tilted toward toilet (directional
check on XY tilt lean) + near toilet bowl + lid held open — prevents false
success from wrong-direction tilt or lid closing before / after pouring.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.state_utils import get_joint_position, tilt_deg_from_pose, world_axis_from_pose


LID_OPEN_MIN = 0.2           # joint_0 > 0.2 → lid substantially open
LID_CLOSED_TARGET = -1.0     # joint_0 min = -1.0 (fully closed)
LID_CLOSED_TOL = 0.10

CLEANER_ID = "obj_021_bleach_cleanser_2"
TOILET_ID = "obj_121_toilet_1"
POUR_DIST_THRESHOLD = 0.45  # cleaner center must be near toilet bowl (accounts for tilted bottle geometry)
POUR_TILT_MIN_DEG = 50.0    # cleaner must be tilted this much to be "pouring"
POUR_LIFT_MIN_Z = 0.75      # cleaner must be lifted above the table surface
POUR_DIRECTION_COS_THRESHOLD = 0.0  # tilt direction must be within ±90° of toilet (>0 → toward, ≤0 → away/sideways)


def _lid_is_open(states: Dict[str, Any]) -> bool:
    toilet = states.get(TOILET_ID)
    if toilet is None:
        return False
    pos = get_joint_position(toilet, "joint_0", 0.0)
    return pos > LID_OPEN_MIN


def _lid_is_closed(states: Dict[str, Any]) -> bool:
    toilet = states.get(TOILET_ID)
    if toilet is None:
        return False
    pos = get_joint_position(toilet, "joint_0", 0.0)
    return abs(pos - LID_CLOSED_TARGET) < LID_CLOSED_TOL


def _cleaner_is_pouring(states: Dict[str, Any]) -> bool:
    cleaner = states.get(CLEANER_ID)
    toilet = states.get(TOILET_ID)
    if cleaner is None or toilet is None:
        return False
    cp = cleaner["pose_world"]
    # Must be lifted above table (hand-held, not sitting on table)
    if float(cp[2]) < POUR_LIFT_MIN_Z:
        return False
    tilt = tilt_deg_from_pose(cleaner)
    if tilt < POUR_TILT_MIN_DEG:
        return False
    tp = toilet["pose_world"]

    # Cleaner center must be near toilet bowl
    dist = math.hypot(float(cp[0] - tp[0]), float(cp[1] - tp[1]))
    if dist >= POUR_DIST_THRESHOLD:
        return False

    # Directional check: the bottle's local Z axis projected onto XY gives
    # the tilt lean direction.  It must point toward the toilet.
    world_z = world_axis_from_pose(cleaner, "z")
    tilt_xy = (float(world_z[0]), float(world_z[1]))
    to_toilet_xy = (float(tp[0] - cp[0]), float(tp[1] - cp[1]))
    tilt_mag = math.hypot(*tilt_xy)
    toilet_mag = math.hypot(*to_toilet_xy)
    if tilt_mag < 1e-9 or toilet_mag < 1e-9:
        return False
    cos_angle = (tilt_xy[0] * to_toilet_xy[0] + tilt_xy[1] * to_toilet_xy[1]) / (tilt_mag * toilet_mag)
    return cos_angle > POUR_DIRECTION_COS_THRESHOLD


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    mode = params.get("mode", "full")

    # -- pour_only: used by the pour_cleaner stage -------------------------
    if mode == "pour_only":
        if not ctx.get("_t51_pour_detected", False):
            if _cleaner_is_pouring(states) and _lid_is_open(states):
                ctx["_t51_pour_detected"] = True
        return bool(ctx.get("_t51_pour_detected", False))

    # -- full: 3-step sequence --------------------------------------------
    step = int(ctx.get("_t51_step", 0))

    if step == 0:  # open lid
        if _lid_is_open(states):
            ctx["_t51_step"] = 1

    if step == 1:  # pour: cleaner tilted near toilet AND lid still open
        if _cleaner_is_pouring(states) and _lid_is_open(states):
            ctx["_t51_step"] = 2

    if step == 2:  # close lid
        if _lid_is_closed(states):
            ctx["_t51_step"] = 3

    return int(ctx.get("_t51_step", 0)) >= 3
