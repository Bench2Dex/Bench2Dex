"""Task-specific success evaluator for 67_faucet_cup_water_fill.

Task description:
  Left hand: place spoon into mug (distance + upright check, spoon's long axis
             vertical, tilt < 45°).
  Right hand: place mug under faucet (XY within 0.05 m of target).
  Left hand: open faucet (joint_0 ≥ 1.5) then close it (joint_0 < 0.2).
  Right hand: place mug onto tray.
  Final: spoon still in mug.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.state_utils import get_joint_position, world_axis_from_pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(states: Dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _has(states: Dict[str, Any], *obj_ids: str) -> bool:
    return all(
        oid in states and states[oid].get("pose_world") is not None
        for oid in obj_ids
    )


def _dist_3d(states: Dict[str, Any], a: str, b: str) -> float:
    pa = _pos(states, a)
    pb = _pos(states, b)
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))


def _upright(
    state: Dict[str, Any], local_axis: str, tolerance_deg: float
) -> bool:
    """Check if *local_axis* points within *tolerance_deg* of world up (+Z)."""
    axis = world_axis_from_pose(state, local_axis)
    return axis[2] > math.cos(math.radians(tolerance_deg))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mug-under-faucet target: world XY where the mug sits under the spout.
MUG_FAUCET_TARGET_X = 0.01
MUG_FAUCET_TARGET_Y = -0.04
MUG_FAUCET_RADIUS = 0.063  # XY tolerance around the spout target

# Faucet joint thresholds (joint_0 controls water flow).
FAUCET_OPEN_THRESHOLD = 1.0   # joint_0 must reach at least this value
FAUCET_CLOSE_THRESHOLD = 0.52  # joint_0 must drop back below this value

# Spoon-in-mug thresholds.
# Spoon local X is the long axis (handle); it should point roughly upward
# when the spoon is placed in the mug.
SPOON_MUG_DIST_THRESHOLD = 0.18  # 3D distance between spoon and mug centres
SPOON_UPRIGHT_TOLERANCE_DEG = 65.0

# Mug-on-tray thresholds.
# The mug rests near the tray surface, is horizontally aligned with the tray
# centre, and must be at rest (not still descending) to count as placed.
MUG_TRAY_XY_TOLERANCE = 0.20
MUG_TRAY_Z_MIN = 0.0     # mug must be above tray surface
MUG_TRAY_Z_MAX = 0.035   # mug must be resting near the tray surface (rest dz=0.020)
MUG_TRAY_STATIC_THRESHOLD = 0.05  # mug must be at rest on the tray


# ---------------------------------------------------------------------------
# Object IDs (must match scene YAML)
# ---------------------------------------------------------------------------

SPOON = "obj_031_spoon_1"
MUG = "obj_025_mug_1"
FAUCET = "obj_094_faucet_1"
TRAY = "obj_130_tray_1"


# ---------------------------------------------------------------------------
# Phase checks
# ---------------------------------------------------------------------------

def _spoon_in_mug(states: Dict[str, Any]) -> bool:
    """Spoon is near the mug and its long axis (local X) is roughly upright."""
    if not _has(states, SPOON, MUG):
        return False
    near = _dist_3d(states, SPOON, MUG) < SPOON_MUG_DIST_THRESHOLD
    upright = _upright(states[SPOON], "x", SPOON_UPRIGHT_TOLERANCE_DEG)
    return near and upright


def _mug_under_faucet(states: Dict[str, Any]) -> bool:
    """Mug centre XY is within the spout target zone."""
    if not _has(states, MUG):
        return False
    mx, my, _ = _pos(states, MUG)
    return math.hypot(mx - MUG_FAUCET_TARGET_X, my - MUG_FAUCET_TARGET_Y) < MUG_FAUCET_RADIUS


def _faucet_cycle_done(states: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    """Faucet joint_0 has gone from closed → open → closed.

    The mug must be under the faucet opening for the whole open-close window:
    the cycle only starts when the tap opens with the mug already under the
    spout, and the mug must stay there until the tap is closed again.  This
    prevents opening the tap before placing the mug from counting as a fill.
    Progress is tracked in *ctx* so that a completed cycle stays true.
    """
    if not _has(states, FAUCET, MUG):
        return False
    joint_0 = get_joint_position(states[FAUCET], "joint_0", 0.0)
    mug_under = _mug_under_faucet(states)

    now_open = joint_0 >= FAUCET_OPEN_THRESHOLD
    prev_open = bool(ctx.get("faucet_prev_open", False))

    state = int(ctx.get("faucet_state", 0))  # 0 = waiting, 1 = open window, 2 = done
    if state == 0:
        # Rising edge: the tap starts opening. The mug must already be under.
        if now_open and not prev_open:
            state = 1
            ctx["faucet_mug_left"] = not mug_under
    elif state == 1:
        # Mug must stay under the spout for the whole open window.
        if not mug_under:
            ctx["faucet_mug_left"] = True
        if joint_0 < FAUCET_CLOSE_THRESHOLD:
            if ctx.get("faucet_mug_left", False):
                state = 0  # invalid: mug left during the open window
            else:
                state = 2  # complete
            ctx["faucet_mug_left"] = False

    ctx["faucet_prev_open"] = now_open
    ctx["faucet_state"] = state
    return state >= 2


def _mug_on_tray(states: Dict[str, Any]) -> bool:
    """Mug is resting on the tray surface (not airborne / still moving)."""
    if not _has(states, MUG, TRAY):
        return False
    mx, my, mz = _pos(states, MUG)
    tx, ty, tz = _pos(states, TRAY)
    xy_ok = math.hypot(mx - tx, my - ty) < MUG_TRAY_XY_TOLERANCE
    z_ok = MUG_TRAY_Z_MIN < (mz - tz) < MUG_TRAY_Z_MAX
    # Mug must be at rest — prevents counting it while it is still descending.
    lv = states[MUG].get("lin_vel_world", [0.0, 0.0, 0.0])
    speed = math.sqrt(sum(float(v) * float(v) for v in lv))
    static_ok = speed < MUG_TRAY_STATIC_THRESHOLD
    return xy_ok and z_ok and static_ok


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    phase = str(params.get("phase", "terminal"))

    # --- Phase-specific returns ------------------------------------------
    if phase == "spoon_in_mug":
        return _spoon_in_mug(states)
    if phase == "mug_under_faucet":
        return _mug_under_faucet(states)
    if phase == "faucet_open_close":
        return _faucet_cycle_done(states, ctx)
    if phase == "mug_on_tray":
        return _mug_on_tray(states)

    # --- Terminal: ordered sequence with latching ------------------------
    if not _has(states, SPOON, MUG, FAUCET, TRAY):
        return False

    spoon_ok = _spoon_in_mug(states)
    mug_faucet_ok = _mug_under_faucet(states)
    faucet_ok = _faucet_cycle_done(states, ctx)
    mug_tray_ok = _mug_on_tray(states)

    # Strict sequence: spoon in mug → mug under faucet → faucet cycle →
    # mug on tray.  Final frame also requires spoon still in mug.
    step = int(ctx.get("step", 0))
    if step == 0 and spoon_ok:
        step = 1
    if step == 1 and mug_faucet_ok:
        step = 2
    if step == 2 and faucet_ok:
        step = 3
    if step == 3 and mug_tray_ok and spoon_ok:
        step = 4
    ctx["step"] = step
    return step >= 4
