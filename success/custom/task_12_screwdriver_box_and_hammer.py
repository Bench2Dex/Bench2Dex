"""Task 12 - Screwdriver Box and Hammer.

Success:
  1. Right screwdriver (flat) placed in the wooden box.
  2. Box moved to the left side of the table.
  3. Left screwdriver (phillips) placed in the wooden box.
  4. Hammer strikes the wood block once, then placed in the box.

The strike detector uses observable rigid-body motion: the hammer swings near
the block AND the block shows a short-window rigid-body response.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


PHILLIPS = "obj_043_phillips_screwdriver_1"
FLAT = "obj_044_flat_screwdriver_2"
HAMMER = "obj_048_hammer_3"
WOOD_BLOCK = "obj_036_wood_block_4"
BOX = "obj_154_wooden_box_1"

BOX_TOLERANCE_M = 0.18

HAMMER_BLOCK_XY_DIST_M = 0.28
HAMMER_SWING_SPEED_MPS = 0.25
BLOCK_RESPONSE_LIN_SPEED_MPS = 0.025
BLOCK_RESPONSE_ANG_SPEED_RADPS = 0.15
STRIKE_RESPONSE_WINDOW_S = 1.00
REQUIRED_STRIKES = 1
STRIKE_COOLDOWN_FRAMES = 15
FALLBACK_DT_S = 1.0 / 20.0

_SCREWDRIVER_INSIDE = [
    {
        "type": "all",
        "conditions": [
            {"type": "object_upright", "object": BOX, "tolerance_deg": 30, "local_axis": "y"},
            {"type": "object_inside", "object": PHILLIPS, "container": BOX, "tolerance": BOX_TOLERANCE_M},
        ],
    },
    {
        "type": "all",
        "conditions": [
            {"type": "object_upright", "object": BOX, "tolerance_deg": 30, "local_axis": "y"},
            {"type": "object_inside", "object": FLAT, "container": BOX, "tolerance": BOX_TOLERANCE_M},
        ],
    },
]

HAMMER_INSIDE = {
    "type": "all",
    "conditions": [
        {"type": "object_upright", "object": BOX, "tolerance_deg": 30, "local_axis": "y"},
        {"type": "object_inside", "object": HAMMER, "container": BOX, "tolerance": BOX_TOLERANCE_M},
    ],
}


def _pose(state: Dict[str, Any]):
    return state["pose_world"]


def _dist_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ap = _pose(a)
    bp = _pose(b)
    return math.hypot(float(ap[0]) - float(bp[0]), float(ap[1]) - float(bp[1]))


def _norm(values) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _dt_s(ctx: Dict[str, Any]) -> float:
    dt = float(ctx.get("dt", FALLBACK_DT_S) or FALLBACK_DT_S)
    return dt if dt > 0.0 else FALLBACK_DT_S


def _tick_time(ctx: Dict[str, Any]) -> float:
    t = float(ctx.get("_t12_time_s", 0.0)) + _dt_s(ctx)
    ctx["_t12_time_s"] = t
    return t


def _speed_mps(state: Dict[str, Any], ctx: Dict[str, Any], key: str) -> float:
    vel = state.get("lin_vel_world")
    if vel is not None:
        return _norm(vel)

    pose = _pose(state)
    prev_key = f"_t12_prev_{key}_xyz"
    prev = ctx.get(prev_key)
    ctx[prev_key] = [float(pose[0]), float(pose[1]), float(pose[2])]
    if prev is None:
        return 0.0
    dx = float(pose[0]) - float(prev[0])
    dy = float(pose[1]) - float(prev[1])
    dz = float(pose[2]) - float(prev[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz) / _dt_s(ctx)


def _angular_speed_radps(state: Dict[str, Any]) -> float:
    vel = state.get("ang_vel_world")
    if vel is None:
        return 0.0
    return _norm(vel)


def _tools_placed(states: Dict[str, Any]) -> bool:
    """Both screwdrivers are inside the box."""
    try:
        return evaluate_conditions(_SCREWDRIVER_INSIDE, states, {})
    except KeyError:
        return False


def _hammer_placed(states: Dict[str, Any]) -> bool:
    """Hammer is inside the box."""
    try:
        return evaluate_conditions([HAMMER_INSIDE], states, {})
    except KeyError:
        return False


def _update_hammer_strikes(states: Dict[str, Any], ctx: Dict[str, Any]) -> int:
    t = _tick_time(ctx)

    # Track whether screwdrivers have ever been placed (used for stage ordering).
    if _tools_placed(states):
        ctx["_t12_tools_placed"] = True

    try:
        hammer = states[HAMMER]
        block = states[WOOD_BLOCK]
    except KeyError:
        return int(ctx.get("_t12_hammer_strikes", 0))

    # Keep pose-difference speed fallbacks warm, but do not allow pre-placement
    # swings or block bumps to be counted.
    hammer_speed = _speed_mps(hammer, ctx, "hammer")
    block_lin_speed = _speed_mps(block, ctx, "block")
    if not bool(ctx.get("_t12_tools_placed", False)):
        ctx["_t12_hammer_striking"] = False
        ctx.pop("_t12_last_hammer_swing_t", None)
        ctx.pop("_t12_last_block_response_t", None)
        return int(ctx.get("_t12_hammer_strikes", 0))

    near = _dist_xy(hammer, block) < HAMMER_BLOCK_XY_DIST_M
    hammer_swing = bool(near and hammer_speed > HAMMER_SWING_SPEED_MPS)
    if hammer_swing:
        ctx["_t12_last_hammer_swing_t"] = t

    block_response = bool(
        block_lin_speed > BLOCK_RESPONSE_LIN_SPEED_MPS
        or _angular_speed_radps(block) > BLOCK_RESPONSE_ANG_SPEED_RADPS
    )
    if block_response:
        ctx["_t12_last_block_response_t"] = t

    last_swing = ctx.get("_t12_last_hammer_swing_t")
    last_response = ctx.get("_t12_last_block_response_t")
    coupled = (
        last_swing is not None
        and last_response is not None
        and abs(float(last_swing) - float(last_response)) <= STRIKE_RESPONSE_WINDOW_S
    )
    striking = bool(coupled and (hammer_swing or block_response))

    prev = bool(ctx.get("_t12_hammer_striking", False))
    cooldown = int(ctx.get("_t12_hammer_cooldown", 0))
    if striking and not prev and cooldown <= 0:
        ctx["_t12_hammer_strikes"] = int(ctx.get("_t12_hammer_strikes", 0)) + 1
        ctx["_t12_hammer_cooldown"] = STRIKE_COOLDOWN_FRAMES
    else:
        ctx["_t12_hammer_cooldown"] = max(0, cooldown - 1)
    ctx["_t12_hammer_striking"] = striking
    return int(ctx.get("_t12_hammer_strikes", 0))


def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    mode = str((params or {}).get("mode", "terminal"))
    strikes = _update_hammer_strikes(states, ctx)

    if mode == "hammer_strike":
        return strikes >= REQUIRED_STRIKES
    if mode == "hammer_placed":
        return _hammer_placed(states)

    # Terminal: both screwdrivers in box + hammer in box + hammer struck wood once
    return _tools_placed(states) and _hammer_placed(states) and strikes >= REQUIRED_STRIKES
