"""Success evaluator for 03_wine_glass_plate_balance.

Three identical wine glasses must be upright and each placed at one of three
target positions (within 0.05 m).  Any glass can occupy any target -- no ordering.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Tuple


GLASS_IDS = ("obj_194_wineglass_1", "obj_194_wineglass_2", "obj_194_wineglass_3")

# Three target positions on the table (x, y), matching the initial glass spawns.
TARGETS: List[Tuple[float, float]] = [
    (0.58, 0.25),
    (0.08, 0.25),
    (-0.42, 0.25),
]

TOLERANCE_M = 0.10
UPRIGHT_TOLERANCE_DEG = 30.0
LOCAL_UPRIGHT_AXIS = "y"  # rpy_deg [90, 0, 0] rotates Y into world +Z


def _pos_xy(states: Dict[str, Any], obj_id: str) -> Tuple[float, float]:
    pose = states[obj_id]["pose_world"]
    return (float(pose[0]), float(pose[1]))


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _upright(states: Dict[str, Any], obj_id: str) -> bool:
    from success.state_utils import world_axis_from_pose

    state = states.get(obj_id)
    if state is None or state.get("pose_world") is None:
        return False
    axis = world_axis_from_pose(state, LOCAL_UPRIGHT_AXIS)
    return axis[2] > math.cos(math.radians(UPRIGHT_TOLERANCE_DEG))


def _count_placed(states: Dict[str, Any]) -> int:
    """Return the max number of glasses that can be matched to unique targets
    within tolerance, considering all permutations (Hungarian for n=3)."""
    if not all(obj_id in states and states[obj_id].get("pose_world") is not None for obj_id in GLASS_IDS):
        return 0

    glass_positions = [_pos_xy(states, g) for g in GLASS_IDS]
    best = 0
    for perm in itertools.permutations(range(3)):
        count = sum(
            1 for gi, ti in enumerate(perm)
            if _dist_xy(glass_positions[gi], TARGETS[ti]) <= TOLERANCE_M
        )
        best = max(best, count)
    return best


def _glasses_static(states: Dict[str, Any]) -> bool:
    """All three glasses must be at rest.

    A glass that merely passes through a target while still rolling must NOT
    be judged placed.
    """
    from success.condition_evaluator import evaluate_conditions
    conds = [
        {"type": "object_static", "object": g, "threshold": 0.05, "check_angular": True}
        for g in GLASS_IDS
    ]
    return evaluate_conditions(conds, states, ctx={})


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str(params.get("phase", "terminal"))

    if phase == "one_placed":
        return _count_placed(states) >= 1
    if phase == "two_placed":
        return _count_placed(states) >= 2
    if phase == "three_placed":
        return _count_placed(states) >= 3

    # terminal: all 3 placed and upright AND at rest
    return (
        _count_placed(states) >= 3
        and all(_upright(states, g) for g in GLASS_IDS)
        and _glasses_static(states)
    )
