"""Success evaluator for 86_short_jigsaw_puzzle.

Short version of task 73: only pieces 2 and 3 are assembled onto the fixed
white piece_1.  Pieces 4/5 exist in the scene but are NOT part of the success
criteria.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions

_PIECES = ("obj_326_piece_2", "obj_326_piece_3")

_TARGETS = {
    "obj_326_piece_2": (-0.065, 0.300),
    "obj_326_piece_3": (-0.150, 0.270),
}
_XY_TOL = 0.020   # assembled-position tolerance
_Z_TOL = 0.010    # pieces rest on table, same height as fixed piece_1


def _pos(states: Dict[str, Any], obj_id: str) -> tuple[float, float, float]:
    pose = states[obj_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _static(states: Dict[str, Any]) -> bool:
    # A slightly relaxed threshold tolerates minor interlock vibration after
    # placement so the dwell window stays contiguous.
    conds = [
        {"type": "object_static", "object": pid, "threshold": 0.10, "check_angular": False}
        for pid in _PIECES
    ]
    return evaluate_conditions(conds, states, ctx={})


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    if not all(pid in states for pid in _PIECES):
        return False
    # XY targets
    for pid, (tx, ty) in _TARGETS.items():
        px, py, _ = _pos(states, pid)
        if math.hypot(px - tx, py - ty) > _XY_TOL:
            return False
    # Z: same table height as fixed piece_1
    ref_z = _pos(states, "obj_326_piece_1")[2]
    for pid in _PIECES:
        pz = _pos(states, pid)[2]
        if abs(pz - ref_z) > _Z_TOL:
            return False
    return _static(states)
