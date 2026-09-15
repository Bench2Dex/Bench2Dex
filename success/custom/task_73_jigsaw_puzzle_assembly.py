"""Success evaluator for 73_jigsaw_puzzle_assembly.

Pieces 2-5 assemble around fixed piece_1 into a rectangle.
"""

from __future__ import annotations

import math
from typing import Any, Dict


StateMap = Dict[str, Any]


PUZZLE_PIECES = tuple(f"obj_326_piece_{index}" for index in range(1, 6))


_PUZZLE_TARGETS = {
    "obj_326_piece_2": (-0.065, 0.300),
    "obj_326_piece_3": (-0.150, 0.270),
    "obj_326_piece_4": (-0.150, 0.190),
    "obj_326_piece_5": (-0.065, 0.215),
}


_PUZZLE_XY_TOL = 0.020


_PUZZLE_Z_TOL = 0.010


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(object_id in states and states[object_id].get("pose_world") is not None for object_id in object_ids)


def _pos(states: StateMap, object_id: str) -> tuple[float, float, float]:
    pose = states[object_id]["pose_world"]
    return float(pose[0]), float(pose[1]), float(pose[2])


def _distance(states: StateMap, first: str, second: str, *, xy_only: bool = False) -> float:
    a = _pos(states, first)
    b = _pos(states, second)
    count = 2 if xy_only else 3
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(count)))


def _speed(state: Dict[str, Any], *, angular: bool = False) -> float:
    key = "ang_vel_world" if angular else "lin_vel_world"
    values = state.get(key, [0.0, 0.0, 0.0])
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _static(states: StateMap, *object_ids: str, linear: float = 0.06, angular: float = 0.5) -> bool:
    return all(
        _speed(states[object_id]) < linear and _speed(states[object_id], angular=True) < angular
        for object_id in object_ids
    )


def _puzzle_assembled(states: StateMap) -> bool:
    if not _has(states, *PUZZLE_PIECES):
        return False
    # All 4 outer pieces at their targets
    for piece, (tx, ty) in _PUZZLE_TARGETS.items():
        if piece not in states:
            return False
        px, py, _ = _pos(states, piece)
        if math.hypot(px - tx, py - ty) > _PUZZLE_XY_TOL:
            return False
    # Z: each outer piece within tolerance of fixed piece_1 height
    ref_z = _pos(states, "obj_326_piece_1")[2]
    for piece in _PUZZLE_TARGETS:
        pz = _pos(states, piece)[2]
        if abs(pz - ref_z) > _PUZZLE_Z_TOL:
            return False
    return _static(states, *PUZZLE_PIECES)


def _puzzle_separated(states: StateMap) -> bool:
    if not _has(states, *PUZZLE_PIECES):
        return False
    maximum = max(
        _distance(states, PUZZLE_PIECES[i], PUZZLE_PIECES[j], xy_only=True)
        for i in range(len(PUZZLE_PIECES))
        for j in range(i + 1, len(PUZZLE_PIECES))
    )
    center = "obj_326_piece_3"
    side_pieces_clear = all(
        _distance(states, piece, center, xy_only=True) > 0.20
        for piece in ("obj_326_piece_1", "obj_326_piece_2", "obj_326_piece_4", "obj_326_piece_5")
    )
    return maximum > 0.48 and side_pieces_clear and _static(states, *PUZZLE_PIECES)


def _puzzle_piece_placed(states: StateMap, piece: str) -> bool:
    """A single outer piece rests at its assembled target next to fixed piece_1."""
    if piece not in _PUZZLE_TARGETS:
        return False
    if not _has(states, piece, "obj_326_piece_1"):
        return False
    tx, ty = _PUZZLE_TARGETS[piece]
    px, py, _ = _pos(states, piece)
    if math.hypot(px - tx, py - ty) > _PUZZLE_XY_TOL:
        return False
    ref_z = _pos(states, "obj_326_piece_1")[2]
    if abs(_pos(states, piece)[2] - ref_z) > _PUZZLE_Z_TOL:
        return False
    return _static(states, piece)


def check_success(states: StateMap, ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str((params or {}).get("phase", "terminal"))
    if phase == "assembled":
        return _puzzle_assembled(states)
    if phase == "separated":
        return _puzzle_separated(states)
    if phase in _PUZZLE_TARGETS:
        return _puzzle_piece_placed(states, phase)
    # Terminal: pieces assembled into rectangle around fixed piece_1
    return _puzzle_assembled(states)
