"""Task-specific success evaluator for 42_trash_disposal.

Supports phases via params:
  - phase: "terminal" (default) — all 3 objects inside trashcan, all static, trashcan upright
  - phase: "stage" + object: "<id>" — single object inside trashcan + static + trashcan upright
  - phase: "placed" + object: "<id>" — single object inside trashcan only (for drop safety)
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


_TRASHCAN = 'obj_122_trashcan_6'

_OBJECT_INSIDE = {
    'obj_341_banana_peel_1': {
        'type': 'object_inside',
        'object': 'obj_341_banana_peel_1',
        'container': _TRASHCAN,
        'tolerance': 0.25,
        'z_offset': 100.0,
    },
    'obj_349_paper_low_2': {
        'type': 'object_inside',
        'object': 'obj_349_paper_low_2',
        'container': _TRASHCAN,
        'tolerance': 0.25,
        'z_offset': 100.0,
    },
    'obj_080_bottle_3': {
        'type': 'object_inside',
        'object': 'obj_080_bottle_3',
        'container': _TRASHCAN,
        'tolerance': 0.25,
        'z_offset': 100.0,
    },
}

_STATIC = [
    {'type': 'object_static', 'object': 'obj_341_banana_peel_1', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_349_paper_low_2', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_080_bottle_3', 'threshold': 0.05, 'check_angular': False},
]

_TRASHCAN_UPRIGHT = [
    {'type': 'object_upright', 'object': _TRASHCAN, 'tolerance_deg': 30},
]

_ALL_OBJ_IDS = list(_OBJECT_INSIDE.keys())


def _static_for(obj_id: str) -> list[dict]:
    return [c for c in _STATIC if c['object'] == obj_id]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str(params.get("phase", "terminal"))

    if phase == "terminal":
        conditions: list[dict] = []
        for obj_id in _ALL_OBJ_IDS:
            conditions.append(_OBJECT_INSIDE[obj_id])
        conditions.extend(_STATIC)
        conditions.extend(_TRASHCAN_UPRIGHT)
        return evaluate_conditions(conditions, states, ctx)

    obj_id = str(params.get("object", ""))

    if phase == "stage":
        if obj_id not in _OBJECT_INSIDE:
            raise ValueError(f"Unknown object for stage phase: {obj_id!r}")
        conditions = [_OBJECT_INSIDE[obj_id]]
        conditions.extend(_static_for(obj_id))
        conditions.extend(_TRASHCAN_UPRIGHT)
        return evaluate_conditions(conditions, states, ctx)

    if phase == "placed":
        if obj_id not in _OBJECT_INSIDE:
            raise ValueError(f"Unknown object for placed phase: {obj_id!r}")
        return evaluate_conditions([_OBJECT_INSIDE[obj_id]], states, ctx)

    raise ValueError(f"Unknown phase: {phase!r}")
