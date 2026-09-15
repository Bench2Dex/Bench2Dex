"""Task-specific success evaluator for 43_fridge_fruit_shelf_sorting.

Supports phases via params:
  - phase: "terminal" (default) — both objects in zone + static + fridge door closed
  - phase: "open_fridge" — fridge door (joint_0) open
  - phase: "close_fridge" — fridge door (joint_0) closed
  - phase: "stage" + object: "<id>" — single object in zone + static
  - phase: "placed" + object: "<id>" — single object in zone only (for drop safety)
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


_FRIDGE = 'obj_111_refrigerator_3'

_OBJECT_IN_ZONE = {
    'obj_011_banana_2': {
        'type': 'object_in_container_zone',
        'object': 'obj_011_banana_2',
        'container': _FRIDGE,
        'object_center_offset': [0.011476, -0.0073435, 0.017963],
        'container_frame': True,
        'container_center_offset': [-0.0110169, 0.000875, -0.0118804],
        'zone_lo': [-0.1190998, -0.1026261, -0.3019416],
        'zone_hi': [0.1190998, 0.1026261, 0.3019416],
    },
    'obj_013_apple_1': {
        'type': 'object_in_container_zone',
        'object': 'obj_013_apple_1',
        'container': _FRIDGE,
        'object_center_offset': [0.000859, -0.0037845, 0.0355515],
        'container_frame': True,
        'container_center_offset': [-0.0110169, 0.000875, -0.0118804],
        'zone_lo': [-0.1190998, -0.1026261, -0.3019416],
        'zone_hi': [0.1190998, 0.1026261, 0.3019416],
    },
}

_STATIC = [
    {'type': 'object_static', 'object': 'obj_011_banana_2', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_013_apple_1', 'threshold': 0.05, 'check_angular': False},
]

# Fridge has two doors: joint_0 and joint_1. Both must be checked.
_DOOR_CLOSED = [
    {'type': 'joint_state', 'object': _FRIDGE, 'joint': 'joint_0', 'target': 'closed', 'tolerance': 0.1},
    {'type': 'joint_state', 'object': _FRIDGE, 'joint': 'joint_1', 'target': 'closed', 'tolerance': 0.1},
]

_DOOR_OPEN = [
    {'type': 'joint_state', 'object': _FRIDGE, 'joint': 'joint_0', 'target': 'open', 'tolerance': 1.22},
    {'type': 'joint_state', 'object': _FRIDGE, 'joint': 'joint_1', 'target': 'open', 'tolerance': 1.22},
]

_ALL_OBJ_IDS = list(_OBJECT_IN_ZONE.keys())


def _static_for(obj_id: str) -> list[dict]:
    return [c for c in _STATIC if c['object'] == obj_id]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str(params.get("phase", "terminal"))

    if phase == "terminal":
        conditions: list[dict] = []
        for obj_id in _ALL_OBJ_IDS:
            conditions.append(_OBJECT_IN_ZONE[obj_id])
        conditions.extend(_STATIC)
        conditions.extend(_DOOR_CLOSED)
        return evaluate_conditions(conditions, states, ctx)

    if phase == "open_fridge":
        return evaluate_conditions(_DOOR_OPEN, states, ctx)

    if phase == "close_fridge":
        return evaluate_conditions(_DOOR_CLOSED, states, ctx)

    obj_id = str(params.get("object", ""))

    if phase == "stage":
        if obj_id not in _OBJECT_IN_ZONE:
            raise ValueError(f"Unknown object for stage phase: {obj_id!r}")
        conditions = [_OBJECT_IN_ZONE[obj_id]]
        conditions.extend(_static_for(obj_id))
        return evaluate_conditions(conditions, states, ctx)

    if phase == "placed":
        if obj_id not in _OBJECT_IN_ZONE:
            raise ValueError(f"Unknown object for placed phase: {obj_id!r}")
        return evaluate_conditions([_OBJECT_IN_ZONE[obj_id]], states, ctx)

    raise ValueError(f"Unknown phase: {phase!r}")
