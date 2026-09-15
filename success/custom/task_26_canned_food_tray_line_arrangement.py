"""Task-specific success evaluator for 26_canned_food_tray_line_arrangement.

Supports phases via params:
  - phase: "terminal" (default) — all 4 objects on tray, all static, tray upright
  - phase: "stage" + object: "<id>" — single object on tray + static + tray upright
  - phase: "placed" + object: "<id>" — single object on tray only (for drop safety)
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


# Object-on-tray conditions with calibrated offsets
_OBJECT_ON = {
    'obj_002_master_chef_can_2': {
        'type': 'object_on',
        'object': 'obj_002_master_chef_can_2',
        'target': 'obj_130_tray_1',
        'tolerance_xy': 0.18,
        'tolerance_z': 0.15,
        'target_center_offset': [1.21e-05, 0.0189337, 0.0],
        'object_center_offset': [-0.0136388, -0.0078084, 0.0629114],
    },
    'obj_171_msg_3': {
        'type': 'object_on',
        'object': 'obj_171_msg_3',
        'target': 'obj_130_tray_1',
        'tolerance_xy': 0.18,
        'tolerance_z': 0.15,
        'target_center_offset': [1.21e-05, 0.0189337, 0.0],
        'object_center_offset': [-2e-06, 0.0475284, 2.2e-06],
    },
    'obj_151_milk_box_4': {
        'type': 'object_on',
        'object': 'obj_151_milk_box_4',
        'target': 'obj_130_tray_1',
        'tolerance_xy': 0.18,
        'tolerance_z': 0.15,
        'target_center_offset': [1.21e-05, 0.0189337, 0.0],
        'object_center_offset': [0.0, 0.0, 0.0],
    },
    'obj_010_potted_meat_can_5': {
        'type': 'object_on',
        'object': 'obj_010_potted_meat_can_5',
        'target': 'obj_130_tray_1',
        'tolerance_xy': 0.18,
        'tolerance_z': 0.15,
        'target_center_offset': [1.21e-05, 0.0189337, 0.0],
        'object_center_offset': [-0.0327845, -0.0265525, 0.0385835],
    },
}

_STATIC = [
    {'type': 'object_static', 'object': 'obj_002_master_chef_can_2', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_171_msg_3', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_151_milk_box_4', 'threshold': 0.05, 'check_angular': False},
    {'type': 'object_static', 'object': 'obj_010_potted_meat_can_5', 'threshold': 0.05, 'check_angular': False},
]

_TRAY_UPRIGHT = [
    {'type': 'object_upright', 'object': 'obj_130_tray_1', 'tolerance_deg': 30, 'local_axis': 'y'},
]

# Vertical axis of each object (the local axis that is world-up when the
# object is spawned upright).  Determined from initial-frame poses.
_OBJECT_UPRIGHT_AXIS = {
    'obj_002_master_chef_can_2': 'z',
    'obj_171_msg_3': 'y',
    'obj_151_milk_box_4': 'y',
    'obj_010_potted_meat_can_5': 'z',
}

_OBJECT_UPRIGHT_TOL_DEG = 25.0


def _upright_for(obj_id: str) -> list[dict]:
    axis = _OBJECT_UPRIGHT_AXIS.get(obj_id)
    if axis is None:
        return []
    return [
        {'type': 'object_upright', 'object': obj_id,
         'tolerance_deg': _OBJECT_UPRIGHT_TOL_DEG, 'local_axis': axis},
    ]


_ALL_OBJ_IDS = list(_OBJECT_ON.keys())


def _static_for(obj_id: str) -> list[dict]:
    return [c for c in _STATIC if c['object'] == obj_id]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str(params.get("phase", "terminal"))

    if phase == "terminal":
        conditions: list[dict] = []
        for obj_id in _ALL_OBJ_IDS:
            conditions.append(_OBJECT_ON[obj_id])
            conditions.extend(_upright_for(obj_id))
        conditions.extend(_STATIC)
        conditions.extend(_TRAY_UPRIGHT)
        return evaluate_conditions(conditions, states, ctx)

    obj_id = str(params.get("object", ""))

    if phase == "stage":
        if obj_id not in _OBJECT_ON:
            raise ValueError(f"Unknown object for stage phase: {obj_id!r}")
        conditions = [_OBJECT_ON[obj_id]]
        conditions.extend(_static_for(obj_id))
        conditions.extend(_upright_for(obj_id))
        conditions.extend(_TRAY_UPRIGHT)
        return evaluate_conditions(conditions, states, ctx)

    if phase == "placed":
        if obj_id not in _OBJECT_ON:
            raise ValueError(f"Unknown object for placed phase: {obj_id!r}")
        return evaluate_conditions([_OBJECT_ON[obj_id]], states, ctx)

    raise ValueError(f"Unknown phase: {phase!r}")
