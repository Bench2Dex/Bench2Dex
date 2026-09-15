"""Task 06 - Fruit Bowl Loading.

Success: bowl moved to center zone of table, and banana, apple_3, apple_4
all inside the bowl AND linear-static (released from hand, not translating).
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


BOWL_MOUTH_CENTER_OFFSET = [-0.02225925, -0.0655335, 0.0]
BOWL_MOUTH_RADIUS = 0.125
BOWL_UPRIGHT_TOLERANCE_DEG = 30


SUCCESS_CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'object_in_zone',
                  'object': 'obj_024_bowl_1',
                  'aabb': [[-0.5, -0.3, 0.7], [0.5, 0.3, 0.85]]},
                 {'type': 'object_upright', 'object': 'obj_024_bowl_1', 'tolerance_deg': 30},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_024_bowl_1',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_011_banana_2',
                                                  'container': 'obj_024_bowl_1',
                                                  'container_center_offset': [-0.02225925,
                                                                              -0.0655335,
                                                                              0.0],
                                                  'tolerance': 0.125,
                                                  'object_center_offset': [0.011476,
                                                                           -0.0073435,
                                                                           0.017963]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_011_banana_2',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_024_bowl_1',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_013_apple_3',
                                                  'container': 'obj_024_bowl_1',
                                                  'container_center_offset': [-0.02225925,
                                                                              -0.0655335,
                                                                              0.0],
                                                  'tolerance': 0.125,
                                                  'object_center_offset': [0.000859,
                                                                           -0.0037845,
                                                                           0.0355515]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_013_apple_3',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_024_bowl_1',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_013_apple_4',
                                                  'container': 'obj_024_bowl_1',
                                                  'container_center_offset': [-0.02225925,
                                                                              -0.0655335,
                                                                              0.0],
                                                  'tolerance': 0.125,
                                                  'object_center_offset': [0.000859,
                                                                           -0.0037845,
                                                                           0.0355515]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_013_apple_4',
                                  'threshold': 0.05,
                                  'check_angular': False}]}]}]


def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    return bool(evaluate_conditions(SUCCESS_CONDITIONS, states, ctx))
