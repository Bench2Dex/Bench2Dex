"""Task-specific success evaluator for 59_living_rm_cleanup_chairs."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'joint_state',
                  'object': 'obj_095_foldingchair_1',
                  'joint': 'joint_0',
                  'target': 'closed',
                  'tolerance': 0.15},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_122_trashcan_3',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_080_bottle_2',
                                                  'container': 'obj_122_trashcan_3',
                                                  'tolerance': 0.2,
                                                  'container_center_offset': [-0.001257,
                                                                              0.0075064,
                                                                              0.0],
                                                  'object_center_offset': [-0.0012765,
                                                                           -0.0004313,
                                                                           0.0300176]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_080_bottle_2',
                                  'threshold': 0.05,
                                  'check_angular': False}]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
