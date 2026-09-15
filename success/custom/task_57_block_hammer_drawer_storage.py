"""Task-specific success evaluator for 57_block_hammer_drawer_storage."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'joint_state',
                  'object': 'obj_116_storagefurniture_1',
                  'joint': 'joint_0',
                  'target': 'closed',
                  'tolerance': 0.05},
                 {'type': 'all',
                  'conditions': [{'type': 'object_in_container_zone',
                                  'object': 'obj_036_wood_block_3',
                                  'container': 'obj_116_storagefurniture_1',
                                  'object_center_offset': [0.0118947, -0.0049425, 0.0514722],
                                  'container_frame': True,
                                  'container_center_offset': [-0.0173264, 0.0011487, -0.0089337],
                                  'zone_lo': [-0.1954526, -0.1585861, -0.2605339],
                                  'zone_hi': [0.1954526, 0.1585861, 0.2605339]},
                                 {'type': 'object_static',
                                  'object': 'obj_036_wood_block_3',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_in_container_zone',
                                  'object': 'obj_048_hammer_2',
                                  'container': 'obj_116_storagefurniture_1',
                                  'object_center_offset': [-0.037723, -0.022711, 0.015792],
                                  'container_frame': True,
                                  'container_center_offset': [-0.0173264, 0.0011487, -0.0089337],
                                  'zone_lo': [-0.1954526, -0.1585861, -0.2605339],
                                  'zone_hi': [0.1954526, 0.1585861, 0.2605339]},
                                 {'type': 'object_static',
                                  'object': 'obj_048_hammer_2',
                                  'threshold': 0.05,
                                  'check_angular': False}]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
