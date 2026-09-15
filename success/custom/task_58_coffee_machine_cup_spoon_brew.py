"""Task-specific success evaluator for 58_coffee_machine_cup_spoon_brew."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_025_mug_2',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_031_spoon_3',
                                                  'container': 'obj_025_mug_2',
                                                  'tolerance': 0.08,
                                                  'container_center_offset': [-0.008857,
                                                                              0.0173395,
                                                                              0.0],
                                                  'object_center_offset': [-0.027133,
                                                                           -0.0050495,
                                                                           0.009833]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_031_spoon_3',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'object_near',
                  'object': 'obj_025_mug_2',
                  'target': 'obj_087_coffeemachine_1',
                  'threshold': 0.2},
                 {'type': 'joint_state',
                  'object': 'obj_087_coffeemachine_1',
                  'joint': 'joint_0',
                  'target': 'open',
                  'tolerance': 0.15}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
