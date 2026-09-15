"""Task-specific success evaluator for 56_glasses_lamp_pen_basket_organize."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_212_basket_2',
                                                  'tolerance_deg': 30,
                                                  'local_axis': 'y'},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_092_eyeglasses_3',
                                                  'container': 'obj_212_basket_2',
                                                  'tolerance': 0.2,
                                                  'container_center_offset': [7.9e-06,
                                                                              0.1722727,
                                                                              2.7e-06],
                                                  'object_center_offset': [0.0525417,
                                                                           -0.0003858,
                                                                           -0.0072597]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_092_eyeglasses_3',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_212_basket_2',
                                                  'tolerance_deg': 30,
                                                  'local_axis': 'y'},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_101_lamp_1',
                                                  'container': 'obj_212_basket_2',
                                                  'tolerance': 0.22,
                                                  'container_center_offset': [7.9e-06,
                                                                              0.1722727,
                                                                              2.7e-06],
                                                  'object_center_offset': [-0.0190962,
                                                                           0.000457,
                                                                           0.0041273]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_101_lamp_1',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'all',
                  'conditions': [{'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_212_basket_2',
                                                  'tolerance_deg': 30,
                                                  'local_axis': 'y'},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_040_large_marker_4',
                                                  'container': 'obj_212_basket_2',
                                                  'tolerance': 0.2,
                                                  'container_center_offset': [7.9e-06,
                                                                              0.1722727,
                                                                              2.7e-06],
                                                  'object_center_offset': [-0.03569,
                                                                           -0.0050825,
                                                                           0.0091745]}]},
                                 {'type': 'object_static',
                                  'object': 'obj_040_large_marker_4',
                                  'threshold': 0.05,
                                  'check_angular': False}]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
