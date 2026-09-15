"""Task-specific success evaluator for 60_breadbasket_fast_food_loading."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_183_breadbasket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_127_french_fries_2',
                                  'container': 'obj_183_breadbasket_1',
                                  'tolerance': 0.18,
                                  'container_center_offset': [1.73e-05, 0.0633889, 0.0004565],
                                  'object_center_offset': [-2.8e-06, 0.0711898, -0.0026298]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_183_breadbasket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_128_hamburg_3',
                                  'container': 'obj_183_breadbasket_1',
                                  'tolerance': 0.18,
                                  'container_center_offset': [1.73e-05, 0.0633889, 0.0004565],
                                  'object_center_offset': [0.0, 0.0269934, -1.78e-05]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_183_breadbasket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_182_bread_4',
                                  'container': 'obj_183_breadbasket_1',
                                  'tolerance': 0.18,
                                  'container_center_offset': [1.73e-05, 0.0633889, 0.0004565],
                                  'object_center_offset': [1e-06, 0.0214334, 2.8e-06]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_183_breadbasket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_163_baguette_5',
                                  'container': 'obj_183_breadbasket_1',
                                  'tolerance': 0.18,
                                  'container_center_offset': [1.73e-05, 0.0633889, 0.0004565],
                                  'object_center_offset': [4.08e-05, 0.025, -2.3e-06]}]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
