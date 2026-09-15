"""Task-specific success evaluator for 27_ball_size_sorted_basket_loading."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_212_basket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_053_mini_soccer_ball_2',
                                  'container': 'obj_212_basket_1',
                                  'tolerance': 0.22,
                                  'container_center_offset': [0.0, 0.075, 0.0],
                                  'object_center_offset': [-0.0074058, 0.0132354, 0.0747246]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_212_basket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_056_tennis_ball_3',
                                  'container': 'obj_212_basket_1',
                                  'tolerance': 0.22,
                                  'container_center_offset': [0.0, 0.075, 0.0],
                                  'object_center_offset': [0.0082115, -0.044278, 0.0331315]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_212_basket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_058_golf_ball_4',
                                  'container': 'obj_212_basket_1',
                                  'tolerance': 0.22,
                                  'container_center_offset': [0.0, 0.075, 0.0],
                                  'object_center_offset': [-0.0065395, -0.0304375, 0.0209435]}]},
                 {'type': 'all',
                  'conditions': [{'type': 'object_upright',
                                  'object': 'obj_212_basket_1',
                                  'tolerance_deg': 30,
                                  'local_axis': 'y'},
                                 {'type': 'object_inside',
                                  'object': 'obj_144_table_tennis_5',
                                  'container': 'obj_212_basket_1',
                                  'tolerance': 0.22,
                                  'container_center_offset': [0.0, 0.075, 0.0],
                                  'object_center_offset': [0.0, 0.0466254, 0.0]}]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
