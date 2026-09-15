"""Task-specific success evaluator for 07_citrus_plate_loading."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'object_inside',
                  'object': 'obj_014_lemon_2',
                  'container': 'obj_029_plate_1',
                  'tolerance': 0.15,
                  'object_center_offset': [-0.0137527, 0.0281495, 0.0341555],
                  'container_center_offset': [-0.0139452, 1.92e-05, 0.0]},
                 {'type': 'object_inside',
                  'object': 'obj_014_lemon_3',
                  'container': 'obj_029_plate_1',
                  'tolerance': 0.15,
                  'object_center_offset': [-0.0137527, 0.0281495, 0.0341555],
                  'container_center_offset': [-0.0139452, 1.92e-05, 0.0]},
                 {'type': 'object_inside',
                  'object': 'obj_017_orange_4',
                  'container': 'obj_029_plate_1',
                  'tolerance': 0.15,
                  'object_center_offset': [-0.0079753, -0.021114, 0.0407422],
                  'container_center_offset': [-0.0139452, 1.92e-05, 0.0]},
                 {'type': 'object_inside',
                  'object': 'obj_017_orange_5',
                  'container': 'obj_029_plate_1',
                  'tolerance': 0.15,
                  'object_center_offset': [-0.0079753, -0.021114, 0.0407422],
                  'container_center_offset': [-0.0139452, 1.92e-05, 0.0]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
