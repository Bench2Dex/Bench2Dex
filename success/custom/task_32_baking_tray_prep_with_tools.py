"""Task-specific success evaluator for 32_baking_tray_prep_with_tools."""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'object_on',
                  'object': 'obj_008_pudding_box_2',
                  'target': 'obj_130_tray_1',
                  'tolerance_xy': 0.18,
                  'tolerance_z': 0.15,
                  'target_center_offset': [1.42e-05, 0.022275, 0.0],
                  'object_center_offset': [0.000505, 0.019123, 0.0189475]},
                 {'type': 'object_on',
                  'object': 'obj_009_gelatin_box_3',
                  'target': 'obj_130_tray_1',
                  'tolerance_xy': 0.18,
                  'tolerance_z': 0.15,
                  'target_center_offset': [1.42e-05, 0.022275, 0.0],
                  'object_center_offset': [-0.022775, -0.008245, 0.014492]},
                 {'type': 'object_on',
                  'object': 'obj_033_spatula_4',
                  'target': 'obj_130_tray_1',
                  'tolerance_xy': 0.18,
                  'tolerance_z': 0.15,
                  'target_center_offset': [1.42e-05, 0.022275, 0.0],
                  'object_center_offset': [-0.027178, -0.097023, 0.0157785]},
                 {'type': 'object_on',
                  'object': 'obj_189_brush_5',
                  'target': 'obj_130_tray_1',
                  'tolerance_xy': 0.18,
                  'tolerance_z': 0.15,
                  'target_center_offset': [1.42e-05, 0.022275, 0.0],
                  'object_center_offset': [0.0, 0.0881286, 0.0]}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
