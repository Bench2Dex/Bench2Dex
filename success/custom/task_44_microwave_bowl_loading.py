"""Task-specific success evaluator for 44_microwave_bowl_loading.

Success (4-stage pipeline):
  1. open_door       — microwave door opened
  2. bowl_in_mw      — bowl inside microwave cooking cavity
  3. baguette_in_mw  — baguette inside microwave cooking cavity AND placed in bowl, linear-static
  4. close_door      — microwave door closed

Terminal condition (all must hold simultaneously for dwell_time_s):
  bowl in MW cavity + baguette in MW cavity + baguette placed in bowl + door closed.

The cavity is defined relative to the microwave's runtime position
(object_in_container_zone) so that it is robust to table-height changes
and physics settling.
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


# Cooking cavity as [dx, dy, dz] offsets from the microwave centre.
# z is compensated for bowl/baguette thickness.
ZONE_LO = [-0.22, -0.15, -0.16]
ZONE_HI = [0.12, 0.15, 0.10]
BOWL_MOUTH_CENTER_OFFSET = [-0.02225925, -0.0655335, 0.0]
BOWL_MOUTH_RADIUS = 0.125
BOWL_UPRIGHT_TOLERANCE_DEG = 30
BAGUETTE_CENTER_OFFSET = [0.0000367, 0.0225, -0.0000020]

CONDITIONS = [{'type': 'all',
  'conditions': [{'type': 'object_in_container_zone',
                  'object': 'obj_024_bowl_3',
                  'container': 'obj_104_microwave_2',
                  'zone_lo': [-0.1428254, -0.1083366, -0.1677036],
                  'zone_hi': [0.1571746, 0.2316634, 0.0922964],
                  'object_center_offset': [-0.0222592, -0.0655335, 0.0404617],
                  'container_center_offset': [-0.0071746, -0.0116634, 0.0077036],
                  'container_frame': True},
                 {'type': 'object_upright', 'object': 'obj_024_bowl_3', 'tolerance_deg': 30},
                 {'type': 'all',
                  'conditions': [{'type': 'object_in_container_zone',
                                  'object': 'obj_163_baguette_1',
                                  'container': 'obj_104_microwave_2',
                                  'zone_lo': [-0.1428254, -0.1083366, -0.1677036],
                                  'zone_hi': [0.1571746, 0.2316634, 0.0922964],
                                  'object_center_offset': [3.67e-05, 0.0225, -2e-06],
                                  'container_center_offset': [-0.0071746, -0.0116634, 0.0077036],
                                  'container_frame': True},
                                 {'type': 'all',
                                  'conditions': [{'type': 'object_upright',
                                                  'object': 'obj_024_bowl_3',
                                                  'tolerance_deg': 30},
                                                 {'type': 'object_inside',
                                                  'object': 'obj_163_baguette_1',
                                                  'container': 'obj_024_bowl_3',
                                                  'object_center_offset': [3.67e-05,
                                                                           0.0225,
                                                                           -2e-06],
                                                  'container_center_offset': [-0.02225925,
                                                                              -0.0655335,
                                                                              0.0],
                                                  'tolerance': 0.125}]},
                                 {'type': 'object_static',
                                  'object': 'obj_163_baguette_1',
                                  'threshold': 0.05,
                                  'check_angular': False}]},
                 {'type': 'joint_state',
                  'object': 'obj_104_microwave_2',
                  'joint': 'joint_0',
                  'target': 'closed',
                  'tolerance': 0.2}]}]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
