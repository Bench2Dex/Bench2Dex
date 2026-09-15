"""Task 22 - Long Reach Tool Box Loading.

Success: all four tools placed inside the wooden box.
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [
    {
        'type': 'all',
        'conditions': [
            {
                'type': 'all',
                'conditions': [
                    {
                        'type': 'object_upright',
                        'object': 'obj_154_wooden_box_1',
                        'tolerance_deg': 30,
                        'local_axis': 'y',
                    },
                    {
                        'type': 'object_inside',
                        'object': 'obj_043_phillips_screwdriver_2',
                        'container': 'obj_154_wooden_box_1',
                        'tolerance': 0.18,
                        'container_center_offset': [0.0, 0.07, 0.0],
                        'object_center_offset': [-0.042698, 0.0033815, 0.0173985],
                    },
                ],
            },
            {
                'type': 'all',
                'conditions': [
                    {
                        'type': 'object_upright',
                        'object': 'obj_154_wooden_box_1',
                        'tolerance_deg': 30,
                        'local_axis': 'y',
                    },
                    {
                        'type': 'object_inside',
                        'object': 'obj_044_flat_screwdriver_3',
                        'container': 'obj_154_wooden_box_1',
                        'tolerance': 0.18,
                        'container_center_offset': [0.0, 0.07, 0.0],
                        'object_center_offset': [-0.0124925, 0.0116655, 0.0171135],
                    },
                ],
            },
            {
                'type': 'all',
                'conditions': [
                    {
                        'type': 'object_upright',
                        'object': 'obj_154_wooden_box_1',
                        'tolerance_deg': 30,
                        'local_axis': 'y',
                    },
                    {
                        'type': 'object_inside',
                        'object': 'obj_042_adjustable_wrench_4',
                        'container': 'obj_154_wooden_box_1',
                        'tolerance': 0.18,
                        'container_center_offset': [0.0, 0.07, 0.0],
                        'object_center_offset': [-0.0034875, -0.0123455, 0.008692],
                    },
                ],
            },
            {
                'type': 'all',
                'conditions': [
                    {
                        'type': 'object_upright',
                        'object': 'obj_154_wooden_box_1',
                        'tolerance_deg': 30,
                        'local_axis': 'y',
                    },
                    {
                        'type': 'object_inside',
                        'object': 'obj_147_drill_5',
                        'container': 'obj_154_wooden_box_1',
                        'tolerance': 0.18,
                        'container_center_offset': [0.0, 0.07, 0.0],
                        'object_center_offset': [0.0, 0.0900622, 0.0],
                    },
                ],
            },
        ]
    }
]


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
