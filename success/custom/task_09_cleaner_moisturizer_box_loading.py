"""Task-specific success evaluator for 09_cleaner_moisturizer_box_loading.

Success condition:
  - Wooden box is upright (opening facing up).
  - Cleaner (moisturizer) is inside the box AND upright (not tilted).
  - Soap is inside the box.
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


CONDITIONS = [
    {
        "type": "all",
        "conditions": [
            # Cleaner (moisturizer): inside box, box upright, cleaner upright
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_200_cleaner_2",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                    },
                    {
                        "type": "object_upright",
                        "object": "obj_200_cleaner_2",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                ],
            },
            # Soap: inside box, box upright
            {
                "type": "all",
                "conditions": [
                    {
                        "type": "object_upright",
                        "object": "obj_154_wooden_box_1",
                        "tolerance_deg": 30,
                        "local_axis": "y",
                    },
                    {
                        "type": "object_inside",
                        "object": "obj_209_soap_3",
                        "container": "obj_154_wooden_box_1",
                        "tolerance": 0.18,
                    },
                ],
            },
        ],
    }
]


def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    """Both cleaner and soap inside box; cleaner upright; box upright."""
    return evaluate_conditions(CONDITIONS, states, ctx)
